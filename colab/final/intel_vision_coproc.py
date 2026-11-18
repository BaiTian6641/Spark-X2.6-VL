"""Dual-GPU Intel Arc Pro B50 (`xpu:0` + `OpenVINO GPU.0`) Vision Co-Processor for Spark-X2.6-VL.

Offloads `Siglip2VisionModel` (`google/siglip2-so400m-patch16-naflex`, all 27 layers = 412.65M params)
onto an Intel Arc discrete GPU (`xpu:0` via `.venv-xpu` PyTorch XPU for full-defrost differentiable
training, and `OpenVINO GPU.0` for compiled inference) while `Spark-X2.6-4B-Loop48-En1B` runs
concurrently on the primary NVIDIA Blackwell CUDA GPU (`cuda:0`).

Key architectural invariants:
- Zero `eval()` or pickle over untrusted sockets: uses length-prefixed JSON headers over a private
  `0600` UNIX domain socket + raw binary tensor buffers in `/dev/shm` (or `TMPDIR` on macOS).
- Asynchronous Dual-GPU backward overlap: `_IntelXpuVisionAutogradFn.backward` dispatches
  `BACKWARD_ASYNC` without blocking `cuda:0`, so the Intel Arc GPU computes 27-layer ViT backward
  concurrently while the NVIDIA GPU advances to the next microbatch.
- Unified `.safetensors` & deterministic recovery: `_dirty_on_xpu` tracks XPU weight mutations and
  synchronizes updated BF16 vision weights back into `model.vision` before every `sample_parameters`,
  `checkpoint`, `evaluate`, and `save_pretrained`, while persisting XPU `AdamW` moments + FP32 master
  weights in `intel_vision_optim.pt`.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import torch.nn as nn

DEFAULT_RUNTIME_PREFIX = Path("/home/baitian6641/.local/intel-b50-runtime")
DEFAULT_XPU_PYTHON = Path("/home/baitian6641/Spark-X2.6-VL/.venv-xpu/bin/python")

_DTYPE_TO_CODE: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.uint8: "u8",
    torch.bool: "bool",
}
_CODE_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}


def intel_arc_runtime_env(
    base_env: dict[str, str] | None = None,
    *,
    runtime_prefix: str | Path | None = None,
) -> dict[str, str]:
    """Populate user-space Level Zero / OpenCL runtime paths for Intel Arc Pro B50 (Xe2 Battlemage)."""
    env = dict(os.environ if base_env is None else base_env)
    prefix = Path(
        runtime_prefix
        or env.get("INTEL_B50_RUNTIME_PREFIX")
        or DEFAULT_RUNTIME_PREFIX
    )
    lib64 = prefix / "root" / "usr" / "lib64"
    opencl_dir = lib64 / "intel-opencl"
    icd_dir = prefix / "icd"
    ze_lib = lib64 / "libze_intel_gpu.so.1"
    if lib64.is_dir():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        parts = [str(lib64)]
        if opencl_dir.is_dir():
            parts.append(str(opencl_dir))
        if existing_ld:
            parts.append(existing_ld)
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    if icd_dir.is_dir():
        env.setdefault("OCL_ICD_VENDORS", str(icd_dir))
    if ze_lib.is_file():
        env.setdefault("ZE_ALT_DRIVERS", str(ze_lib))
    return env


def apply_inprocess_intel_arc_env(runtime_prefix: str | Path | None = None) -> None:
    """Apply `OCL_ICD_VENDORS`/`ZE_ALT_DRIVERS` and preload user-space IGC `.so` dependencies in-process."""
    import ctypes

    env = intel_arc_runtime_env(runtime_prefix=runtime_prefix)
    for key in ("OCL_ICD_VENDORS", "ZE_ALT_DRIVERS", "LD_LIBRARY_PATH"):
        if key in env:
            os.environ[key] = env[key]
    prefix = Path(runtime_prefix or os.environ.get("INTEL_B50_RUNTIME_PREFIX") or DEFAULT_RUNTIME_PREFIX)
    lib64 = prefix / "root" / "usr" / "lib64"
    if lib64.is_dir():
        for soname in ("libiga64.so.2", "libigc.so.2", "libigdfcl.so.2", "libze_loader.so.1"):
            cand = lib64 / soname
            if cand.is_file():
                try:
                    ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _shm_base_dir() -> Path:
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def _write_tensor_pack(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    meta: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        cpu_t = tensor.detach().contiguous().cpu()
        dt_code = _DTYPE_TO_CODE.get(cpu_t.dtype)
        if dt_code is None:
            cpu_t = cpu_t.to(torch.float32)
            dt_code = "f32"
        raw = bytes(cpu_t.reshape(-1).view(torch.uint8).numpy())
        end = offset + len(raw)
        meta[name] = {"dtype": dt_code, "shape": list(cpu_t.shape), "offsets": [offset, end]}
        blobs.append(raw)
        offset = end
    hdr = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for raw in blobs:
            f.write(raw)
    os.replace(tmp, path)


def _read_tensor_pack(path: Path, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    raw_all = path.read_bytes()
    (hdr_len,) = struct.unpack("<Q", raw_all[:8])
    meta = json.loads(raw_all[8 : 8 + hdr_len].decode("utf-8"))
    body = memoryview(raw_all)[8 + hdr_len :]
    out: dict[str, torch.Tensor] = {}
    for name, info in meta.items():
        dt = _CODE_TO_DTYPE[info["dtype"]]
        shape = info["shape"]
        s, e = info["offsets"]
        if e > s:
            chunk = bytearray(body[s:e])
            t = torch.frombuffer(chunk, dtype=dt).reshape(shape)
        else:
            t = torch.empty(shape, dtype=dt)
        out[name] = t.to(device=device) if str(device) != "cpu" else t.clone()
    return out


def _send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("<I", len(raw)) + raw)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    rem = n
    while rem > 0:
        b = sock.recv(rem)
        if not b:
            raise EOFError("Intel Arc vision daemon closed connection unexpectedly")
        chunks.append(b)
        rem -= len(b)
    return b"".join(chunks)


def _recv_msg(sock: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack("<I", _recv_exact(sock, 4))
    raw = _recv_exact(sock, length)
    return json.loads(raw.decode("utf-8"))


class _IntelXpuVisionAutogradFn(torch.autograd.Function):
    """Differentiable cross-process/cross-GPU autograd bridge (`cuda:0` <-> `xpu:0`)."""

    @staticmethod
    def forward(
        ctx,
        grad_trigger: torch.Tensor,
        bridge: "IntelArcXpuVisionBridge",
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        out_device_str: str,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        slot_id, feat_cpu = bridge._rpc_forward_train(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        ctx.bridge = bridge
        ctx.slot_id = slot_id
        return feat_cpu.to(device=out_device_str, dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        bridge: IntelArcXpuVisionBridge = ctx.bridge
        slot_id: int = ctx.slot_id
        if grad_output is not None and bridge is not None:
            mb_weight = float(getattr(bridge, "active_microbatch_weight", 1.0))
            bridge._rpc_backward_async(slot_id=slot_id, grad_output=grad_output, mb_weight=mb_weight)
        return None, None, None, None, None, None, None


class IntelArcXpuVisionBridge:
    """Client managing the persistent Intel Arc Pro B50 (`xpu:0`) vision training/eval daemon."""

    def __init__(
        self,
        vision_model_path: str | Path,
        *,
        xpu_python: str | Path | None = None,
        device_spec: str = "xpu:0",
        vision_dtype: str = "bfloat16",
        weight_decay: float = 0.01,
        grad_checkpoint_token_threshold: int = 2304,
        startup_timeout_s: float = 120.0,
    ) -> None:
        self.vision_model_path = Path(vision_model_path).resolve()
        cand_py = Path(xpu_python or os.environ.get("SPARK_XPU_PYTHON") or DEFAULT_XPU_PYTHON)
        if not cand_py.is_file():
            # Fallback to current interpreter (e.g. CPU unit test mode on macOS/CI).
            cand_py = Path(sys.executable)
            if device_spec.startswith("xpu"):
                device_spec = "cpu"
        self.xpu_python = cand_py
        self.device_spec = device_spec
        self.vision_dtype = vision_dtype
        self.weight_decay = float(weight_decay)
        self.grad_checkpoint_token_threshold = int(grad_checkpoint_token_threshold)
        self.active_microbatch_weight: float = 1.0
        self._dirty_on_xpu: bool = False
        self._trainable: bool = False
        self._unfreeze_last_n_layers: int = 0
        self._last_step_stats: dict[str, Any] = {}

        base = _shm_base_dir()
        self.ipc_dir = Path(tempfile.mkdtemp(prefix="spark_b50_xpu_", dir=str(base)))
        os.chmod(self.ipc_dir, 0o700)
        # Keep UNIX socket path short (< 100 chars on Linux/macOS).
        sock_root = Path("/tmp") if Path("/tmp").is_dir() else base
        self.sock_path = sock_root / f"sp_b50_{os.getpid()}_{id(self) & 0xFFFFF:x}.sock"
        self.req_pack = self.ipc_dir / "req.pack"
        self.resp_pack = self.ipc_dir / "resp.pack"
        self.grad_ring_size = 8
        self._grad_seq = 0
        self._slot_counter = 0

        env = intel_arc_runtime_env()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing_py}" if existing_py else repo_root
        env["TOKENIZERS_PARALLELISM"] = "false"

        cmd = [
            str(self.xpu_python),
            "-m",
            "colab.final.intel_vision_coproc",
            "--daemon",
            "--socket",
            str(self.sock_path),
            "--ipc-dir",
            str(self.ipc_dir),
            "--vision-path",
            str(self.vision_model_path),
            "--device",
            str(self.device_spec),
            "--dtype",
            str(self.vision_dtype),
            "--weight-decay",
            str(self.weight_decay),
            "--grad-ckpt-tokens",
            str(self.grad_checkpoint_token_threshold),
        ]
        self.log_path = self.ipc_dir / "daemon.log"
        self._log_fh = self.log_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

        deadline = time.monotonic() + float(startup_timeout_s)
        self._sock: socket.socket | None = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._log_fh.flush()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"Intel Arc XPU vision daemon exited early (code={self._proc.returncode}):\n{tail}"
                )
            if self.sock_path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(str(self.sock_path))
                    self._sock = s
                    break
                except OSError:
                    s.close()
            time.sleep(0.05)

        if self._sock is None:
            self.close()
            raise TimeoutError("Timed out waiting for Intel Arc XPU vision daemon socket")

        self.device_info = self._rpc({"cmd": "PING"})
        atexit.register(self.close)

    def _rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("Intel Arc XPU vision bridge is closed")
        _send_msg(self._sock, req)
        resp = _recv_msg(self._sock)
        if not resp.get("ok", False):
            raise RuntimeError(f"Intel Arc XPU vision daemon error: {resp.get('error', 'unknown')}")
        return resp

    def sync_from_host(self, vision_module: nn.Module) -> None:
        state = {k: v.detach().cpu() for k, v in vision_module.state_dict().items()}
        _write_tensor_pack(self.req_pack, state)
        self._rpc({"cmd": "SYNC_FROM_HOST", "pack": str(self.req_pack)})
        self._dirty_on_xpu = False

    def sync_to_host(self, vision_module: nn.Module, *, force: bool = False) -> bool:
        if not force and not self._dirty_on_xpu:
            return False
        self._rpc({"cmd": "SYNC_TO_HOST", "pack": str(self.resp_pack)})
        tensors = _read_tensor_pack(self.resp_pack, device="cpu")
        with torch.no_grad():
            param_map = dict(vision_module.named_parameters())
            buf_map = dict(vision_module.named_buffers())
            for name, cpu_t in tensors.items():
                target = param_map.get(name)
                if target is None:
                    target = buf_map.get(name)
                if target is not None:
                    target.data.copy_(cpu_t.to(device=target.device, dtype=target.dtype))
        self._dirty_on_xpu = False
        return True

    def configure_trainability(self, *, trainable: bool, unfreeze_last_n_layers: int = 27) -> dict[str, Any]:
        self._trainable = bool(trainable)
        self._unfreeze_last_n_layers = int(unfreeze_last_n_layers) if trainable else 0
        resp = self._rpc(
            {
                "cmd": "SET_TRAINABLE",
                "trainable": self._trainable,
                "unfreeze_last_n_layers": self._unfreeze_last_n_layers,
            }
        )
        self.device_info.update(
            trainable_parameters=resp.get("trainable_parameters", 0),
            total_parameters=resp.get("total_parameters", 0),
            unfrozen_layers=resp.get("unfrozen_layers", 0),
        )
        return resp

    def _rpc_forward_train(
        self,
        *,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        self._slot_counter = (self._slot_counter + 1) & 0x3FFFFFFF
        slot_id = self._slot_counter
        _write_tensor_pack(
            self.req_pack,
            {
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            },
        )
        self._rpc(
            {
                "cmd": "FORWARD_TRAIN",
                "slot_id": slot_id,
                "req_pack": str(self.req_pack),
                "resp_pack": str(self.resp_pack),
            }
        )
        out = _read_tensor_pack(self.resp_pack, device="cpu")["last_hidden_state"]
        return slot_id, out

    def _rpc_backward_async(
        self,
        *,
        slot_id: int,
        grad_output: torch.Tensor,
        mb_weight: float = 1.0,
    ) -> None:
        if self._sock is None:
            return
        ring_idx = self._grad_seq % self.grad_ring_size
        self._grad_seq += 1
        grad_pack = self.ipc_dir / f"grad_{ring_idx}.pack"
        _write_tensor_pack(grad_pack, {"grad_output": grad_output.detach().to(dtype=torch.bfloat16)})
        # Fire-and-forget over FIFO UNIX stream socket! Daemon executes ViT backward on `xpu:0`
        # concurrently while `.venv-blackwell` moves on to the next microbatch; daemon sends no reply
        # for `BACKWARD_ASYNC`, and any error is surfaced on the next synchronous RPC (`FORWARD*`/`STEP`).
        _send_msg(
            self._sock,
            {
                "cmd": "BACKWARD_ASYNC",
                "slot_id": int(slot_id),
                "grad_pack": str(grad_pack),
                "mb_weight": float(mb_weight),
            },
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        training_mode: bool,
        out_device: str | torch.device,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        pv = batch["pixel_values"]
        pam = batch["pixel_attention_mask"]
        ss = batch["spatial_shapes"]
        if training_mode and self._trainable and torch.is_grad_enabled():
            trigger = torch.zeros(1, device=out_device, dtype=torch.float32, requires_grad=True)
            return _IntelXpuVisionAutogradFn.apply(
                trigger,
                self,
                pv,
                pam,
                ss,
                str(out_device),
                out_dtype,
            )
        _write_tensor_pack(
            self.req_pack,
            {
                "pixel_values": pv,
                "pixel_attention_mask": pam,
                "spatial_shapes": ss,
            },
        )
        self._rpc(
            {
                "cmd": "FORWARD_EVAL",
                "req_pack": str(self.req_pack),
                "resp_pack": str(self.resp_pack),
            }
        )
        out = _read_tensor_pack(self.resp_pack, device="cpu")["last_hidden_state"]
        return out.to(device=out_device, dtype=out_dtype)

    def zero_grad(self) -> None:
        if self._sock is not None:
            self._rpc({"cmd": "ZERO_GRAD"})

    def step(
        self,
        *,
        lr: float,
        max_grad_norm: float = 1.0,
        post_scale: float = 1.0,
    ) -> dict[str, Any]:
        if not self._trainable:
            return {"applied": False, "finite": True, "vision_grad_norm": 0.0}
        resp = self._rpc(
            {
                "cmd": "STEP",
                "lr": float(lr),
                "max_grad_norm": float(max_grad_norm),
                "post_scale": float(post_scale),
            }
        )
        if resp.get("applied", False):
            self._dirty_on_xpu = True
        self._last_step_stats = resp
        return resp

    def optim_state_dict(self) -> dict[str, Any]:
        opt_file = self.ipc_dir / "xpu_optim_export.pt"
        self._rpc({"cmd": "SAVE_OPTIM", "path": str(opt_file)})
        if not opt_file.is_file():
            return {}
        return torch.load(opt_file, map_location="cpu", weights_only=True)

    def load_optim_state_dict(self, state: dict[str, Any]) -> None:
        opt_file = self.ipc_dir / "xpu_optim_import.pt"
        torch.save(dict(state) if isinstance(state, dict) else {}, opt_file)
        self._rpc({"cmd": "LOAD_OPTIM", "path": str(opt_file)})

    def telemetry(self) -> dict[str, Any]:
        return {
            **self.device_info,
            "trainable_active": self._trainable,
            "unfreeze_last_n_layers": self._unfreeze_last_n_layers,
            "dirty_on_xpu": self._dirty_on_xpu,
            "last_step": self._last_step_stats,
        }

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                _send_msg(sock, {"cmd": "SHUTDOWN"})
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        log_fh = getattr(self, "_log_fh", None)
        if log_fh is not None and not log_fh.closed:
            log_fh.close()
        sock_path = getattr(self, "sock_path", None)
        if sock_path is not None:
            with_suppress = sock_path
            try:
                with_suppress.unlink(missing_ok=True)
            except Exception:
                pass
        ipc_dir = getattr(self, "ipc_dir", None)
        if ipc_dir is not None and ipc_dir.exists():
            shutil.rmtree(ipc_dir, ignore_errors=True)


class OpenVinoArcVisionRunner:
    """OpenVINO 2026+ hardware-accelerated SigLIP2 vision runner on Intel Arc (`GPU.0`)."""

    def __init__(
        self,
        vision_module: nn.Module,
        *,
        preferred_device: str = "GPU.0",
        cache_dir: str | Path | None = None,
        max_cached_shapes: int = 12,
    ) -> None:
        apply_inprocess_intel_arc_env()
        import openvino as ov

        self._ov = ov
        self.vision_module = vision_module
        self.core = ov.Core()
        if cache_dir is not None:
            cdir = Path(cache_dir)
            cdir.mkdir(parents=True, exist_ok=True)
            self.core.set_property({"CACHE_DIR": str(cdir)})
        avail = list(self.core.available_devices)
        if preferred_device in avail:
            self.device_name = preferred_device
        elif "GPU" in avail:
            self.device_name = "GPU"
        else:
            self.device_name = "CPU"
        try:
            self.full_device_name = str(self.core.get_property(self.device_name, "FULL_DEVICE_NAME"))
        except Exception:
            self.full_device_name = self.device_name
        self.max_cached_shapes = max(1, int(max_cached_shapes))
        self._compiled_cache: dict[tuple[Any, ...], Any] = {}

    def invalidate(self) -> None:
        self._compiled_cache.clear()

    def _compile_for_key(
        self,
        key: tuple[Any, ...],
        pv_f32: torch.Tensor,
        pam_i64: torch.Tensor,
        ss_i64: torch.Tensor,
    ) -> Any:
        if key in self._compiled_cache:
            return self._compiled_cache[key]
        if len(self._compiled_cache) >= self.max_cached_shapes:
            oldest = next(iter(self._compiled_cache))
            del self._compiled_cache[oldest]

        cpu_vis = self.vision_module.eval()

        class _SingleShapeWrap(nn.Module):
            def __init__(self, m: nn.Module):
                super().__init__()
                self.m = m

            def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
                return self.m(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                ).last_hidden_state

        orig_dtypes = {k: p.dtype for k, p in cpu_vis.named_parameters()}
        needs_cast = any(dt != torch.float32 for dt in orig_dtypes.values())
        try:
            if needs_cast:
                cpu_vis.to(dtype=torch.float32)
            with torch.no_grad():
                ov_model = self._ov.convert_model(
                    _SingleShapeWrap(cpu_vis),
                    example_input=(pv_f32, pam_i64, ss_i64),
                )
        finally:
            if needs_cast:
                first_dt = next(iter(orig_dtypes.values()))
                cpu_vis.to(dtype=first_dt)

        compile_props = {"INFERENCE_PRECISION_HINT": "f16"} if self.device_name.startswith("GPU") else {}
        compiled = self.core.compile_model(ov_model, self.device_name, compile_props)
        self._compiled_cache[key] = compiled
        return compiled

    @torch.no_grad()
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        out_device: str | torch.device,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        pv = batch["pixel_values"].detach().to(device="cpu", dtype=torch.float32).contiguous()
        pam = batch["pixel_attention_mask"].detach().to(device="cpu", dtype=torch.int64).contiguous()
        ss = batch["spatial_shapes"].detach().to(device="cpu", dtype=torch.int64).contiguous()
        shapes_tuple = tuple((int(r[0]), int(r[1])) for r in ss.tolist())
        key = (int(pv.shape[0]), int(pv.shape[1]), shapes_tuple)
        compiled = self._compile_for_key(key, pv, pam, ss)
        raw_out = compiled([pv.numpy(), pam.numpy(), ss.numpy()])[0]
        return torch.from_numpy(raw_out).to(device=out_device, dtype=out_dtype)


def _run_xpu_daemon(args: argparse.Namespace) -> None:
    from transformers import Siglip2VisionConfig, Siglip2VisionModel

    req_dev = str(args.device)
    if req_dev.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device(req_dev)
        dev_type = "xpu"
        props = torch.xpu.get_device_properties(device)
        dev_name = str(props.name)
        total_mem = int(props.total_memory)
    elif req_dev.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(req_dev)
        dev_type = "cuda"
        props = torch.cuda.get_device_properties(device)
        dev_name = str(props.name)
        total_mem = int(props.total_memory)
    else:
        device = torch.device("cpu")
        dev_type = "cpu"
        dev_name = "CPUFallback"
        total_mem = 0

    vis_dtype = getattr(torch, str(args.dtype), torch.bfloat16)
    if dev_type == "cpu" and vis_dtype == torch.float16:
        vis_dtype = torch.float32

    vis_cfg = Siglip2VisionConfig.from_pretrained(args.vision_path, local_files_only=True)
    vision = Siglip2VisionModel(vis_cfg).to(device=device, dtype=vis_dtype)
    if hasattr(getattr(vision, "vision_model", None), "head"):
        del vision.vision_model.head
    if hasattr(vision, "vision_model"):
        vision.vision_model.use_head = False
    vision.eval()

    total_params = sum(p.numel() for p in vision.parameters())
    trainable_params = 0
    unfrozen_layers = 0
    grad_ckpt_tokens = int(args.grad_ckpt_tokens)
    weight_decay = float(args.weight_decay)
    beta1, beta2, eps = 0.9, 0.95, 1e-8

    # Persistent FP32 gradient accumulation & AdamW state on `xpu:0` (`~4.9 GiB` total for 412.6M params,
    # leaving `~11 GiB` free on the 16 GB Intel Arc Pro B50 for activations!).
    grad_accums: dict[str, torch.Tensor] = {}
    optim_state: dict[str, dict[str, Any]] = {}
    saved_slots: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]] = {}
    pending_async_error: str | None = None
    has_accumulated_grads: bool = False

    def _get_rng():
        if dev_type == "xpu":
            return torch.xpu.get_rng_state(device)
        if dev_type == "cuda":
            return torch.cuda.get_rng_state(device)
        return torch.get_rng_state()

    def _set_rng(st):
        if st is None:
            return
        if dev_type == "xpu":
            torch.xpu.set_rng_state(st, device)
        elif dev_type == "cuda":
            torch.cuda.set_rng_state(st, device)
        else:
            torch.set_rng_state(st)

    def _sync_dev():
        if dev_type == "xpu":
            torch.xpu.synchronize(device)
        elif dev_type == "cuda":
            torch.cuda.synchronize(device)

    def _peak_bytes() -> int:
        if dev_type == "xpu":
            return int(torch.xpu.max_memory_allocated(device))
        if dev_type == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
        return 0

    def _configure_trainable(enable: bool, last_n: int) -> dict[str, Any]:
        nonlocal trainable_params, unfrozen_layers, has_accumulated_grads
        saved_slots.clear()
        has_accumulated_grads = False
        vision.requires_grad_(False)
        vmod = getattr(vision, "vision_model", vision)
        enc_layers = getattr(getattr(vmod, "encoder", None), "layers", None)
        n_total_layers = len(enc_layers) if enc_layers is not None else 0
        if enable and last_n > 0:
            if last_n >= n_total_layers:
                vision.requires_grad_(True)
                unfrozen_layers = n_total_layers
            else:
                if enc_layers is not None:
                    for lyr in enc_layers[-int(last_n) :]:
                        lyr.requires_grad_(True)
                post_ln = getattr(vmod, "post_layernorm", None)
                if post_ln is not None:
                    post_ln.requires_grad_(True)
                unfrozen_layers = int(last_n)
            vision.train()
        else:
            unfrozen_layers = 0
            vision.eval()

        trainable_names = {n for n, p in vision.named_parameters() if p.requires_grad}
        for stale in list(grad_accums.keys()):
            if stale not in trainable_names:
                del grad_accums[stale]
        for n, p in vision.named_parameters():
            if p.requires_grad and n not in grad_accums:
                grad_accums[n] = torch.zeros_like(p, dtype=torch.float32, device=device)
            elif n in grad_accums:
                grad_accums[n].zero_()
        trainable_params = sum(p.numel() for p in vision.parameters() if p.requires_grad)
        return {
            "ok": True,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "unfrozen_layers": unfrozen_layers,
        }

    sock_path = Path(args.socket)
    sock_path.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    srv.listen(1)
    conn, _ = srv.accept()

    try:
        while True:
            try:
                msg = _recv_msg(conn)
            except EOFError:
                break
            cmd = msg.get("cmd")
            if cmd == "SHUTDOWN":
                break

            # Surface any asynchronous backward failure on the very next synchronous RPC!
            if cmd != "BACKWARD_ASYNC" and pending_async_error is not None:
                err_txt = pending_async_error
                pending_async_error = None
                _send_msg(conn, {"ok": False, "error": f"Async XPU backward failed: {err_txt}"})
                continue

            if cmd == "BACKWARD_ASYNC":
                grad_pack = Path(msg["grad_pack"])
                slot_id = int(msg["slot_id"])
                mb_weight = float(msg.get("mb_weight", 1.0))
                try:
                    slot = saved_slots.pop(slot_id, None)
                    if slot is not None and trainable_params > 0:
                        pv, pam, ss, rng_st = slot
                        g_dict = _read_tensor_pack(grad_pack, device=device)
                        grad_out = g_dict["grad_output"].to(device=device, dtype=vis_dtype)
                        use_ckpt = int(pv.shape[0]) * int(pv.shape[1]) >= grad_ckpt_tokens
                        if use_ckpt and getattr(vision, "supports_gradient_checkpointing", False):
                            vision.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs={"use_reentrant": False}
                            )
                        try:
                            _set_rng(rng_st)
                            vision.zero_grad(set_to_none=True)
                            amp_ctx = (
                                torch.autocast(dev_type, dtype=vis_dtype)
                                if dev_type in ("xpu", "cuda")
                                else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
                            )
                            with torch.enable_grad(), amp_ctx:
                                out = vision(
                                    pixel_values=pv,
                                    pixel_attention_mask=pam,
                                    spatial_shapes=ss,
                                ).last_hidden_state
                                loss_surrogate = (out.float() * grad_out.float()).sum()
                            loss_surrogate.backward()
                            with torch.no_grad():
                                for n, p in vision.named_parameters():
                                    if p.requires_grad and p.grad is not None:
                                        grad_accums[n].add_(p.grad.detach().float(), alpha=mb_weight)
                                        p.grad = None
                            has_accumulated_grads = True
                        finally:
                            if use_ckpt and hasattr(vision, "gradient_checkpointing_disable"):
                                vision.gradient_checkpointing_disable()
                        del pv, pam, ss, grad_out
                except Exception as exc:
                    pending_async_error = f"{type(exc).__name__}: {exc}"
                finally:
                    grad_pack.unlink(missing_ok=True)
                continue

            try:
                if cmd == "PING":
                    _send_msg(
                        conn,
                        {
                            "ok": True,
                            "device": str(device),
                            "device_name": dev_name,
                            "total_memory_bytes": total_mem,
                            "total_parameters": total_params,
                            "trainable_parameters": trainable_params,
                            "torch_version": torch.__version__,
                        },
                    )
                elif cmd == "SYNC_FROM_HOST":
                    tensors = _read_tensor_pack(Path(msg["pack"]), device="cpu")
                    with torch.no_grad():
                        param_map = dict(vision.named_parameters())
                        buf_map = dict(vision.named_buffers())
                        for name, cpu_t in tensors.items():
                            target = param_map.get(name)
                            if target is None:
                                target = buf_map.get(name)
                            if target is not None:
                                target.data.copy_(cpu_t.to(device=device, dtype=target.dtype))
                            if name in optim_state and "master_weight" in optim_state[name]:
                                optim_state[name]["master_weight"].copy_(cpu_t.to(device=device, dtype=torch.float32))
                    _sync_dev()
                    _send_msg(conn, {"ok": True})
                elif cmd == "SYNC_TO_HOST":
                    _sync_dev()
                    state = {k: v.detach().cpu() for k, v in vision.state_dict().items()}
                    _write_tensor_pack(Path(msg["pack"]), state)
                    _send_msg(conn, {"ok": True})
                elif cmd == "SET_TRAINABLE":
                    res = _configure_trainable(
                        bool(msg.get("trainable", False)),
                        int(msg.get("unfreeze_last_n_layers", 27)),
                    )
                    _send_msg(conn, res)
                elif cmd in ("FORWARD_EVAL", "FORWARD_TRAIN"):
                    inp = _read_tensor_pack(Path(msg["req_pack"]), device=device)
                    pv = inp["pixel_values"].to(device=device, dtype=vis_dtype)
                    pam = inp["pixel_attention_mask"].to(device=device)
                    ss = inp["spatial_shapes"].to(device=device)
                    rng_st = _get_rng() if cmd == "FORWARD_TRAIN" else None
                    was_train = vision.training
                    if cmd == "FORWARD_EVAL":
                        vision.eval()
                    amp_ctx = (
                        torch.autocast(dev_type, dtype=vis_dtype)
                        if dev_type in ("xpu", "cuda")
                        else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
                    )
                    with torch.no_grad(), amp_ctx:
                        feat = vision(
                            pixel_values=pv,
                            pixel_attention_mask=pam,
                            spatial_shapes=ss,
                        ).last_hidden_state
                    if cmd == "FORWARD_EVAL" and was_train:
                        vision.train()
                    if cmd == "FORWARD_TRAIN":
                        saved_slots[int(msg["slot_id"])] = (pv, pam, ss, rng_st)
                    _write_tensor_pack(Path(msg["resp_pack"]), {"last_hidden_state": feat})
                    _send_msg(conn, {"ok": True})
                elif cmd == "ZERO_GRAD":
                    saved_slots.clear()
                    has_accumulated_grads = False
                    for g in grad_accums.values():
                        g.zero_()
                    vision.zero_grad(set_to_none=True)
                    _send_msg(conn, {"ok": True})
                elif cmd == "STEP":
                    saved_slots.clear()
                    if not has_accumulated_grads or trainable_params <= 0:
                        _send_msg(
                            conn,
                            {
                                "ok": True,
                                "applied": False,
                                "finite": True,
                                "vision_grad_norm": 0.0,
                                "xpu_peak_bytes": _peak_bytes(),
                            },
                        )
                        continue
                    lr = float(msg["lr"])
                    max_norm = float(msg.get("max_grad_norm", 1.0))
                    post_scale = float(msg.get("post_scale", 1.0))
                    with torch.no_grad():
                        total_sq = torch.zeros((), device=device, dtype=torch.float32)
                        finite_ok = True
                        for g in grad_accums.values():
                            if post_scale != 1.0:
                                g.mul_(post_scale)
                            if not torch.isfinite(g).all():
                                finite_ok = False
                                break
                            total_sq = total_sq + g.square().sum()
                        if not finite_ok:
                            for g in grad_accums.values():
                                g.zero_()
                            has_accumulated_grads = False
                            _send_msg(
                                conn,
                                {
                                    "ok": True,
                                    "applied": False,
                                    "finite": False,
                                    "vision_grad_norm": float("nan"),
                                    "xpu_peak_bytes": _peak_bytes(),
                                },
                            )
                            continue
                        gnorm = float(total_sq.sqrt().item())
                        clip_coef = min(1.0, max_norm / (gnorm + 1e-12)) if max_norm > 0.0 else 1.0
                        for n, p in vision.named_parameters():
                            if not p.requires_grad or n not in grad_accums:
                                continue
                            g = grad_accums[n]
                            if clip_coef < 1.0:
                                g.mul_(clip_coef)
                            st = optim_state.get(n)
                            if st is None:
                                st = {
                                    "step": 0,
                                    "exp_avg": torch.zeros_like(p, dtype=torch.float32, device=device),
                                    "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32, device=device),
                                    "master_weight": p.detach().to(device=device, dtype=torch.float32).clone(),
                                }
                                optim_state[n] = st
                            st["step"] += 1
                            t = int(st["step"])
                            exp_avg = st["exp_avg"]
                            exp_avg_sq = st["exp_avg_sq"]
                            master = st["master_weight"]
                            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                            bc1 = 1.0 - beta1**t
                            bc2 = 1.0 - beta2**t
                            step_sz = lr / bc1
                            denom_t = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                            no_decay = p.ndim < 2 or n.endswith(".bias") or "norm" in n.lower()
                            wd = 0.0 if no_decay else weight_decay
                            if wd != 0.0:
                                master.mul_(1.0 - lr * wd)
                            master.addcdiv_(exp_avg, denom_t, value=-step_sz)
                            p.data.copy_(master.to(dtype=p.dtype))
                            g.zero_()
                        has_accumulated_grads = False
                        _sync_dev()
                        _send_msg(
                            conn,
                            {
                                "ok": True,
                                "applied": True,
                                "finite": True,
                                "vision_grad_norm": gnorm,
                                "clip_coef": clip_coef,
                                "xpu_peak_bytes": _peak_bytes(),
                            },
                        )
                elif cmd == "SAVE_OPTIM":
                    cpu_opt = {
                        n: {
                            "step": int(st["step"]),
                            "exp_avg": st["exp_avg"].detach().cpu(),
                            "exp_avg_sq": st["exp_avg_sq"].detach().cpu(),
                            "master_weight": st["master_weight"].detach().cpu(),
                        }
                        for n, st in optim_state.items()
                    }
                    torch.save(cpu_opt, Path(msg["path"]))
                    _send_msg(conn, {"ok": True})
                elif cmd == "LOAD_OPTIM":
                    raw_opt = torch.load(Path(msg["path"]), map_location="cpu", weights_only=True)
                    optim_state.clear()
                    if isinstance(raw_opt, dict):
                        for n, st in raw_opt.items():
                            optim_state[n] = {
                                "step": int(st["step"]),
                                "exp_avg": st["exp_avg"].to(device=device, dtype=torch.float32),
                                "exp_avg_sq": st["exp_avg_sq"].to(device=device, dtype=torch.float32),
                                "master_weight": st["master_weight"].to(device=device, dtype=torch.float32),
                            }
                    _send_msg(conn, {"ok": True})
                else:
                    _send_msg(conn, {"ok": False, "error": f"Unknown daemon command: {cmd!r}"})
            except Exception as exc:
                _send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()
        srv.close()
        sock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", action="store_true", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--ipc-dir", required=True)
    parser.add_argument("--vision-path", required=True)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-ckpt-tokens", type=int, default=2304)
    args = parser.parse_args()
    _run_xpu_daemon(args)


if __name__ == "__main__":
    main()
