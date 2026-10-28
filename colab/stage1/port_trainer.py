"""Make a separate CUDA entrypoint; never edit the historical ROCm trainer."""
import ast
import hashlib
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('ROCm trainer changed; the reviewed CUDA transformation no longer applies')
    return source.replace(old, new, 1)


def materialize(project, minimum_gpu_gib=90, require_reviewed_gpu=True):
    project = Path(project)
    original = project / 'scripts/train_multimodal_studio.py'
    source = original.read_text()
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    source = replace_once(source, 'Bounded full-parameter Spark vision experiment through installed Studio.',
                          'CUDA adaptation of the reviewed full-parameter Spark vision trainer.')
    if require_reviewed_gpu:
        guard = f'''    properties = torch.cuda.get_device_properties(0)
    if ('RTX PRO 6000' not in properties.name.upper() or 'BLACKWELL' not in properties.name.upper()
            or properties.total_memory < {int(minimum_gpu_gib)} * 1024**3 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('The observed G4 qualification target is not assigned')'''
    else:
        guard = f'''    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < {int(minimum_gpu_gib)} * 1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            'Full-parameter Spark training needs at least {int(minimum_gpu_gib)} GiB on one CUDA GPU '
            'with native BF16; this assignment reports '
            + str(properties.total_memory) + ' bytes')'''
    source = replace_once(source, '    assert torch.cuda.is_available() and torch.version.hip', '''    if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError('This entrypoint requires NVIDIA CUDA, not the historical ROCm runtime')
''' + guard)
    source = replace_once(source, "    result['hip']=torch.version.hip", "    result['hip']=torch.version.hip\n    result['cuda']=torch.version.cuda")
    source = replace_once(source, "        'versions':result['versions']}",
                          "        'versions':result['versions'],'cuda':torch.version.cuda,\n        'python':sys.version.split()[0],'gpu_capability':list(torch.cuda.get_device_capability(0))}")
    source = replace_once(source, "        saved_sources=json.loads((Path(args.resume_from)/'experiment_sources.json').read_text())", '''        saved_sources=json.loads((Path(args.resume_from)/'experiment_sources.json').read_text())
        previous_contract=saved_sources.get('training_contract',{})
        for binding in ('source_hashes','versions','cuda','python','gpu_capability'):
            if previous_contract.get(binding)!=training_contract.get(binding):
                raise ValueError('CUDA optimizer resume binding differs: '+binding)''')
    ast.parse(source)
    target = project / 'scripts/train_multimodal_cuda.py'
    if target.exists() and target.read_text() != source:
        raise ValueError('Refusing to overwrite a changed CUDA entrypoint')
    if not target.exists():
        target.write_text(source)
    if hashlib.sha256(original.read_bytes()).hexdigest() != before:
        raise AssertionError('Historical trainer changed')
    return {'original_rocm_sha256': before, 'cuda_entrypoint_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'cuda_entrypoint': str(target), 'original_unchanged': True}
