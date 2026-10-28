"""The small launcher must fetch exactly the reviewed immutable source."""
import ast,hashlib,json,re,zipfile
from pathlib import Path
import nbformat


def test_notebook_schema_and_every_code_cell_parses_without_outputs():
    path=Path('colab/Spark_Formal_Full_Finetune.ipynb'); nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    assert nb.metadata.spark_formal.training_executed is False
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert not cell.outputs and cell.execution_count is None
            ast.parse(cell.source)


def test_downloaded_source_archive_is_exact_and_contains_no_training_artifacts():
    nb=json.load(open('colab/Spark_Formal_Full_Finetune.ipynb')); code=''.join(nb['cells'][1]['source']); tree=ast.parse(code)
    literals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'SOURCE_SHA256','SOURCE_COMMIT','SOURCE_URL','SOURCE_BYTES'}}
    assert re.fullmatch('[0-9a-f]{40}',literals['SOURCE_COMMIT'])
    assert literals['SOURCE_URL']==f"https://raw.githubusercontent.com/BaiTian6641/Spark-X2.6-VL/{literals['SOURCE_COMMIT']}/releases/spark-formal-runtime.zip"
    assert Path('colab/Spark_Formal_Full_Finetune.ipynb').stat().st_size<64*1024
    assert 'base64' not in code and 'b64decode' not in code
    archive=Path('output/colab-formal-final/spark-formal-runtime.zip')
    if not archive.exists(): archive=Path('releases/spark-formal-runtime.zip')
    payload=archive.read_bytes()
    assert len(payload)==literals['SOURCE_BYTES']
    assert hashlib.sha256(payload).hexdigest()==literals['SOURCE_SHA256']
    with zipfile.ZipFile(__import__('io').BytesIO(payload)) as z:
        m=json.loads(z.read('SOURCE_MANIFEST.json'))
        assert len(m['files'])==len({f['path'] for f in m['files']})
        for f in m['files']:
            name=f['path']; content=z.read(name)
            assert not name.endswith(('.pt','.safetensors','.ipynb','.env'))
            assert '/research/' not in name and '/data/' not in name and '/private-upload-state/' not in name
            assert len(content)==f['bytes'] and hashlib.sha256(content).hexdigest()==f['sha256']
            assert content==Path(name).read_bytes(), 'GitHub runtime bundle is stale relative to reviewed source'


def test_bootstrap_cell_verifies_before_writing_and_refuses_modified_source(tmp_path,monkeypatch):
    """Execute the actual launcher cell against the shipped archive, without networking."""
    import io,urllib.request
    nb=json.load(open('colab/Spark_Formal_Full_Finetune.ipynb'))
    code=''.join(nb['cells'][1]['source']).replace("pathlib.Path('/content/spark-formal-source')",f'pathlib.Path({str(tmp_path)!r})')
    archive=Path('output/colab-formal-final/spark-formal-runtime.zip')
    if not archive.exists(): archive=Path('releases/spark-formal-runtime.zip')
    payload=archive.read_bytes()
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(payload))
    state={}; exec(compile(code,'launcher','exec'),state)
    source=state['SOURCE_ROOT']; script=source/'colab/final/run.py'
    assert script.read_bytes()==Path('colab/final/run.py').read_bytes()
    script.write_text('local change must not be overwritten')
    import pytest
    with pytest.raises(AssertionError,match='Changed source'):
        exec(compile(code,'launcher','exec'),{})
    assert script.read_text()=='local change must not be overwritten'
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(payload[:-1]+b'x'))
    with pytest.raises(RuntimeError,match='SHA256 mismatch'):
        exec(compile(code,'launcher','exec'),{})


def test_runtime_archive_has_every_local_import_and_instructor_imports_in_isolation(tmp_path):
    import importlib.util,subprocess,sys
    archive=Path('output/colab-formal-final/spark-formal-runtime.zip')
    if not archive.exists(): archive=Path('releases/spark-formal-runtime.zip')
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        for name in sorted(n for n in names if n.endswith('.py')):
            module=name[:-3].replace('/','.')
            package=module.removesuffix('.__init__') if name.endswith('/__init__.py') else module.rsplit('.',1)[0]
            for node in ast.walk(ast.parse(z.read(name))):
                modules=[]
                if isinstance(node,ast.ImportFrom):
                    modules=[importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module]
                elif isinstance(node,ast.Import): modules=[alias.name for alias in node.names]
                for required in modules:
                    if required and (required=='colab' or required.startswith('colab.')):
                        prefix=required.replace('.','/')
                        assert prefix+'.py' in names or prefix+'/__init__.py' in names, f'{name} needs missing runtime {required}'
        z.extractall(tmp_path)
    code="import sys; sys.path.insert(0,sys.argv[1]); import colab.final.teacher as t; assert t.MODEL=='mimo-v2.6-pro'; assert callable(t.costs)"
    subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],cwd=tmp_path,check=True)


def test_export_runtime_keeps_instructor_and_environment_setup_dependencies(tmp_path):
    import subprocess,sys
    from types import SimpleNamespace
    from colab.final.engine import copy_reload_assets
    copy_reload_assets(SimpleNamespace(paths={}),tmp_path)
    runtime=tmp_path/'runtime'
    for name in ('colab/formal/teacher_readiness.py','colab/final/bootstrap.lock.json','colab/final/requirements-preflight.lock','colab/stage1/requirements.in'):
        assert (runtime/name).is_file()
    subprocess.run([sys.executable,'-I','-c',"import sys; sys.path.insert(0,sys.argv[1]); import colab.final.teacher",str(runtime)],cwd=tmp_path,check=True)


def test_notebook_runner_uses_the_captured_output_helper(tmp_path,monkeypatch):
    import io,contextlib,os,sys
    from colab.final import launcher
    nb=json.load(open('colab/Spark_Formal_Full_Finetune.ipynb'))
    tree=ast.parse(''.join(nb['cells'][8]['source']))
    runner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_campaign')
    python=tmp_path/'env-cpu/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    # Execute the actual notebook wrapper against a real child module, replacing
    # only the heavyweight campaign. Child fd output must reach Python capture.
    module=tmp_path/'colab/final';module.mkdir(parents=True)
    (module.parent/'__init__.py').touch();(module/'__init__.py').touch()
    (module/'run.py').write_text("import json,sys,os\ns=json.loads(sys.stdin.readline())\nassert s['HF_TOKEN']=='test-credential'\nassert 'HF_TOKEN' not in os.environ\nprint('child stdout',flush=True)\nprint('child stderr',file=sys.stderr,flush=True)\n")
    state=dict(WORK=tmp_path,SOURCE_ROOT=tmp_path,CONFIG_PATH=tmp_path/'config',SOURCE_SHA256='a'*64,
               HF_SECRET='test-credential',INSTRUCTOR_SECRET=None,json=json,os=os,_launcher=launcher)
    exec(compile(ast.Module(body=[runner],type_ignores=[]),'notebook-runner','exec'),state)
    monkeypatch.setenv('HF_TOKEN','must-not-be-inherited')
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream): state['run_campaign']('cpu')
    assert 'child stdout' in stream.getvalue() and 'child stderr' in stream.getvalue()
    assert 'test-credential' not in stream.getvalue()
