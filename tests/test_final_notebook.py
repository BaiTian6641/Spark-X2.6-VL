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


def test_secret_stdin_uses_a_real_newline_not_literal_backslash_n():
    nb=json.load(open('colab/Spark_Formal_Full_Finetune.ipynb'));found=False
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        for n in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='secret_payload' for t in n.targets):
                assert isinstance(n.value.right,ast.Call) and n.value.right.func.id=='chr' and n.value.right.args[0].value==10; found=True
    assert found
