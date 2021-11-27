import sys
import io
import json
import dataclasses
import functools
from pathlib import Path
import os
import multiprocessing as mp
import subprocess as sp
import re


# GLOBALS
DRIVE_DIR = Path('/content/drive/MyDrive')
DATASETS_DIR = DRIVE_DIR / 'datasets'
MODELS_DIR = DRIVE_DIR / 'models'
LOGS_DIR = DRIVE_DIR / 'tb_logs'

SPACY_DIR = Path('.spacy').resolve()
SPACY_DIR.mkdir(exist_ok=True)
SPACY_REMOTE_DIR = MODELS_DIR / 'spacy'

TPU_COUNT = 0
GPU_COUNT = int(os.getenv('COLAB_GPU', 0))
CPU_COUNT = mp.cpu_count()
RUNTIME = None


def pip_install(packages):
    packages = packages.split(' ') if isinstance(packages, str) else packages
    sp.check_call([
        sys.executable, '-m', 'pip', 'install', '-U', '-q',
        *packages
    ])


def init_processors():
    global RUNTIME, TPU_COUNT
    if TPU_COUNT:
        print('SETUP TPU...')
        pip_install('cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl')
        print('TPU count:', TPU_COUNT)
        RUNTIME = 'TPU'
        TPU_COUNT = 8 if os.environ.get('COLAB_TPU_ADDR') else 0
    elif GPU_COUNT:
        print('SETUP GPU...')
        print('GPU count:', GPU_COUNT)
        RUNTIME = 'GPU'
    else:
        print('SETUP CPU...')
        print('CPU count:', CPU_COUNT)
        RUNTIME = 'CPU'


def init_packages():
    print('Init packages')
    pip_install('spacy spacy-transformers')


def init_spacy(models):
    print('Init spacy')
    import spacy
    import spacy_transformers
    if GPU_COUNT:
        spacy.require_gpu()

    for name in models:
        untar(SPACY_REMOTE_DIR / name, SPACY_DIR)


def init(spacy=False, packages=True, processors=True, full=False):
    if full or processors:
        init_processors()
    if full or packages:
        init_packages()

    from google.colab import drive
    drive.mount('/content/drive')

    if full or spacy:
        init_spacy(spacy if spacy else [])



# Drive functions
def zcat(inp, out):
    with gzip.open(inp, 'rb') as fi, open(out, 'wb') as oi:
        data = True
        while data:
            data = fi.read(8192)
            oi.write(data)

def untar(inp, out):
    sp.check_call(['tar', '-xzf', str(inp), '-C', str(out)])


def _get_remote(remote_path, extract=True):
    remote_name = remote_path.name
    name = re.sub(r'(\.gz|\.tar)+$', '', remote_name)
    local_path = Path(name).absolute()

    if extract and remote_name.exists():
        if remote_name.endswith('.tar.gz'):
            local_path.mkdir(parents=True, exist_ok=True)
            untar(remote_path, local_path)
        elif remote_name.endswith('.gz'):
            # zcat(remote_path, local_path)
            with local_path.open('wb') as f:
                sp.check_call(['zcat', remote_path], stdout=f)

    return remote_path, local_path


def get_dataset(remote_name, extract=True):
    return _get_remote(DATASETS_DIR / remote_name, extract)


def get_model(remote_name, extract=True):
    return _get_remote(MODELS_DIR / remote_name, extract)


# Common functions

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

DUMPS = functools.partial(json.dumps, cls=JSONEncoder)

def write_jsonl(f, generator, dumps=None):
    if isinstance(f, io.IOBase):
        _write_jsonl(f, generator, dumps)
    else:
        with open(f, 'w') as f:
            _write_jsonl(f, generator, dumps)

def _write_jsonl(f, generator, dumps):
    dumps = dumps or DUMPS
    for line in generator:
        f.write(dumps(line, ensure_ascii=False, sort_keys=True))
        f.write('\n')

def read_jsonl(f):
    if isinstance(f, io.IOBase):
        yield from _read_jsonl(f)
    else:
        with open(f) as f:
            yield from _read_jsonl(f)

def _read_jsonl(f):
    yield from (json.loads(line) for line in f)
