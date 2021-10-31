import sys
import io
import json
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
    pip_install('spacy spacy-transformers')


def init_spacy():
    import spacy
    import spacy_transformers
    if GPU_COUNT:
        spacy.require_gpu()


def init(spacy=False, packages=True, processors=True, full=False):
    if full or processors:
        init_processors()
    if full or packages:
        init_packages()
    if full or spacy:
        init_spacy()


# Drive functions
def zcat(inp, out):
    with gzip.open(inp, 'rb') as fi, open(out, 'wb') as oi:
        data = True
        while data:
            data = fi.read(8192)
            oi.write(data)


def _get_remote(remote_path, extract=True):
    remote_name = remote_path.name
    name = re.sub(r'(\.gz|\.tar)+$', '', remote_name)
    local_path = Path(name).absolute()

    if extract:
        if remote_name.endswith('.tar.gz'):
            sp.check_call(['tar', '-xzf', remote_path, '-C', local_path])
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
def write_jsonl(f, generator):
    if isinstance(f, io.IOBase):
        _write_jsonl(f, generator)
    else:
        with open(f, 'w') as f:
            _write_jsonl(f, generator)

def _write_jsonl(f, generator):
    print(2)
    for line in generator:
        f.write(json.dumps(line, ensure_ascii=False, sort_keys=True))
        f.write('\n')

def read_jsonl(f):
    if isinstance(f, io.IOBase):
        yield from _read_jsonl(f)
    else:
        with open(f) as f:
            yield from _read_jsonl(f)

def _read_jsonl(f):
    yield from (json.loads(line) for line in f)
