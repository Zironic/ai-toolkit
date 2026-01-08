import os
from pathlib import Path
import json
import hashlib
import tempfile
import time

from toolkit.cache_utils import (
    compute_file_sha256,
    compute_param_digest,
    compute_combined_hash,
    atomic_write,
    find_cached_file,
)


def test_compute_file_sha256(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_bytes(b'hello world')
    h = compute_file_sha256(p)
    assert h == hashlib.sha256(b'hello world').hexdigest()


def test_compute_param_digest_ordering():
    a = {'b': 2, 'a': 1}
    b = {'a': 1, 'b': 2}
    d1 = compute_param_digest(a, length=16)
    d2 = compute_param_digest(b, length=16)
    assert d1 == d2


def test_compute_combined_hash(tmp_path):
    p1 = tmp_path / 'f1.bin'
    p2 = tmp_path / 'f2.bin'
    p1.write_bytes(b'one')
    p2.write_bytes(b'two')
    ch = compute_combined_hash([p1, p2])
    # recompute manually
    import hashlib
    d1 = hashlib.sha256(b'one').hexdigest()
    d2 = hashlib.sha256(b'two').hexdigest()
    expected = hashlib.sha256((d1 + d2).encode('utf-8')).hexdigest()
    assert ch == expected


def test_atomic_write_and_find(tmp_path):
    target = tmp_path / 'out.bin'
    def writer(p: Path):
        p.write_bytes(b'data')
    atomic_write(target, writer)
    assert target.exists()
    assert target.read_bytes() == b'data'
    # test find_cached_file exact
    found = find_cached_file(target)
    assert found == target
    # create legacy style file
    legacy = tmp_path / 'out_legacy.safetensors'
    legacy.write_text('legacy')
    # expected non-existing path
    exp = tmp_path / 'out_nonexistent.safetensors'
    # fallback will look for files starting with base "out_" — should find the legacy file
    assert find_cached_file(exp) is not None
    exp2 = tmp_path / 'out_legacy_dummy.safetensors'
    # construct expected path with base 'out'
    exp3 = tmp_path / 'out_dummy.safetensors'
    # find using exp3 should find legacy file starting with 'out_'
    found2 = find_cached_file(exp3)
    assert found2 is not None
    assert found2.name.startswith('out_')
