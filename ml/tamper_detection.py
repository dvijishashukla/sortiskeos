import hashlib
import json
from pathlib import Path

def hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_hash(path: Path, hash_file_path: Path):
    hash_file_path.write_text(hash_file(path))

def verify_hash(path: Path, hash_file_path: Path) -> bool:
    if not hash_file_path.exists():
        return True  # first run, no hash yet
    stored = hash_file_path.read_text().strip()
    current = hash_file(path)
    return stored == current
