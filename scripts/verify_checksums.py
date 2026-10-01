"""Verify packaged source, data, and results using only the Python standard library."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
checked=0
for line in (root/"ARTIFACTS.sha256").read_text(encoding="utf-8").splitlines():
    expected,name=line.split("  ",1)
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError(f"Invalid manifest path: {name}")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected:raise ValueError(f"Changed file: {name}")
    checked+=1
print(f"Verified SHA-256 for {checked} packaged files.")
