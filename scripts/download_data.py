"""Download the exact official archive and safely extract only recognized files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import requests

URL="https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,help="Use a locally downloaded official archive")
    args=parser.parse_args()
    raw=Path("data/raw");raw.mkdir(parents=True,exist_ok=True)
    path=args.archive
    if path is None:
        path=raw/"hackathonlicence.zip"
        with requests.get(URL,stream=True,timeout=(20,120)) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_content(2**20):f.write(chunk)
    manifest=json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest["archive_sha256"]:
        raise ValueError("Archive checksum changed. Inspect the new data vintage before updating the manifest.")
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            name=Path(member.filename).name
            if name in ["consumption.parquet","connection.parquet","market_access.parquet"]:
                (raw/name).write_bytes(z.read(member))
            elif name.lower().endswith(".pdf"):
                (raw/"source_description_license.pdf").write_bytes(z.read(member))
    print("Official archive verified and extracted to data/raw")


if __name__=="__main__":main()
