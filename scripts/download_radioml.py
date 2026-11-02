"""Fetch and unpack RadioML 2016.10a into data/ (never commit it).

DeepSig distributes the dataset under CC BY-NC-SA 4.0 behind a short form, so there is
no stable direct URL. Download RML2016.10a.tar.bz2 from https://www.deepsig.ai/datasets
after accepting the licence, then run:

    python scripts/download_radioml.py --archive ~/Downloads/RML2016.10a.tar.bz2

or, if you have a direct link of your own:

    python scripts/download_radioml.py --url <link>
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

TARGET = "RML2016.10a_dict.pkl"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=Path, help="local RML2016.10a.tar.bz2")
    src.add_argument("--url", help="direct download link")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    if args.url:
        archive = args.out / "RML2016.10a.tar.bz2"
        print(f"downloading {args.url} -> {archive}")
        urllib.request.urlretrieve(args.url, archive)  # noqa: S310 - user-supplied link

    with tarfile.open(archive, "r:bz2") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(TARGET)), None)
        if member is None:
            raise SystemExit(f"{TARGET} not found in {archive}")
        member.name = TARGET  # flatten; also blocks path traversal from the archive
        tar.extract(member, args.out, filter="data")
    print(f"wrote {args.out / TARGET}. Set data.source: radioml in your config to use it.")


if __name__ == "__main__":
    main()
