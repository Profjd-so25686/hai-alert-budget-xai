"""Command line for the chain-of-custody manifest.

    python scripts/custody.py hash data/raw docs/hai_manifest.json
    python scripts/custody.py verify data/raw docs/hai_manifest.json
    python scripts/custody.py verify data/raw docs/hai_manifest.json --only test3 test4 test5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from haistudy.custody import verify_manifest, write_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash"); h.add_argument("folder"); h.add_argument("manifest")
    v = sub.add_parser("verify"); v.add_argument("folder"); v.add_argument("manifest"); v.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "hash":
        m = write_manifest(Path(a.folder), Path(a.manifest))
        print(f"Hashed {len(m['files'])} files into {a.manifest}")
        return 0
    r = verify_manifest(Path(a.folder), Path(a.manifest), a.only)
    print(f"ok: {len(r['ok'])}  changed: {len(r['changed'])}  missing: {len(r['missing'])}")
    for k in ("changed", "missing"):
        for f in r[k]:
            print(f"  {k}: {f}")
    return 0 if not r["changed"] and not r["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
