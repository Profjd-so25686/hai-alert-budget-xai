from pathlib import Path

from haistudy.custody import sha256_of, verify_manifest, write_manifest


def test_known_digest(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_detects_change_and_missing(tmp_path: Path):
    data = tmp_path / "raw"
    data.mkdir()
    (data / "test3.csv").write_text("1,2,3\n")
    (data / "test4.csv").write_text("4,5,6\n")
    (data / "train1.csv").write_text("7,8,9\n")
    manifest = tmp_path / "m.json"
    write_manifest(data, manifest)

    assert verify_manifest(data, manifest) == {"ok": ["test3.csv", "test4.csv", "train1.csv"], "changed": [], "missing": []}

    (data / "test3.csv").write_text("tampered\n")
    (data / "test4.csv").unlink()
    r = verify_manifest(data, manifest, only=["test3", "test4", "test5"])
    assert r == {"ok": [], "changed": ["test3.csv"], "missing": ["test4.csv"]}
