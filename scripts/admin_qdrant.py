from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qdrant_utils import ROOT, collection_name, get_client

BACKUPS = ROOT / "data" / "backups"
STORAGE = ROOT / "qdrant_storage"


def snapshot_dirs(coll: str) -> list[Path]:
    return [
        STORAGE / "snapshots" / coll,
        STORAGE / "collections" / coll / "snapshots",
        STORAGE / ".snapshots" / coll,
    ]


def find_snapshot_file(coll: str, name: str) -> Path | None:
    for folder in snapshot_dirs(coll):
        candidate = folder / name
        if candidate.exists():
            return candidate
    p = Path(name)
    if p.exists():
        return p
    return None


def cmd_info(_args):
    client = get_client()
    coll = collection_name()
    if not client.collection_exists(coll):
        raise SystemExit(f"collection absente : {coll}")

    info = client.get_collection(coll)
    count = client.count(coll).count
    print("collection :", coll)
    print("points     :", count)
    print("status     :", info.status)
    print("vectors    :", info.config.params.vectors)
    print("indexed    :", getattr(info, "indexed_vectors_count", "?"))
    schema = getattr(info, "payload_schema", None) or {}
    if schema:
        print("payload indexes :")
        for field, meta in schema.items():
            print(f"  - {field}: {meta}")
    else:
        print("payload indexes : (aucun / pas exposé)")


def cmd_snapshot(_args):
    client = get_client()
    coll = collection_name()
    snap = client.create_snapshot(collection_name=coll)
    print("snapshot créé :", snap.name)

    BACKUPS.mkdir(parents=True, exist_ok=True)
    src = find_snapshot_file(coll, snap.name)
    if src is None:
        print("fichier pas trouvé dans qdrant_storage (normal si Qdrant est distant).")
        print("il est côté serveur, restore marchera avec le nom.")
        return

    dest = BACKUPS / snap.name
    shutil.copy2(src, dest)
    print("copie locale :", dest)


def cmd_list(_args):
    client = get_client()
    coll = collection_name()
    snaps = client.list_snapshots(collection_name=coll)
    if not snaps:
        print("aucun snapshot")
        return
    for s in snaps:
        print(s.name, "|", getattr(s, "creation_time", ""), "|", getattr(s, "size", ""))


def cmd_restore(args):
    client = get_client()
    coll = collection_name()
    name = args.name

    src = find_snapshot_file(coll, name)
    backup = BACKUPS / name
    if src is None and backup.exists():
        dest_dir = snapshot_dirs(coll)[0]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, dest_dir / name)
        src = dest_dir / name
        print("snapshot recopié dans le volume :", src)

    location = f"file:///qdrant/storage/snapshots/{coll}/{Path(name).name}"
    print("recover depuis", location)
    client.recover_snapshot(
        collection_name=coll,
        location=location,
        wait=True,
    )
    print("restore ok, count =", client.count(coll).count)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info").set_defaults(func=cmd_info)
    sub.add_parser("snapshot").set_defaults(func=cmd_snapshot)
    sub.add_parser("list-snapshots").set_defaults(func=cmd_list)

    p_res = sub.add_parser("restore")
    p_res.add_argument("--name", required=True)
    p_res.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
