from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import SnapshotPriority

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


def snapshot_sparse_corrupt_reason(path: Path) -> str | None:
    """Docker Desktop + Windows bind-mount snapshots often store small files as GNU sparse holes."""
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issparse() and (
                member.name.endswith("wal/first-index")
                or member.name.endswith("newest_clocks.json")
            ):
                return member.name
        nested = next((m for m in members if m.name.endswith(".tar") and "/segments/" in m.name), None)
        if nested is None:
            return None
        extracted = archive.extractfile(nested)
        if extracted is None:
            return None
        with tarfile.open(fileobj=extracted, mode="r:") as inner:
            for member in inner.getmembers():
                if member.issparse() and member.name.endswith("version.info"):
                    return f"{nested.name} -> {member.name}"
    return None


def wait_for_qdrant(timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            get_client().get_collections()
            return
        except Exception as exc:
            last_err = exc
            time.sleep(1)
    raise SystemExit(f"Qdrant pas joignable après {timeout:.0f}s : {last_err}")


def wipe_broken_collection(coll: str) -> None:
    """Stop Qdrant and drop the on-disk collection after a failed restore/WAL crash."""
    print("collection cassée, reset du dossier via docker…")
    subprocess.run(["docker", "compose", "stop", "qdrant"], cwd=ROOT, check=True)
    coll_dir = STORAGE / "collections" / coll
    if coll_dir.exists():
        shutil.rmtree(coll_dir)
        print("supprimé :", coll_dir)
    subprocess.run(["docker", "compose", "start", "qdrant"], cwd=ROOT, check=True)
    wait_for_qdrant()


def collection_is_usable(client, coll: str) -> bool:
    try:
        if not client.collection_exists(coll):
            return False
        client.get_collection(coll)
        return True
    except UnexpectedResponse:
        return False


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
    before = {s.name for s in (client.list_snapshots(collection_name=coll) or [])}

    print("création du snapshot…")
    started = client.create_snapshot(collection_name=coll, wait=False)
    expected = getattr(started, "name", None)

    deadline = time.monotonic() + 600
    snap = None
    while time.monotonic() < deadline:
        listed = client.list_snapshots(collection_name=coll) or []
        if expected:
            snap = next((s for s in listed if s.name == expected), None)
        else:
            snap = next((s for s in listed if s.name not in before), None)
        if snap is not None:
            break
        time.sleep(1)
    if snap is None:
        raise SystemExit("timeout: snapshot pas listé après 10 min (Qdrant bloqué ?)")

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
    client = get_client(timeout=600)
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

    if src is not None:
        reason = snapshot_sparse_corrupt_reason(src)
        if reason:
            raise SystemExit(
                "snapshot inutilisable : fichiers métadonnées vides (GNU sparse) "
                f"({reason}).\n"
                "Cause typique : volume Docker bind-mounté depuis Windows.\n"
                "Relance l'index : python scripts/ingest_qdrant.py --recreate"
            )

    if collection_is_usable(client, coll):
        print("suppression de", coll, "avant restore…")
        client.delete_collection(coll)
    else:
        try:
            leftover = client.collection_exists(coll)
        except UnexpectedResponse:
            leftover = True
        if leftover:
            wipe_broken_collection(coll)
            client = get_client(timeout=600)

    location = f"file:///qdrant/storage/snapshots/{coll}/{Path(name).name}"
    print("recover depuis", location)
    client.recover_snapshot(
        collection_name=coll,
        location=location,
        priority=SnapshotPriority.SNAPSHOT,
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
