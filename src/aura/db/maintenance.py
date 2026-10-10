"""Operator commands for the database file: verify, encrypt, decrypt, back up, rotate the key.

Run inside the bot's container, where the key is in the environment:

    docker compose exec aura python -m aura.db.maintenance verify data/aura.db
    docker compose run --rm --no-deps --entrypoint python aura \\
        -m aura.db.maintenance encrypt data/aura.db data/aura.encrypted.db

`encrypt`, `decrypt` and `rekey` need the bot STOPPED (they read or rewrite the
file as a whole); `verify` and `backup` are safe while it runs. Every command
prints integrity results, the schema fingerprint and row counts per table --
never a row's content -- and exits non-zero on any failure. The key is read
from `DATABASE_ENCRYPTION_KEY` (and the new key for `rekey` from
`DATABASE_ENCRYPTION_NEW_KEY`), never from the command line, so it does not
land in a shell history or a process list.

Imports only `aura.db.encryption`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from aura.db.encryption import (
    DATABASE_KEY_ENV,
    NEW_DATABASE_KEY_ENV,
    DatabaseOpenError,
    DatabaseReport,
    backup_database,
    export_encrypted,
    export_plaintext,
    inspect_database,
    rekey_database,
    validate_database_key,
)


def _key_from_env(name: str, *, required: bool) -> str | None:
    """Return the validated key in an environment variable, or None when unset and optional."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        if required:
            raise DatabaseOpenError(f"{name} is not set.")
        return None
    try:
        return validate_database_key(raw)
    except ValueError as exc:
        raise DatabaseOpenError(str(exc).replace(DATABASE_KEY_ENV, name)) from exc


def format_report(label: str, report: DatabaseReport) -> str:
    """Render a report as the lines this tool prints.

    Parameters
    ----------
    label
        What the report is about, e.g. "backup".
    report
        The checks and counts.

    Returns
    -------
    str
        One summary line plus one line per table: counts only.
    """
    lines = [
        f"{label}: integrity={report.integrity} cipher={report.cipher_integrity} "
        f"schema={report.schema_hash} tables={len(report.table_counts)} "
        f"rows={sum(report.table_counts.values())}"
    ]
    lines.extend(f"  {table}\t{count}" for table, count in report.table_counts.items())
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aura.db.maintenance")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check a file with the configured key (or none)")
    verify.add_argument("path")
    encrypt = commands.add_parser("encrypt", help="write an encrypted copy of a plaintext file")
    encrypt.add_argument("source")
    encrypt.add_argument("target")
    decrypt = commands.add_parser("decrypt", help="write a plaintext copy of an encrypted file")
    decrypt.add_argument("source")
    decrypt.add_argument("target")
    backup = commands.add_parser("backup", help="online backup with the same key")
    backup.add_argument("source")
    backup.add_argument("target")
    rekey = commands.add_parser("rekey", help="re-encrypt a file under DATABASE_ENCRYPTION_NEW_KEY")
    rekey.add_argument("path")
    return parser


def run(argv: Sequence[str]) -> int:
    """Run one maintenance command and print its report.

    Parameters
    ----------
    argv
        The arguments after the program name.

    Returns
    -------
    int
        0 on success, 1 on any refusal or failure (the reason is printed to
        stderr, without content or key).
    """
    args = _build_parser().parse_args(list(argv))
    try:
        if args.command == "verify":
            key = _key_from_env(DATABASE_KEY_ENV, required=False)
            print(format_report("verify", inspect_database(args.path, key)))
        elif args.command == "encrypt":
            key = _key_from_env(DATABASE_KEY_ENV, required=True)
            assert key is not None
            print(format_report("source", inspect_database(args.source, None)))
            print(format_report("encrypted copy", export_encrypted(args.source, args.target, key)))
        elif args.command == "decrypt":
            key = _key_from_env(DATABASE_KEY_ENV, required=True)
            assert key is not None
            print(format_report("plaintext copy", export_plaintext(args.source, args.target, key)))
        elif args.command == "backup":
            key = _key_from_env(DATABASE_KEY_ENV, required=False)
            print(format_report("backup", backup_database(args.source, args.target, key)))
        else:
            old_key = _key_from_env(DATABASE_KEY_ENV, required=True)
            new_key = _key_from_env(NEW_DATABASE_KEY_ENV, required=True)
            assert old_key is not None and new_key is not None
            print(format_report("rekeyed", rekey_database(args.path, old_key, new_key)))
    except DatabaseOpenError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through run()
    sys.exit(run(sys.argv[1:]))
