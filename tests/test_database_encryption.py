"""Encryption at rest (P7a, R1): SQLCipher files, the maintenance commands, and the key-loss case."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
from pathlib import Path

import pytest

from aura.db.encryption import (
    DatabaseOpenError,
    backup_database,
    connect_database,
    create_encrypted_database,
    export_encrypted,
    export_plaintext,
    inspect_database,
    is_integrity_error,
    rekey_database,
    validate_database_key,
)
from aura.db.maintenance import run
from tests.privacy_data import BEFORE, GUILD_A, MEMBER, add_fact, open_database, populate

pytest.importorskip("sqlcipher3")

CANARY = "CANARY-p7a-encryption-4b1d"


def _key() -> str:
    return secrets.token_hex(32)


async def _plain_database(path: Path) -> None:
    conn = await open_database(str(path))
    await populate(conn)
    await add_fact(conn, guild_id=GUILD_A, author=MEMBER, when=BEFORE, content=CANARY)
    await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    await conn.close()


@pytest.fixture
def plain(tmp_path: Path) -> Path:
    path = tmp_path / "aura.db"
    asyncio.run(_plain_database(path))
    return path


class TestKeyValidation:
    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "abc",
            "g" * 64,
            "a" * 63,
            "a" * 65,
            "a" * 32 + " " + "a" * 31,
            "'; DROP TABLE facts; --",
        ],
    )
    def test_anything_but_64_hex_characters_is_refused(self, raw: str) -> None:
        with pytest.raises(ValueError) as error:
            validate_database_key(raw)
        assert raw.strip() == "" or raw not in str(error.value)

    def test_a_valid_key_is_normalised_to_lower_case(self) -> None:
        assert validate_database_key(" " + "AB" * 32 + "\n") == "ab" * 32


class TestTheEncryptedFile:
    def test_the_copy_has_the_same_rows_and_none_of_the_text(
        self, plain: Path, tmp_path: Path
    ) -> None:
        key = _key()
        target = tmp_path / "aura.encrypted.db"
        report = export_encrypted(plain, target, key)
        original = inspect_database(plain, None)
        assert report.table_counts == original.table_counts
        assert report.schema_hash == original.schema_hash
        assert (report.integrity, report.cipher_integrity) == ("ok", "ok")
        data = target.read_bytes()
        assert not data.startswith(b"SQLite format 3")
        assert CANARY.encode() not in data
        assert CANARY.encode() in plain.read_bytes()
        assert (target.stat().st_mode & 0o777) == 0o600

    def test_two_exports_of_the_same_file_have_the_same_schema_hash(
        self, plain: Path, tmp_path: Path
    ) -> None:
        key = _key()
        first = export_encrypted(plain, tmp_path / "one.db", key)
        second = export_encrypted(plain, tmp_path / "two.db", key)
        assert first.schema_hash == second.schema_hash
        assert first.table_counts == second.table_counts

    def test_an_existing_target_is_never_overwritten(self, plain: Path, tmp_path: Path) -> None:
        target = tmp_path / "taken.db"
        target.write_bytes(b"keep me")
        with pytest.raises(DatabaseOpenError):
            export_encrypted(plain, target, _key())
        assert target.read_bytes() == b"keep me"
        (tmp_path / "wal.db-wal").write_bytes(b"x")
        with pytest.raises(DatabaseOpenError):
            export_encrypted(plain, tmp_path / "wal.db", _key())

    def test_a_copy_that_does_not_match_its_source_is_refused(
        self, plain: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aura.db.encryption as encryption

        real_inspect = encryption.inspect_database

        def short_by_one(path: str | Path, key: str | None) -> encryption.DatabaseReport:
            report = real_inspect(path, key)
            counts = dict(report.table_counts)
            counts["facts"] -= 1
            return encryption.DatabaseReport(
                report.integrity, report.cipher_integrity, report.schema_hash, counts
            )

        monkeypatch.setattr(encryption, "inspect_database", short_by_one)
        with pytest.raises(DatabaseOpenError):
            export_encrypted(plain, tmp_path / "enc.db", _key())

    def test_decrypting_gives_the_plaintext_back(self, plain: Path, tmp_path: Path) -> None:
        key = _key()
        export_encrypted(plain, tmp_path / "enc.db", key)
        back = export_plaintext(tmp_path / "enc.db", tmp_path / "back.db", key)
        assert back.table_counts == inspect_database(plain, None).table_counts
        assert (tmp_path / "back.db").read_bytes().startswith(b"SQLite format 3")


class TestOpening:
    def test_a_wrong_key_is_refused_and_changes_nothing(self, plain: Path, tmp_path: Path) -> None:
        key = _key()
        target = tmp_path / "enc.db"
        export_encrypted(plain, target, key)
        before = target.read_bytes()

        async def attempt() -> None:
            conn = await connect_database(str(target), _key())
            await conn.close()

        with pytest.raises(DatabaseOpenError):
            asyncio.run(attempt())
        assert target.read_bytes() == before

    def test_an_encrypted_file_without_a_key_is_refused(self, plain: Path, tmp_path: Path) -> None:
        target = tmp_path / "enc.db"
        export_encrypted(plain, target, _key())

        async def attempt() -> None:
            conn = await connect_database(str(target), None)
            await conn.close()

        with pytest.raises(DatabaseOpenError):
            asyncio.run(attempt())

    def test_a_plaintext_file_with_a_key_is_refused(self, plain: Path) -> None:
        before = plain.read_bytes()

        async def attempt() -> None:
            conn = await connect_database(str(plain), _key())
            await conn.close()

        with pytest.raises(DatabaseOpenError):
            asyncio.run(attempt())
        assert plain.read_bytes() == before

    def test_a_missing_file_is_never_created_when_a_key_is_set(self, tmp_path: Path) -> None:
        async def attempt() -> None:
            conn = await connect_database(str(tmp_path / "missing.db"), _key())
            await conn.close()

        with pytest.raises(DatabaseOpenError):
            asyncio.run(attempt())
        assert not (tmp_path / "missing.db").exists()

    def test_without_a_key_a_missing_file_is_created_as_before(self, tmp_path: Path) -> None:
        async def attempt() -> None:
            conn = await connect_database(str(tmp_path / "new.db"), None)
            await conn.execute("CREATE TABLE t (x)")
            await conn.close()

        asyncio.run(attempt())
        assert (tmp_path / "new.db").exists()

    def test_the_right_key_reads_and_writes(self, plain: Path, tmp_path: Path) -> None:
        key = _key()
        target = tmp_path / "enc.db"
        export_encrypted(plain, target, key)

        async def use() -> int:
            conn = await connect_database(str(target), key)
            await conn.execute("DELETE FROM facts WHERE content = ?", (CANARY,))
            await conn.commit()
            async with conn.execute("SELECT COUNT(*) FROM facts") as cursor:
                row = await cursor.fetchone()
            await conn.close()
            assert row is not None
            return int(row[0])

        assert asyncio.run(use()) == inspect_database(plain, None).table_counts["facts"] - 1

    def test_a_constraint_violation_is_recognised_from_either_driver(self, tmp_path: Path) -> None:
        key = _key()
        path = tmp_path / "c.db"
        create_encrypted_database(path, key)
        from aura.db.encryption import open_sync

        conn = open_sync(path, key)
        conn.execute("CREATE TABLE t (x UNIQUE)")
        conn.execute("INSERT INTO t VALUES (1)")
        with pytest.raises(Exception) as error:
            conn.execute("INSERT INTO t VALUES (1)")
        conn.close()
        assert is_integrity_error(error.value)
        assert is_integrity_error(sqlite3.IntegrityError())
        assert not is_integrity_error(ValueError())


class TestBackupAndRotation:
    def test_a_backup_is_encrypted_with_the_same_key_and_reads_back(
        self, plain: Path, tmp_path: Path
    ) -> None:
        key = _key()
        export_encrypted(plain, tmp_path / "enc.db", key)
        report = backup_database(tmp_path / "enc.db", tmp_path / "bak.db", key)
        assert report.integrity == "ok" and report.cipher_integrity == "ok"
        assert CANARY.encode() not in (tmp_path / "bak.db").read_bytes()
        with pytest.raises(DatabaseOpenError):
            inspect_database(tmp_path / "bak.db", _key())

    def test_rotating_the_key_keeps_every_row(self, plain: Path, tmp_path: Path) -> None:
        old, new = _key(), _key()
        export_encrypted(plain, tmp_path / "enc.db", old)
        counts = inspect_database(tmp_path / "enc.db", old).table_counts
        assert rekey_database(tmp_path / "enc.db", old, new).table_counts == counts
        with pytest.raises(DatabaseOpenError):
            inspect_database(tmp_path / "enc.db", old)
        with pytest.raises(DatabaseOpenError):
            rekey_database(tmp_path / "enc.db", new, new)


class TestKeyLoss:
    """The written key-loss procedure (DEPLOYMENT.md), exercised end to end."""

    def test_without_the_key_nothing_opens_and_nothing_is_overwritten(
        self, plain: Path, tmp_path: Path
    ) -> None:
        key = _key()
        live = tmp_path / "enc.db"
        export_encrypted(plain, live, key)
        backup_database(live, tmp_path / "bak.db", key)
        live_bytes = live.read_bytes()
        # Step 1 of the procedure: with a guessed or empty key the bot refuses
        # to start and leaves the file as it is -- it is never replaced by an
        # empty database.
        for attempt_key in (_key(), None):
            with pytest.raises(DatabaseOpenError):
                inspect_database(live, attempt_key)
        assert live.read_bytes() == live_bytes
        # Step 2: the backups share the key; without it they are lost too.
        with pytest.raises(DatabaseOpenError):
            inspect_database(tmp_path / "bak.db", _key())

    def test_the_copy_in_the_password_manager_restores_everything(
        self, plain: Path, tmp_path: Path
    ) -> None:
        key = _key()
        escrowed = str(key)  # the copy kept outside the server
        live = tmp_path / "enc.db"
        export_encrypted(plain, live, key)
        assert (
            inspect_database(live, escrowed).table_counts
            == inspect_database(plain, None).table_counts
        )


class TestMaintenanceCommand:
    def test_every_command_prints_counts_and_never_content(
        self,
        plain: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        key, new_key = _key(), _key()
        monkeypatch.setenv("DATABASE_ENCRYPTION_KEY", key)
        monkeypatch.setenv("DATABASE_ENCRYPTION_NEW_KEY", new_key)
        enc, bak, back = tmp_path / "enc.db", tmp_path / "bak.db", tmp_path / "back.db"
        assert run(["encrypt", str(plain), str(enc)]) == 0
        assert run(["verify", str(enc)]) == 0
        assert run(["backup", str(enc), str(bak)]) == 0
        assert run(["decrypt", str(enc), str(back)]) == 0
        assert run(["rekey", str(bak)]) == 0
        out = capsys.readouterr()
        assert CANARY not in out.out + out.err
        assert key not in out.out + out.err and new_key not in out.out + out.err
        assert "integrity=ok" in out.out and "facts\t" in out.out

    def test_a_refusal_exits_non_zero_without_the_key(
        self,
        plain: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        wrong = _key()
        monkeypatch.setenv("DATABASE_ENCRYPTION_KEY", wrong)
        assert run(["verify", str(plain)]) == 1
        monkeypatch.setenv("DATABASE_ENCRYPTION_KEY", "not-a-key")
        assert run(["encrypt", str(plain), str(tmp_path / "x.db")]) == 1
        monkeypatch.delenv("DATABASE_ENCRYPTION_KEY")
        assert run(["encrypt", str(plain), str(tmp_path / "y.db")]) == 1
        out = capsys.readouterr()
        assert wrong not in out.err and "not-a-key" not in out.err
        assert not (tmp_path / "x.db").exists() and not (tmp_path / "y.db").exists()


class TestLedgerMigration:
    def test_the_ledger_is_encrypted_by_the_same_command_and_keeps_its_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        from aura.db.deletion import MemberDeletionMode
        from aura.privacy.ledger import DeletionKind, DeletionLedger, DeletionReason, LedgerEntry

        plain, encrypted = tmp_path / "ledger.db", tmp_path / "ledger.encrypted.db"
        moment = datetime(2026, 10, 1, tzinfo=UTC)

        async def write() -> None:
            ledger = await DeletionLedger.open(str(plain), None)
            await ledger.record(
                LedgerEntry(
                    kind=DeletionKind.MEMBER,
                    reason=DeletionReason.MEMBER_REQUEST,
                    requested_at=moment,
                    user_id=MEMBER,
                    mode=MemberDeletionMode.UNLINK,
                )
            )
            await ledger.close()

        asyncio.run(write())
        key = _key()
        monkeypatch.setenv("DATABASE_ENCRYPTION_KEY", key)
        assert run(["encrypt", str(plain), str(encrypted)]) == 0

        async def read() -> list[LedgerEntry]:
            ledger = await DeletionLedger.open(str(encrypted), key)
            entries = await ledger.entries()
            await ledger.close()
            return entries

        [entry] = asyncio.run(read())
        assert entry.user_id == MEMBER and entry.requested_at == moment
