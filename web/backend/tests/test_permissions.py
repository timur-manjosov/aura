"""Permission-bitmask parsing, and every hostile shape it must survive.

The one authorization decision this service makes reduces to a bit test on a
number that arrives as text over the network. That makes the parser -- not the
bit test -- the part worth attacking, so most of this file is malformed input.
"""
from __future__ import annotations

import pytest
from fake_discord import PERMISSION_ADMINISTRATOR, PERMISSION_MANAGE_GUILD

from aura_web.permissions import (
    MANAGE_GUILD,
    MAX_PERMISSIONS_DIGITS,
    has_manage_guild,
    parse_permissions,
    parse_snowflake,
    sanitize_guild_name,
    sanitize_icon_hash,
)


class TestBitValuesMatchDiscord:
    def test_the_constants_agree_with_discords_documented_bits(self) -> None:
        """Guards against a transposed shift, which no other test would notice."""
        assert MANAGE_GUILD == PERMISSION_MANAGE_GUILD == 32
        assert PERMISSION_ADMINISTRATOR == 8


class TestParsePermissions:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0", 0),
            ("32", 32),
            ("8", 8),
            ("2199023255551", 2199023255551),
            (32, 32),
            (0, 0),
            ("  32  ", 32),
        ],
    )
    def test_accepts_every_well_formed_shape(self, raw: object, expected: int) -> None:
        assert parse_permissions(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "abc",
            "32.0",
            "0x20",
            "32,",
            "+32",
            [32],
            {"permissions": 32},
            object(),
        ],
    )
    def test_rejects_every_malformed_shape(self, raw: object) -> None:
        assert parse_permissions(raw) is None

    def test_rejects_a_negative_string(self) -> None:
        """-1 & MANAGE_GUILD is MANAGE_GUILD in Python. Rejection is the only safe answer."""
        assert parse_permissions("-1") is None
        assert has_manage_guild("-1") is False

    def test_rejects_a_negative_int(self) -> None:
        assert parse_permissions(-1) is None
        assert has_manage_guild(-1) is False

    def test_rejects_booleans(self) -> None:
        """bool is an int subclass; True would otherwise parse as permission bit 0."""
        assert parse_permissions(True) is None
        assert parse_permissions(False) is None

    def test_rejects_an_absurdly_long_digit_string(self) -> None:
        """A megabyte of digits is a CPU-time lever, not a permission set."""
        assert parse_permissions("9" * (MAX_PERMISSIONS_DIGITS + 1)) is None
        assert parse_permissions("1" * 100_000) is None

    def test_accepts_exactly_the_maximum_length(self) -> None:
        assert parse_permissions("1" * MAX_PERMISSIONS_DIGITS) is not None

    def test_a_value_far_past_64_bits_still_parses(self) -> None:
        """Discord serialises permissions as a string precisely to outgrow 64 bits."""
        huge = str((1 << 130) | MANAGE_GUILD)
        assert len(huge) <= MAX_PERMISSIONS_DIGITS
        assert has_manage_guild(huge) is True


class TestHasManageGuild:
    def test_the_manage_guild_bit_grants_access(self) -> None:
        assert has_manage_guild(str(PERMISSION_MANAGE_GUILD)) is True

    def test_the_administrator_bit_grants_access_on_its_own(self) -> None:
        """Discord's computed mask does not fold ADMINISTRATOR into the other bits."""
        assert has_manage_guild(str(PERMISSION_ADMINISTRATOR)) is True

    def test_an_unrelated_permission_does_not_grant_access(self) -> None:
        assert has_manage_guild("2048") is False

    def test_no_permissions_does_not_grant_access(self) -> None:
        assert has_manage_guild("0") is False

    def test_every_neighbouring_bit_is_rejected(self) -> None:
        """16 and 64 sit either side of MANAGE_GUILD; an off-by-one shift would pass one."""
        assert has_manage_guild("16") is False
        assert has_manage_guild("64") is False

    def test_an_unparseable_value_denies_rather_than_defaults(self) -> None:
        assert has_manage_guild("not-a-number") is False
        assert has_manage_guild(None) is False


class TestParseSnowflake:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1000", "1000"), (1000, "1000"), ("  1000 ", "1000"), ("0001000", "1000")],
    )
    def test_normalises_well_formed_ids(self, raw: object, expected: str) -> None:
        assert parse_snowflake(raw) == expected

    @pytest.mark.parametrize(
        "raw", [None, "", "abc", "-1", -1, True, "12.5", "9" * 40, [1], {"id": 1}]
    )
    def test_rejects_malformed_ids(self, raw: object) -> None:
        assert parse_snowflake(raw) is None

    def test_two_spellings_of_one_id_normalise_to_the_same_string(self) -> None:
        """Otherwise one guild could appear twice, or dedupe could silently fail."""
        assert parse_snowflake("007") == parse_snowflake(7) == "7"


class TestSanitizeGuildName:
    def test_keeps_an_ordinary_name(self) -> None:
        assert sanitize_guild_name("Aura Test Server", fallback="x") == "Aura Test Server"

    def test_keeps_non_latin_names_intact(self) -> None:
        """Nine locales ship with this project; a name filter that mangles them is a bug."""
        for name in ["日本語サーバー", "한국어 서버", "Türkçe Sunucu", "Серверная", "🎮 Gaming"]:
            assert sanitize_guild_name(name, fallback="x") == name

    def test_strips_control_characters(self) -> None:
        assert sanitize_guild_name("Evil\x00\x1b[31mServer", fallback="x") == "Evil[31mServer"

    def test_strips_a_newline_that_would_break_any_consumer(self) -> None:
        assert "\n" not in sanitize_guild_name("Line\nBreak", fallback="x")

    def test_truncates_an_overlong_name(self) -> None:
        assert len(sanitize_guild_name("x" * 5000, fallback="f")) == 100

    @pytest.mark.parametrize("raw", [None, 123, ["name"], "", "   ", "\x00\x01"])
    def test_falls_back_when_there_is_nothing_usable(self, raw: object) -> None:
        assert sanitize_guild_name(raw, fallback="fallback") == "fallback"


class TestSanitizeIconHash:
    def test_accepts_a_plain_hash(self) -> None:
        assert sanitize_icon_hash("a1b2c3d4") == "a1b2c3d4"

    def test_accepts_an_animated_icon_hash(self) -> None:
        assert sanitize_icon_hash("a_1b2c3d4") == "a_1b2c3d4"

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            123,
            "../../../etc/passwd",
            "abc/def",
            "abc.png",
            "abc?x=1",
            "abc#frag",
            "abc def",
            "https://evil.test/x",
            "a" * 65,
            "ábc",
        ],
    )
    def test_rejects_anything_that_would_escape_the_cdn_path(self, raw: object) -> None:
        """The value is interpolated into /icons/{id}/{icon}.png by whoever renders it."""
        assert sanitize_icon_hash(raw) is None
