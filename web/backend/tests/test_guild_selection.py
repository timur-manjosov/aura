"""The two-condition filter, as a pure function.

Every combination of (user can manage?) x (Aura is present?) is covered
explicitly, including the two that must NOT appear -- a filter tested only on
its positive case passes just as well when one of its two conditions has been
dropped.
"""
from __future__ import annotations

import pytest

from aura_web.discord_api import PartialGuild
from aura_web.guild_selection import select_manageable_guilds

MANAGE = "32"
ADMIN = "8"
NOTHING = "2048"


def guild(guild_id: str, name: str = "Server", permissions: str = MANAGE) -> PartialGuild:
    return PartialGuild(id=guild_id, name=name, icon=None, permissions=permissions)


class TestBothConditionsRequired:
    def test_manageable_and_aura_present_is_included(self) -> None:
        selected = select_manageable_guilds([guild("1")], frozenset({"1"}))

        assert [entry.id for entry in selected] == ["1"]

    def test_manageable_but_aura_absent_is_excluded(self) -> None:
        selected = select_manageable_guilds([guild("1")], frozenset())

        assert selected == []

    def test_aura_present_but_not_manageable_is_excluded(self) -> None:
        selected = select_manageable_guilds(
            [guild("1", permissions=NOTHING)], frozenset({"1"})
        )

        assert selected == []

    def test_neither_condition_is_excluded(self) -> None:
        selected = select_manageable_guilds(
            [guild("1", permissions=NOTHING)], frozenset({"2"})
        )

        assert selected == []

    def test_administrator_counts_as_manageable(self) -> None:
        selected = select_manageable_guilds([guild("1", permissions=ADMIN)], frozenset({"1"}))

        assert [entry.id for entry in selected] == ["1"]

    def test_only_the_qualifying_guild_survives_a_mixed_list(self) -> None:
        selected = select_manageable_guilds(
            [
                guild("1", "Yes", MANAGE),
                guild("2", "No bot", MANAGE),
                guild("3", "No rights", NOTHING),
                guild("4", "Neither", NOTHING),
            ],
            frozenset({"1", "3"}),
        )

        assert [entry.id for entry in selected] == ["1"]


class TestEmptyResults:
    def test_no_guilds_at_all_gives_an_empty_list(self) -> None:
        assert select_manageable_guilds([], frozenset({"1"})) == []

    def test_aura_in_nothing_gives_an_empty_list(self) -> None:
        assert select_manageable_guilds([guild("1")], frozenset()) == []

    def test_an_empty_result_is_a_list_not_an_error(self) -> None:
        """The brief's requirement: a user who manages nothing sees a list, not a failure."""
        result = select_manageable_guilds([guild("1", permissions=NOTHING)], frozenset({"1"}))

        assert result == []
        assert isinstance(result, list)


class TestMalformedPermissions:
    @pytest.mark.parametrize("permissions", ["", "abc", "-1", "0x20", "9" * 100])
    def test_an_unparseable_bitmask_denies_rather_than_grants(self, permissions: str) -> None:
        selected = select_manageable_guilds(
            [guild("1", permissions=permissions)], frozenset({"1"})
        )

        assert selected == []


class TestOutputShape:
    def test_the_permission_bitmask_is_not_carried_to_the_browser(self) -> None:
        """The decision is the server's; shipping its input invites a client-side redo."""
        selected = select_manageable_guilds([guild("1")], frozenset({"1"}))

        assert not hasattr(selected[0], "permissions")

    def test_results_are_sorted_by_name_case_insensitively(self) -> None:
        selected = select_manageable_guilds(
            [guild("1", "zebra"), guild("2", "Apple"), guild("3", "banana")],
            frozenset({"1", "2", "3"}),
        )

        assert [entry.name for entry in selected] == ["Apple", "banana", "zebra"]

    def test_identical_names_are_ordered_stably_by_id(self) -> None:
        selected = select_manageable_guilds(
            [guild("20", "Same"), guild("3", "Same")], frozenset({"3", "20"})
        )

        assert [entry.id for entry in selected] == ["20", "3"]

    def test_the_same_input_always_produces_the_same_order(self) -> None:
        guilds = [guild(str(index), f"Server {index % 3}") for index in range(20)]
        bot_guilds = frozenset(str(index) for index in range(20))

        first = select_manageable_guilds(guilds, bot_guilds)
        second = select_manageable_guilds(list(reversed(guilds)), bot_guilds)

        assert [entry.id for entry in first] == [entry.id for entry in second]
