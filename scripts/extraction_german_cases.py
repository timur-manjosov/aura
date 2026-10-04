"""German evaluation batches for fact extraction with noise, jokes, quotes and corrections (P4).

The model bake-off compares extraction models on scripts/extraction_eval_cases.py
plus these batches. That set has nineteen German messages; P4 asks for German
chat as it really reads -- noise, jokes, quotes of someone else's words, and
corrections of a value just posted -- so these batches add exactly those
shapes. Everything is invented; the repository is public.

Labels follow extraction_eval_cases.py: `expect_fact` is True when the message
asserts something checkable about the server (a rule, a time, a change, a
milestone) that a member could later ask about, False otherwise. A correction
is a fact with the CORRECTED value; the wrong value is a forbidden substring of
the batch, so a distilled sentence that keeps it is an invented (stale) fact.
Every batch carries at least one genuine fact as a control, so a model that
extracts nothing cannot score well.
"""

from __future__ import annotations

from typing import Final

from extraction_eval_cases import EvalBatch, EvalMessage

GERMAN_BATCHES: Final[tuple[EvalBatch, ...]] = (
    EvalBatch(
        name="de-noise",
        locale="de",
        channel_name="allgemein",
        messages=(
            EvalMessage("hahaha", False, "laughter"),
            EvalMessage("ok", False, "acknowledgement"),
            EvalMessage("gn8 leute", False, "greeting"),
            EvalMessage(
                "Ab nächster Woche ist der Filmabend immer donnerstags um 20 Uhr statt freitags.",
                True,
                "control: a schedule change",
            ),
            EvalMessage("😂😂😂", False, "emoji only"),
            EvalMessage("wer ist noch wach", False, "question, no fact"),
        ),
    ),
    EvalBatch(
        name="de-jokes",
        locale="de",
        channel_name="off-topic",
        messages=(
            EvalMessage(
                "ab heute ist kaffee pflicht für alle mods lol", False, "joke phrased as a rule"
            ),
            EvalMessage(
                "neue regel: wer verliert, muss den nächsten raid bezahlen 😂",
                False,
                "joke phrased as a rule",
            ),
            EvalMessage(
                "Die Spendenaktion hat 800 Euro für das Tierheim gesammelt.",
                True,
                "control: a milestone",
            ),
            EvalMessage(
                "der server wird bestimmt nie wieder gewartet, so wie der läuft xD",
                False,
                "sarcasm",
            ),
            EvalMessage("ich bin ab jetzt offiziell der beste spieler hier", False, "boast"),
        ),
    ),
    EvalBatch(
        name="de-quotes",
        locale="de",
        channel_name="allgemein",
        messages=(
            EvalMessage(
                "mein kumpel meinte, auf seinem server sind memes komplett verboten",
                False,
                "another server's rule, reported",
            ),
            EvalMessage(
                "jemand hat gefragt „gibt es morgen ein turnier?“ – keine ahnung",
                False,
                "quoted question",
            ),
            EvalMessage(
                "Laut Ankündigung von Admin Kim beginnt das Wintertunier am 6. Januar.",
                True,
                "control: an announcement relayed with its source",
            ),
            EvalMessage(
                "im film sagt er „ab morgen gilt das kriegsrecht“, so geil",
                False,
                "a quote from a film",
            ),
        ),
    ),
    EvalBatch(
        name="de-corrections",
        locale="de",
        channel_name="ankündigungen",
        messages=(
            EvalMessage(
                "Das Quiz am Samstag startet um 18 Uhr.", True, "first value, corrected below"
            ),
            EvalMessage(
                "Korrektur: das Quiz startet um 19 Uhr, nicht um 18 Uhr. Sorry!",
                True,
                "the correction: 19 Uhr",
            ),
            EvalMessage("danke für die info", False, "acknowledgement"),
        ),
        forbidden_substrings=(),
    ),
    EvalBatch(
        name="de-corrections-single",
        locale="de",
        channel_name="ankündigungen",
        messages=(
            EvalMessage(
                "Kleine Korrektur zu gestern: Die Anmeldung zum Bastelwettbewerb läuft bis "
                "zum 12. Mai, nicht bis zum 10. Mai.",
                True,
                "a correction of yesterday's value: 12. Mai",
            ),
            EvalMessage("ah ok gut zu wissen", False, "acknowledgement"),
        ),
        forbidden_substrings=("bis zum 10. mai",),
    ),
    EvalBatch(
        name="de-hypothetical",
        locale="de",
        channel_name="vorschläge",
        messages=(
            EvalMessage(
                "wäre cool wenn es einen eigenen kanal für fotos gäbe",
                False,
                "a wish, not a fact",
            ),
            EvalMessage(
                "vielleicht machen wir nächsten monat ein turnier, mal sehen",
                False,
                "an uncertain plan",
            ),
            EvalMessage(
                "Vorschläge für neue Kanäle werden ab sofort per Umfrage im Kanal #abstimmung "
                "entschieden.",
                True,
                "control: a procedure",
            ),
            EvalMessage("sollen wir abstimmen ob es mehr emojis gibt?", False, "a question"),
        ),
    ),
)
