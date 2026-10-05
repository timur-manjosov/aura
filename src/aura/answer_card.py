"""Answer cards: the look of Aura's messages in the v2 answer format, rendered by code.

A validated answer contract (`aura.answer_contract.ContractAnswer`) goes in; an
`AnswerCard` comes out -- a plain description of what the reader sees, with
every piece of untrusted text already collapsed, bounded and escaped. Three thin
adapters turn a card into something Discord can show:

* `card_to_embed` -- a refined classic embed (CardStyle.EMBED, the default):
  accent colour, the question in the small author line, the lead and the points
  in the description, the code-rendered notes in italics, one sources field,
  a quiet footer.
* `card_to_layout_view` -- the same card as a Components V2 container
  (CardStyle.CONTAINER): subtext for the quiet parts and a divider before the
  sources. Always sent with mentions disabled.
* `card_to_plain_text` -- the same card as one plain message, the fallback when
  Discord refuses a card.

What a model wrote and a check must read is kept apart from the markup:
`AnswerCard.checked_lead` and `AnswerCard.checked_points` are exactly the lead
and the points the card displays, unescaped, each point with the facts it cites.
The notes (the "not recorded" line and the conflict or "unclear" caveat) are not
among them: they are templates filled by code, not prose.

Invariants this module maintains
--------------------------------
* Model-written and member-written text is collapsed to one line, cut to its
  bound and only then escaped (`aura.rendering`), so it can never become
  markup: no masked link, no mention, channel or timestamp token, no heading,
  no mass mention. Plain-text slots Discord does not render as markdown (the
  embed's author line and footer) are collapsed and bounded, not escaped.
* A rendered card fits Discord's limits by construction: description at most
  `DESCRIPTION_MAX_CHARS` (points are dropped from the end, whole, never cut),
  each sources field at most 1,024 characters, the embed under 6,000 in total,
  a layout view under 4,000 text characters and 40 components, a plain-text
  fallback under 2,000 characters.
* What the card shows and what `checked_points` holds are built from the same
  list of displayed points; they cannot drift apart.

Pure and Discord-connection-free apart from building `discord.Embed` and
`discord.ui` objects, which needs no gateway. Imports no retrieval, grounding,
database or LLM module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import discord

from aura.answer_contract import AnswerPoint, ContractAnswer, RelationKind
from aura.db.models import Fact
from aura.i18n import t
from aura.rendering import (
    collapse_display_text,
    discord_timestamp,
    escape_display_markdown,
    inline_fact_text,
    link_label,
    shorten,
    source_link,
)
from aura.theme import (
    ACCENT_COLORS,
    BULLET,
    CHANNEL_LABEL_MAX_CHARS,
    CITATION_SEPARATOR,
    COMPONENTS_V2_TEXT_LIMIT,
    DESCRIPTION_MAX_CHARS,
    EMBED_AUTHOR_NAME_LIMIT,
    EMBED_FIELD_COUNT_LIMIT,
    EMBED_FIELD_NAME_LIMIT,
    EMBED_FIELD_VALUE_LIMIT,
    EMBED_FOOTER_LIMIT,
    EMBED_TOTAL_LIMIT,
    KIND_SYMBOLS,
    MAX_POINTS,
    PROACTIVE_MAX_POINTS,
    QUESTION_DISPLAY_CHARS,
    SOURCE_SEPARATOR,
    SUPERSCRIPT_DIGITS,
    MessageKind,
)

# Discord's own cap on a plain message's content.
PLAIN_MESSAGE_LIMIT: Final = 2000

# The note each relation kind adds to a card, by translation key. A
# complementary pair needs none: merging it is the answer.
_RELATION_NOTE_KEYS: Final[dict[RelationKind, str]] = {
    RelationKind.SAME_DETAIL_CONFLICT: "answer_note_conflict",
    RelationKind.UNCLEAR_IF_SAME: "answer_note_unclear",
}


@dataclass(frozen=True)
class CardItem:
    """One line of a card section: the main text and an optional quiet line under it.

    Attributes
    ----------
    text
        Markdown, already collapsed, bounded and escaped by the builder.
    meta
        Markdown for the quiet line (a source link and a date), or None.
    """

    text: str
    meta: str | None = None


@dataclass(frozen=True)
class CardSection:
    """A titled group of lines in a card (P5: the digest, onboarding and /aura-plan).

    Attributes
    ----------
    heading
        Plain text, collapsed and bounded; shown in bold (or as an embed
        field's name, where no markdown renders).
    items
        The lines shown, in order.
    hidden
        How many further lines exist but are not shown.
    more_template
        A trusted, localized template with a `{count}` placeholder, shown as a
        last line when `hidden` is above zero; None shows nothing.
    numbered
        Number the lines (1., 2., ...) instead of bulleting them.
    """

    heading: str
    items: tuple[CardItem, ...]
    hidden: int = 0
    more_template: str | None = None
    numbered: bool = False


@dataclass(frozen=True)
class AnswerCard:
    """Everything one Aura message shows, already escaped and bounded.

    Attributes
    ----------
    kind
        The kind of message; picks the accent colour.
    top_line
        Plain text for the top of the card -- the kind's symbol and the
        question, or proactive relief's framing label -- or None. Collapsed and
        bounded, not escaped.
    paragraphs
        Markdown paragraphs of the body, in order: the lead, or a note.
    bullets
        Markdown lines shown as a bulleted list after the paragraphs, without
        the bullet character.
    notes
        Markdown lines shown in a subdued style after the bullets: the
        conflict or "unclear" caveat, then the "not recorded" line.
    sources_label
        The heading of the sources block, or None when there are no sources.
    sources
        One markdown line per cited fact, numbered as the citations are.
    footer
        Plain text for the quiet footer, or None.
    checked_lead
        The lead exactly as displayed, unescaped; None for no-model cards.
    checked_points
        The points exactly as displayed, unescaped, each with the real IDs of
        the facts it cites.
    cited_fact_ids
        The cited facts' real IDs in display order: citation 1 is the first.
    sections
        Titled groups of lines after the body (P5); empty for every answer card,
        which then renders exactly as before sections existed.
    """

    kind: MessageKind
    top_line: str | None
    paragraphs: tuple[str, ...]
    bullets: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    sources_label: str | None = None
    sources: tuple[str, ...] = ()
    footer: str | None = None
    checked_lead: str | None = None
    checked_points: tuple[AnswerPoint, ...] = ()
    cited_fact_ids: tuple[int, ...] = ()
    sections: tuple[CardSection, ...] = ()


def question_line(question: str | None) -> str | None:
    """Return the question as the top line of a card, or None when there is none.

    Parameters
    ----------
    question
        The asker's question, as received.

    Returns
    -------
    str or None
        The answer symbol and the question on one line, the question cut to
        `QUESTION_DISPLAY_CHARS`; None for a missing or blank question.
    """
    if question is None:
        return None
    collapsed = collapse_display_text(question)
    if not collapsed:
        return None
    return f"{KIND_SYMBOLS[MessageKind.ANSWER]} {shorten(collapsed, QUESTION_DISPLAY_CHARS)}"


def _superscript(number: int) -> str:
    return str(number).translate(SUPERSCRIPT_DIGITS)


def _citation_links(
    fact_ids: Sequence[int], display_numbers: Mapping[int, int], facts_by_id: Mapping[int, Fact]
) -> str:
    return CITATION_SEPARATOR.join(
        f"[{_superscript(display_numbers[fact_id])}]({source_link(facts_by_id[fact_id])})"
        for fact_id in fact_ids
    )


def _source_line(number: int, fact: Fact, channel_names: Mapping[int, str], locale: str) -> str:
    """Return one sources line: the number, the channel as a link, the recording date.

    Notes
    -----
    A channel the resolver could not name comes back as its own ID (see
    aura.discord_context); a raw ID is not a label anyone can read, so the line
    says "message" instead. The date is a Discord timestamp token, rendered in
    each reader's own locale.
    """
    name = channel_names.get(fact.channel_id)
    if name and name != str(fact.channel_id):
        label = link_label(f"#{name}", CHANNEL_LABEL_MAX_CHARS)
    else:
        label = link_label(t("answer_source_message", locale), CHANNEL_LABEL_MAX_CHARS)
    return (
        f"`{number}` [{label}]({source_link(fact)}){SOURCE_SEPARATOR}"
        f"{discord_timestamp(fact.created_at)}"
    )


def compose_description(card: AnswerCard, *, notes_as_subtext: bool) -> str:
    """Join a card's body into one markdown text.

    Parameters
    ----------
    card
        The card.
    notes_as_subtext
        Render the notes as Discord subtext (`-#`, Components V2) instead of
        italics (embeds, where subtext is not documented).

    Returns
    -------
    str
        Paragraphs, then the bulleted points, then the notes, blocks separated
        by blank lines.
    """
    parts = list(card.paragraphs)
    if card.bullets:
        parts.append("\n".join(f"{BULLET} {bullet}" for bullet in card.bullets))
    if card.notes:
        parts.append(
            "\n".join(f"-# {note}" if notes_as_subtext else f"*{note}*" for note in card.notes)
        )
    return "\n\n".join(parts)


def plain_answer_text(card: AnswerCard) -> str:
    """Return a card's lead and points as plain text with numbered citations.

    Parameters
    ----------
    card
        An answer card.

    Returns
    -------
    str
        The lead on the first line, then one "- text [1, 2]" line per displayed
        point, numbered as the card's sources are; empty for a card without a
        lead. The notes are not included: they are code templates.
    """
    if card.checked_lead is None:
        return ""
    display_numbers = {fact_id: n for n, fact_id in enumerate(card.cited_fact_ids, start=1)}
    lines = [card.checked_lead]
    lines.extend(
        f"- {point.text} [{', '.join(str(display_numbers[f]) for f in point.fact_ids)}]"
        for point in card.checked_points
    )
    return "\n".join(lines)


def _answer_notes(answer: ContractAnswer, locale: str) -> tuple[str, ...]:
    """Return the code-rendered notes of an answer: its caveats, then its gaps."""
    notes = [
        t(key, locale) for kind, key in _RELATION_NOTE_KEYS.items() if answer.has_relation(kind)
    ]
    if answer.not_covered_topics:
        topics = t("answer_list_separator", locale).join(
            escape_display_markdown(topic, at_line_start=False)
            for topic in answer.not_covered_topics
        )
        notes.append(t("answer_not_recorded", locale, topics=topics))
    return tuple(notes)


def build_answer_card(
    answer: ContractAnswer,
    facts: Sequence[Fact],
    *,
    question: str | None,
    locale: str,
    channel_names: Mapping[int, str],
    proactive: bool = False,
) -> AnswerCard:
    """Render a validated contract answer as a card.

    Parameters
    ----------
    answer
        The validated answer. It must cite at least one fact; an answer that
        cites none is shown as the "no information" notice instead (see
        aura.commands.ask).
    facts
        The facts the answer was synthesized from; every cited ID must be here.
    question
        The asker's question, shown at the top; ignored for proactive relief,
        whose top line is its framing label.
    locale
        The reader's locale, for every label.
    channel_names
        Channel names by CHANNEL id (aura.discord_context.fact_channel_names).
    proactive
        Render proactive relief's variant: its framing label and footer, and at
        most `PROACTIVE_MAX_POINTS` points.

    Returns
    -------
    AnswerCard
        The card; `checked_lead` and `checked_points` are exactly what it shows.

    Raises
    ------
    ValueError
        If the answer cites no fact, or a fact that is not in `facts`.

    Notes
    -----
    Citations are renumbered in display order: the first fact the answer uses
    is source 1, whatever its number in the prompt was, so the reader sees
    1, 2, 3 and never a gap. Points that would push the description past
    `DESCRIPTION_MAX_CHARS` are dropped from the end, whole -- a point is never
    cut, since a cut can drop the qualifier that made it true.
    """
    if not answer.used_fact_ids:
        raise ValueError("an answer card needs at least one cited fact")
    facts_by_id = {fact.id: fact for fact in facts}
    missing = [fact_id for fact_id in answer.used_fact_ids if fact_id not in facts_by_id]
    if missing:
        raise ValueError(f"the answer cites fact(s) {missing} that were not supplied")
    cited = [facts_by_id[fact_id] for fact_id in answer.used_fact_ids]
    display_numbers = {fact.id: number for number, fact in enumerate(cited, start=1)}

    kind = MessageKind.PROACTIVE if proactive else MessageKind.ANSWER
    notes = _answer_notes(answer, locale)
    sources = tuple(
        _source_line(number, fact, channel_names, locale)
        for number, fact in enumerate(cited, start=1)
    )
    shown = list(answer.points[: PROACTIVE_MAX_POINTS if proactive else MAX_POINTS])

    def _card(points: Sequence[AnswerPoint]) -> AnswerCard:
        return AnswerCard(
            kind=kind,
            top_line=(
                collapse_display_text(t("proactive_reply_label", locale))
                if proactive
                else question_line(question)
            ),
            paragraphs=(escape_display_markdown(answer.lead),),
            bullets=tuple(
                f"{escape_display_markdown(point.text)} "
                f"{_citation_links(point.fact_ids, display_numbers, facts_by_id)}"
                for point in points
            ),
            notes=notes,
            sources_label=t("ask_sources_label", locale),
            sources=sources,
            footer=t("proactive_reply_footer", locale) if proactive else t("answer_footer", locale),
            checked_lead=answer.lead,
            checked_points=tuple(points),
            cited_fact_ids=tuple(fact.id for fact in cited),
        )

    card = _card(shown)
    while shown and len(compose_description(card, notes_as_subtext=False)) > DESCRIPTION_MAX_CHARS:
        shown.pop()
        card = _card(shown)
    return card


def build_fact_list_card(
    kind: MessageKind, note: str, facts: Sequence[Fact], *, question: str | None
) -> AnswerCard:
    """Render a no-model reply: a note, then each fact verbatim with its source and date.

    Parameters
    ----------
    kind
        RELATED (nothing answers, these may be related) or LIMIT (a daily cap
        was reached).
    note
        The localized note above the list; trusted template text.
    facts
        The facts to list, best first.
    question
        The asker's question, shown at the top.

    Returns
    -------
    AnswerCard
        The card. Facts that would push the description past
        `DESCRIPTION_MAX_CHARS` are left out from the end.

    Notes
    -----
    Each fact line is exactly the legacy free answer's (`aura.rendering`): the
    fact's own sentence as a link to its source message, and its date. Nothing
    in it is model-written.
    """
    lines = [
        f"[{inline_fact_text(fact.content)}]({source_link(fact)}){SOURCE_SEPARATOR}"
        f"{discord_timestamp(fact.created_at)}"
        for fact in facts
    ]

    def _card(shown: Sequence[str]) -> AnswerCard:
        return AnswerCard(
            kind=kind,
            top_line=question_line(question),
            paragraphs=(f"{KIND_SYMBOLS[kind]} {note}",),
            bullets=tuple(shown),
        )

    card = _card(lines)
    while lines and len(compose_description(card, notes_as_subtext=False)) > DESCRIPTION_MAX_CHARS:
        lines.pop()
        card = _card(lines)
    return card


def build_notice_card(kind: MessageKind, text: str, *, question: str | None) -> AnswerCard:
    """Render a one-line notice: an error, or "nothing recorded on that".

    Parameters
    ----------
    kind
        ERROR, or RELATED for the plain "no information" reply.
    text
        The localized notice; trusted template text.
    question
        The asker's question, shown at the top.

    Returns
    -------
    AnswerCard
        A card with the kind's symbol before the notice.
    """
    return AnswerCard(
        kind=kind, top_line=question_line(question), paragraphs=(f"{KIND_SYMBOLS[kind]} {text}",)
    )


def _sources_fields(lines: Sequence[str]) -> list[str]:
    """Split source lines into field values of at most `EMBED_FIELD_VALUE_LIMIT` characters."""
    values: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line])
        if current and len(candidate) > EMBED_FIELD_VALUE_LIMIT:
            values.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        values.append("\n".join(current))
    # A single line is bounded far below the limit, but a value Discord would
    # refuse is never sent: cut as a last resort.
    return [shorten(value, EMBED_FIELD_VALUE_LIMIT) for value in values]


def section_lines(section: CardSection, *, subtext: bool) -> list[str]:
    """Return the markdown lines of one section.

    Parameters
    ----------
    section
        The section.
    subtext
        Put each item's quiet line on its own line as Discord subtext
        (Components V2); otherwise append it after a separator (embeds, where
        subtext is not documented).

    Returns
    -------
    list[str]
        One entry per item, then the "and N more" line when lines are hidden.
    """
    lines: list[str] = []
    for number, item in enumerate(section.items, start=1):
        marker = f"{number}." if section.numbered else BULLET
        if item.meta is None:
            lines.append(f"{marker} {item.text}")
        elif subtext:
            lines.append(f"{marker} {item.text}\n-# {item.meta}")
        else:
            lines.append(f"{marker} {item.text}{SOURCE_SEPARATOR}{item.meta}")
    if section.hidden > 0 and section.more_template:
        lines.append(section.more_template.format(count=section.hidden))
    return lines


def _without_last_item(card: AnswerCard) -> AnswerCard | None:
    """Return `card` with one line fewer in its longest section, or None when none is left."""
    if not card.sections:
        return None
    longest = max(range(len(card.sections)), key=lambda index: len(card.sections[index].items))
    section = card.sections[longest]
    if not section.items:
        return None
    shorter = CardSection(
        heading=section.heading,
        items=section.items[:-1],
        hidden=section.hidden + 1,
        more_template=section.more_template,
        numbered=section.numbered,
    )
    sections = (*card.sections[:longest], shorter, *card.sections[longest + 1 :])
    return replace(card, sections=sections)


def _section_fields(section: CardSection) -> list[tuple[str, str]]:
    """Return one section as embed fields (name, value), split at the value limit."""
    name = shorten(collapse_display_text(section.heading) or "·", EMBED_FIELD_NAME_LIMIT)
    values: list[str] = []
    current: list[str] = []
    for line in section_lines(section, subtext=False):
        candidate = "\n".join([*current, line])
        if current and len(candidate) > EMBED_FIELD_VALUE_LIMIT:
            values.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        values.append("\n".join(current))
    return [(name, shorten(value, EMBED_FIELD_VALUE_LIMIT)) for value in values]


def card_to_embed(card: AnswerCard) -> discord.Embed:
    """Render a card as a classic embed (CardStyle.EMBED).

    Parameters
    ----------
    card
        The card.

    Returns
    -------
    discord.Embed
        Accent colour of the card's kind; the top line as the author line; the
        body as the description; the sources as one or more fields under the
        same heading; the footer.

    Notes
    -----
    The author line and the footer are plain-text slots: Discord renders no
    markdown in them, which is why the question can sit there unescaped.
    Mentions inside an embed never notify anyone.
    """
    embed = discord.Embed(
        description=compose_description(card, notes_as_subtext=False),
        color=ACCENT_COLORS[card.kind],
    )
    if card.top_line:
        embed.set_author(name=shorten(card.top_line, EMBED_AUTHOR_NAME_LIMIT))
    if card.sections:
        return _embed_with_sections(embed, card)
    if card.sources and card.sources_label:
        for value in _sources_fields(card.sources):
            embed.add_field(name=card.sources_label, value=value, inline=False)
    if card.footer:
        embed.set_footer(text=shorten(card.footer, EMBED_FOOTER_LIMIT))
    return embed


def _embed_with_sections(base: discord.Embed, card: AnswerCard) -> discord.Embed:
    """Finish an embed for a card with sections, dropping lines until it fits.

    Notes
    -----
    Each section becomes one or more fields under its heading. Lines are taken
    from the end of the longest section, one at a time, until the embed is
    within Discord's 6,000-character total and 25-field limits; the section
    then says how many lines it no longer shows.
    """
    current: AnswerCard | None = card
    while current is not None:
        embed = base.copy()
        fields = [field for section in current.sections for field in _section_fields(section)]
        source_fields = (
            [(current.sources_label, value) for value in _sources_fields(current.sources)]
            if current.sources and current.sources_label
            else []
        )
        for name, value in [*fields, *source_fields][:EMBED_FIELD_COUNT_LIMIT]:
            embed.add_field(name=name, value=value, inline=False)
        if current.footer:
            embed.set_footer(text=shorten(current.footer, EMBED_FOOTER_LIMIT))
        if len(embed) <= EMBED_TOTAL_LIMIT and len(fields) + len(source_fields) <= (
            EMBED_FIELD_COUNT_LIMIT
        ):
            return embed
        current = _without_last_item(current)
    # Every line is gone and the frame alone is still too large: show the frame.
    embed = base.copy()
    if card.footer:
        embed.set_footer(text=shorten(card.footer, EMBED_FOOTER_LIMIT))
    return embed


def _layout_texts(card: AnswerCard) -> tuple[str | None, str, str | None, str | None]:
    """Return a layout view's four text blocks: top line, body, sources, footer."""
    top = f"-# {escape_display_markdown(card.top_line)}" if card.top_line else None
    body = compose_description(card, notes_as_subtext=True)
    sources = None
    if card.sources and card.sources_label:
        sources = "\n".join(
            [f"-# **{escape_display_markdown(card.sources_label)}**"]
            + [f"-# {line}" for line in card.sources]
        )
    footer = f"-# {escape_display_markdown(card.footer)}" if card.footer else None
    return top, body, sources, footer


def components_v2_available() -> bool:
    """Report whether the installed discord.py can build Components V2 layouts (2.6 or later).

    Returns
    -------
    bool
        True when `discord.ui` has the layout classes `card_to_layout_view` uses.

    Notes
    -----
    requirements.txt asks for discord.py 2.4 or later, and the image's
    dependency layer is cached on that file, so an older image can carry a
    discord.py without these classes. The container style is then unavailable
    rather than broken: the senders fall back to the embed.
    """
    return all(
        hasattr(discord.ui, name)
        for name in ("LayoutView", "Container", "TextDisplay", "Separator")
    )


def _section_texts(card: AnswerCard) -> list[str]:
    """Return one text block per section: the bold heading, then its lines."""
    return [
        "\n".join(
            [
                f"**{escape_display_markdown(section.heading)}**",
                *section_lines(section, subtext=True),
            ]
        )
        for section in card.sections
    ]


def layout_text_length(card: AnswerCard) -> int:
    """Return how many text characters `card_to_layout_view` would send for `card`."""
    return sum(len(text) for text in _layout_texts(card) if text) + sum(
        len(text) for text in _section_texts(card)
    )


def card_to_layout_view(card: AnswerCard, *, caption: str | None = None) -> discord.ui.LayoutView:
    """Render a card as a Components V2 container (CardStyle.CONTAINER).

    Parameters
    ----------
    card
        The card.
    caption
        Optional trusted markdown shown above the container, outside it -- the
        operator preview's label for each sample. None for every real answer.

    Returns
    -------
    discord.ui.LayoutView
        One container in the kind's accent colour: the top line as subtext,
        the body, a divider and the sources as subtext, the footer as subtext.
        At most six components and, by the card's own bounds, far fewer than
        4,000 text characters; a footer that would cross that limit is left
        out, then the sources, then the top line.

    Notes
    -----
    A message carrying this view cannot also carry an embed or plain content,
    and a mention inside a text display can notify: send it with
    `discord.AllowedMentions.none()`. Untrusted text in the card is already
    escaped, so no mention token can form in the first place; the
    allowed-mentions setting is the second lock.
    """
    if card.sections:
        return _layout_view_with_sections(card, caption=caption)
    top, body, sources, footer = _layout_texts(card)
    blocks: list[str | None] = [top, body, sources, footer]
    budget = COMPONENTS_V2_TEXT_LIMIT - (len(caption) if caption else 0)
    while sum(len(text) for text in blocks if text) > budget:
        for index in (3, 2, 0):
            if blocks[index]:
                blocks[index] = None
                break
        else:
            blocks[1] = shorten(body, max(budget, 1))
    shown_top, shown_body, shown_sources, shown_footer = blocks
    container: discord.ui.Container[discord.ui.LayoutView] = discord.ui.Container(
        accent_colour=ACCENT_COLORS[card.kind]
    )
    if shown_top:
        container.add_item(discord.ui.TextDisplay(shown_top))
    container.add_item(discord.ui.TextDisplay(shown_body or "…"))
    if shown_sources:
        container.add_item(
            discord.ui.Separator(visible=True, spacing=discord.SeparatorSpacing.small)
        )
        container.add_item(discord.ui.TextDisplay(shown_sources))
    if shown_footer:
        container.add_item(discord.ui.TextDisplay(shown_footer))
    view = discord.ui.LayoutView(timeout=None)
    if caption:
        view.add_item(discord.ui.TextDisplay(caption))
    view.add_item(container)
    return view


def _layout_view_with_sections(card: AnswerCard, *, caption: str | None) -> discord.ui.LayoutView:
    """Render a card with sections as one container, dropping lines until it fits.

    Notes
    -----
    Top line, body, one text display per section, a divider and the sources,
    the footer -- at most six plus the number of sections components. Lines go
    from the end of the longest section, one at a time, until the text is under
    4,000 characters; the section then says how many it no longer shows.
    """
    budget = COMPONENTS_V2_TEXT_LIMIT - (len(caption) if caption else 0)
    current: AnswerCard = card
    while layout_text_length(current) > budget:
        shorter = _without_last_item(current)
        if shorter is None:
            break
        current = shorter
    top, body, sources, footer = _layout_texts(current)
    container: discord.ui.Container[discord.ui.LayoutView] = discord.ui.Container(
        accent_colour=ACCENT_COLORS[current.kind]
    )
    if top:
        container.add_item(discord.ui.TextDisplay(top))
    if body:
        container.add_item(discord.ui.TextDisplay(body))
    for text in _section_texts(current):
        container.add_item(discord.ui.TextDisplay(shorten(text, max(budget, 1))))
    if sources:
        container.add_item(
            discord.ui.Separator(visible=True, spacing=discord.SeparatorSpacing.small)
        )
        container.add_item(discord.ui.TextDisplay(sources))
    if footer:
        container.add_item(discord.ui.TextDisplay(footer))
    view = discord.ui.LayoutView(timeout=None)
    if caption:
        view.add_item(discord.ui.TextDisplay(caption))
    view.add_item(container)
    return view


def card_to_plain_text(card: AnswerCard) -> str:
    """Render a card as one plain message, the fallback when Discord refuses a card.

    Parameters
    ----------
    card
        The card.

    Returns
    -------
    str
        The top line, the body with bullets and notes, and the sources, one
        block per paragraph, at most `PLAIN_MESSAGE_LIMIT` characters. The
        footer is left out; the sources are left out before the body is cut.

    Notes
    -----
    A plain message renders markdown and can notify: send it with
    `discord.AllowedMentions.none()`. The top line is escaped here because,
    unlike an embed's author line, plain content renders markdown.
    """
    parts = [f"**{escape_display_markdown(card.top_line)}**"] if card.top_line else []
    parts.append(compose_description(card, notes_as_subtext=False))
    parts.extend(
        "\n".join(
            [
                f"**{escape_display_markdown(section.heading)}**",
                *section_lines(section, subtext=False),
            ]
        )
        for section in card.sections
    )
    with_sources = list(parts)
    if card.sources and card.sources_label:
        with_sources.append("\n".join([card.sources_label, *card.sources]))
    text = "\n\n".join(with_sources)
    if len(text) <= PLAIN_MESSAGE_LIMIT:
        return text
    return shorten("\n\n".join(parts), PLAIN_MESSAGE_LIMIT)
