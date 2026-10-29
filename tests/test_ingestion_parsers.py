"""Tests for the wiki markup cleaner and the page parsers.

Everything here runs on inline markup, so the chunking policy is pinned down
without touching the network or a database.
"""

from __future__ import annotations

from app.ingestion import (
    ArticleParser,
    GraffitiParser,
    ParserKind,
    QuotesParser,
    WikiSource,
    WikitextCleaner,
)
from app.schemas import LoreChunkCreate
from app.taxonomy import LoreTopic, SourceType

GRAFFITI_MARKUP = """
{| style="width: 80%;"
| style="padding: 2px;" | '''"Hold up."'''
|}

{{pix}}
[[Image:Witch.jpg|thumb|Tips left by survivors.]]
'''Graffiti''' is found all over the campaigns.

== ''Left 4 Dead'' ==
=== [[No Mercy]] ===
==== [[The Subway]] ====
*NO CURE (scrawled over a quarantine poster)
*It takes at least 4 days to change
:*No it doesn't
*<u>ENTIRE</u> CITY IS GONE – READ THIS &amp; '''RUN'''

==== [[Rooftop Finale]] ====
*NO ZOMBIE IS SAFE FROM CHICAGO TED<ref name="x">A footnote.</ref>

=== [[Dead Air]] ===
==== [[The Alleys]] ====
(note- These writings can be found in a normal room)

*THE ARMY LEFT A HOWITZER
"""

INFOBOX_MARKUP = """{{Infobox infected
|Image = <gallery>
Tank.png
</gallery>
|Name = The Tank
|Health = 3000 <small>(Easy)</small><br>4000 <small>(Normal)</small>
|Abilities = Immense strength
}}
{{pix}}
The Tank is a Special Infected that throws chunks of concrete."""

ARTICLE_MARKUP = """
The '''Green Flu''' is the virus responsible for the infection.<ref>Source.</ref>
It is highly contagious.

== Origin ==
The origin of the virus is unknown. Several theories circulate among survivors,
blaming the government, CEDA or a biological weapon that escaped a laboratory.

== Statistics ==
{| class="wikitable"
! Name !! Value
|-
| Health || 4000
|-
| Damage || 60
|-
| Speed || slow, but relentless once it has picked a target in the open
|-
| Threat || the only infected that reliably forces a team to relocate mid-horde
|}
"""

QUOTES_MARKUP = """
{{game grammar}}
A list of [[Ellis]]' '''Quotes'''.

== Intro ==
*"Kill all sons of bitches. That's my official instructions."
*"Ugh!"
*"Watch out!"
*"I like to keep my head down and my gun up." (2 versions)
*''[When a Tank appears]''

== General ==
'''[After closing the [[Safe Room]] door]'''
*"We are a right damn good team!"
*"Oh, hell yeah!"
*"We opened up a can of whoop-ass on them!"
*''(laughs)'' I'm thinking we can take on anything.
"""


def graffiti_source() -> WikiSource:
    return WikiSource("Graffiti", LoreTopic.GRAFFITI, ParserKind.GRAFFITI)


def article_source() -> WikiSource:
    return WikiSource("Green Flu", LoreTopic.INFECTION, ParserKind.ARTICLE)


def quotes_source() -> WikiSource:
    return WikiSource(
        "Ellis/Quotes",
        LoreTopic.QUOTES,
        ParserKind.QUOTES,
        character_name="Эллис",
    )


def parse_graffiti(markup: str = GRAFFITI_MARKUP) -> list[LoreChunkCreate]:
    """Run the graffiti parser on ``markup`` as if it came from the wiki."""
    return GraffitiParser(WikitextCleaner()).parse(graffiti_source(), markup, source_url="wiki")


def parse_article(markup: str = ARTICLE_MARKUP) -> list[LoreChunkCreate]:
    """Run the article parser on ``markup`` as if it came from the wiki."""
    return ArticleParser(WikitextCleaner()).parse(article_source(), markup, source_url="wiki")


def parse_quotes(markup: str = QUOTES_MARKUP) -> list[LoreChunkCreate]:
    """Run the quotes parser on ``markup`` as if it came from the wiki."""
    return QuotesParser(WikitextCleaner()).parse(quotes_source(), markup, source_url="wiki")


class TestWikitextCleaner:
    def test_strips_templates_galleries_and_footnotes(self) -> None:
        text = WikitextCleaner().clean("{{pix}}<gallery>\nA.png\n</gallery>Fact<ref>note</ref>.")

        assert "pix" not in text
        assert "A.png" not in text
        assert "note" not in text
        assert text == "Fact."

    def test_keeps_link_labels_and_drops_media_links(self) -> None:
        markup = "See [[No Mercy|the campaign]] and [[Tank]]. [[File:X.png|thumb|A Tank]]"

        text = WikitextCleaner().clean(markup)

        assert "the campaign" in text
        assert "Tank" in text
        assert "File:" not in text
        assert "X.png" not in text

    def test_an_infobox_keeps_its_data_and_loses_its_layout(self) -> None:
        """The health figures live in the infobox; dropping templates would drop them."""
        text = WikitextCleaner().clean(INFOBOX_MARKUP)

        assert "Health — 3000 (Easy)" in text
        assert "4000 (Normal)" in text
        assert "Abilities — Immense strength" in text
        assert "Infobox" not in text
        assert "Tank.png" not in text

    def test_layout_templates_are_still_dropped(self) -> None:
        assert "pix" not in WikitextCleaner().clean(INFOBOX_MARKUP)

    def test_a_piped_link_inside_an_infobox_does_not_eat_the_next_parameters(self) -> None:
        """Splitting parameters on every pipe tears a link apart and loses the rest."""
        markup = (
            "{{Infobox infected\n"
            "|Ability icon = [[File:icon.png|50px]]\n"
            "|Health = 4000\n"
            "|Speed = Fast\n"
            "}}"
        )

        text = WikitextCleaner().clean(markup)

        assert "Health — 4000" in text
        assert "Speed — Fast" in text

    def test_a_parameter_holding_only_an_image_is_dropped(self) -> None:
        markup = "{{Infobox infected\n|Ability icon = [[File:a.png]]\n|Health = 1\n}}"

        text = WikitextCleaner().clean(markup)

        assert "Ability icon" not in text
        assert "Health — 1" in text

    def test_an_infobox_without_parameters_leaves_nothing(self) -> None:
        assert WikitextCleaner().clean("{{Infobox infected}}Body") == "Body"

    def test_flattens_tables_instead_of_dropping_them(self) -> None:
        text = WikitextCleaner().clean(ARTICLE_MARKUP)

        assert "Health — 4000" in text
        assert "Name — Value" in text

    def test_unwraps_emphasis_underline_and_entities(self) -> None:
        text = WikitextCleaner().clean("*<u>ENTIRE</u> CITY IS GONE &amp; '''RUN'''")

        assert "ENTIRE CITY IS GONE & RUN" in text

    def test_sections_expose_their_nesting_path(self) -> None:
        sections = WikitextCleaner().sections(GRAFFITI_MARKUP)

        assert [section.path for section in sections] == [
            ("Left 4 Dead",),
            ("Left 4 Dead", "No Mercy"),
            ("Left 4 Dead", "No Mercy", "The Subway"),
            ("Left 4 Dead", "No Mercy", "Rooftop Finale"),
            ("Left 4 Dead", "Dead Air"),
            ("Left 4 Dead", "Dead Air", "The Alleys"),
        ]

    def test_a_campaign_heading_carries_no_inscriptions_of_its_own(self) -> None:
        sections = WikitextCleaner().sections(GRAFFITI_MARKUP)
        campaign = next(
            section for section in sections if section.path == ("Left 4 Dead", "No Mercy")
        )

        assert campaign.body.strip() == ""

    def test_lead_is_the_text_before_the_first_heading(self) -> None:
        lead = WikitextCleaner().lead(ARTICLE_MARKUP)

        assert "Green Flu" in lead
        assert "Origin" not in lead

    def test_list_items_report_their_depth(self) -> None:
        items = WikitextCleaner().list_items("*root\n:*reply\n*second")

        assert [(item.depth, item.text) for item in items] == [
            (1, "root"),
            (2, "reply"),
            (1, "second"),
        ]

    def test_blank_list_markers_are_ignored(self) -> None:
        items = WikitextCleaner().list_items("*\n*  \n*real")

        assert [item.text for item in items] == ["real"]


class TestGraffitiParser:
    def test_keeps_replies_together_with_the_inscription_they_answer(self) -> None:
        records = parse_graffiti()

        change = next(record for record in records if "4 days to change" in record.content)
        assert change.content == "It takes at least 4 days to change\n↳ No it doesn't"

    def test_titles_name_the_campaign_and_chapter_without_the_game(self) -> None:
        records = parse_graffiti()

        assert {record.title for record in records} == {
            "No Mercy › The Subway",
            "No Mercy › Rooftop Finale",
            "Dead Air › The Alleys",
        }
        assert all(record.source_type is SourceType.GRAFFITI for record in records)
        assert all(record.topic is LoreTopic.GRAFFITI for record in records)

    def test_keeps_the_note_about_where_the_writings_are(self) -> None:
        alley = [record for record in parse_graffiti() if record.title == "Dead Air › The Alleys"]

        assert any("normal room" in record.content for record in alley)
        assert any("HOWITZER" in record.content for record in alley)

    def test_every_record_carries_its_provenance(self) -> None:
        records = parse_graffiti()

        assert all(record.source_url == "wiki" for record in records)

    def test_a_long_conversation_is_split_on_line_boundaries(self) -> None:
        markup = "== C ==\n=== [[No Mercy]] ===\n==== [[The Subway]] ====\n" + "\n".join(
            f"*Inscription number {index} written by somebody on this wall" for index in range(20)
        )

        records = parse_graffiti(markup)

        assert len(records) > 1
        assert all(len(record.content) <= 800 for record in records)
        assert all(record.content.startswith("Inscription") for record in records)


class TestArticleParser:
    def test_lead_becomes_an_overview_record(self) -> None:
        records = parse_article()
        overview = next(record for record in records if record.title == "Green Flu › Overview")

        assert "highly contagious" in overview.content
        assert overview.source_type is SourceType.OFFICIAL
        assert overview.topic is LoreTopic.INFECTION

    def test_section_statistics_survive_the_cleaner(self) -> None:
        statistics = next(
            record for record in parse_article() if record.title == "Green Flu › Statistics"
        )

        assert "Health — 4000" in statistics.content

    def test_records_stay_within_the_chunk_budget(self) -> None:
        markup = "== Big ==\n" + "\n\n".join(
            f"Paragraph {index} " + "word " * 60 for index in range(20)
        )

        records = parse_article(markup)

        assert len(records) > 1
        assert all(len(record.content) <= 900 for record in records)

    def test_short_sections_are_dropped(self) -> None:
        assert parse_article("== Tiny ==\nToo short.") == []


class TestQuotesParser:
    def test_filters_one_word_reactions_and_keeps_wordy_lines(self) -> None:
        intro = next(record for record in parse_quotes() if record.title == "Эллис · Intro")

        assert "Kill all sons of bitches" in intro.content
        assert "Ugh!" not in intro.content
        assert "Watch out!" not in intro.content

    def test_groups_lines_by_the_situation_they_were_recorded_for(self) -> None:
        assert {record.title for record in parse_quotes()} == {
            "Эллис · Intro",
            "Эллис · General",
        }

    def test_records_are_tagged_as_the_character_voice(self) -> None:
        records = parse_quotes()

        assert all(record.source_type is SourceType.VOICE for record in records)
        assert all(record.character_name == "Эллис" for record in records)
        assert all(record.topic is LoreTopic.QUOTES for record in records)

    def test_wiki_bookkeeping_is_not_part_of_a_line(self) -> None:
        intro = next(record for record in parse_quotes() if record.title == "Эллис · Intro")

        assert "I like to keep my head down and my gun up." in intro.content
        assert "versions" not in intro.content

    def test_stage_directions_are_not_dialogue(self) -> None:
        intro = next(record for record in parse_quotes() if record.title == "Эллис · Intro")

        assert "When a Tank appears" not in intro.content

    def test_a_situation_with_a_single_line_is_dropped(self) -> None:
        markup = '== Only ==\n*"A single line that says nothing about the speaker."'

        assert parse_quotes(markup) == []

    def test_quotes_page_without_a_character_is_ignored(self) -> None:
        source = WikiSource("Ellis/Quotes", LoreTopic.QUOTES, ParserKind.QUOTES)

        assert QuotesParser(WikitextCleaner()).parse(source, QUOTES_MARKUP, source_url="wiki") == []
