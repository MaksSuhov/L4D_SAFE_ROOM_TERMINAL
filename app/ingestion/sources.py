"""Declarative registry of the wiki pages that make up the corpus.

Every page the pipeline reads is declared here once, together with the topic it
belongs to and the parser that knows its layout. Adding a page to the corpus is
therefore a one-line change, and the set of sources is reviewable at a glance
instead of being scattered across the code.

The corpus is read from the **Russian** Left 4 Dead Wiki (``/ru`` branch of the
Fandom wiki), because the terminal answers in Russian: keeping the archive and
the questions in one language makes retrieval monolingual, which is worth far
more than the extra detail the English articles carry. The registry therefore
holds Russian titles, and the traps noted below are Russian-wiki specific.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.taxonomy import LoreTopic, SourceType


class ParserKind(StrEnum):
    """Layout families the wiki uses, each handled by its own parser."""

    GRAFFITI = "graffiti"
    """Campaign › chapter › list of wall inscriptions."""

    ARTICLE = "article"
    """Encyclopaedia prose with stat tables, split by heading."""

    QUOTES = "quotes"
    """Dialogue lists grouped by situation."""

    SUBTITLES = "subtitles"
    """A Russian/English subtitle table, one character per page."""

    @property
    def source_type(self) -> SourceType:
        """Evidence tier produced by this layout.

        Deriving the tier from the layout keeps the two in step: a quotes or
        subtitle page can only ever produce voice samples, and an article page
        can only ever produce official lore.
        """
        return {
            ParserKind.GRAFFITI: SourceType.GRAFFITI,
            ParserKind.ARTICLE: SourceType.OFFICIAL,
            ParserKind.QUOTES: SourceType.VOICE,
            ParserKind.SUBTITLES: SourceType.VOICE,
        }[self]


@dataclass(frozen=True, slots=True)
class WikiSource:
    """One wiki page of the corpus."""

    page: str
    topic: LoreTopic
    parser: ParserKind
    character_name: str | None = None
    """Character the page belongs to (their quotes) or is about (their biography)."""


BILL = "Билл"
ZOEY = "Зои"
FRANCIS = "Фрэнсис"
LOUIS = "Луис"
COACH = "Тренер"
NICK = "Ник"
ROCHELLE = "Рошель"
ELLIS = "Эллис"

CHARACTER_PAGES: tuple[tuple[str, str], ...] = (
    ("Билл", BILL),
    ("Зои", ZOEY),
    ("Фрэнсис", FRANCIS),
    ("Луис", LOUIS),
    ("Тренер", COACH),
    ("Ник", NICK),
    ("Рошель", ROCHELLE),
    ("Эллис", ELLIS),
)
"""Wiki article title and the display name stored in the database.

``Зой``/``Френсис`` also exist on the Russian wiki as orphaned duplicates; only
the canonical spelling of each survivor is declared here.
"""

QUOTE_PAGES: tuple[tuple[str, str], ...] = (
    ("Субтитры (Left 4 Dead)/Билл", BILL),
    ("Субтитры (Left 4 Dead)/Зои", ZOEY),
    ("Субтитры (Left 4 Dead)/Фрэнсис", FRANCIS),
    ("Субтитры (Left 4 Dead)/Луис", LOUIS),
    ("Субтитры (Left 4 Dead 2)/Тренер", COACH),
    ("Субтитры (Left 4 Dead 2)/Ник", NICK),
    ("Субтитры (Left 4 Dead 2)/Рошель", ROCHELLE),
    ("Субтитры (Left 4 Dead 2)/Эллис", ELLIS),
)
"""Where the recorded lines of each survivor live.

The Russian wiki stores the dialogue as subtitle tables split by game rather
than as curated quote pages, so each survivor is declared on the page of the
game they debut in. ``Субтитры (The Passing)/…`` holds a second, shorter dump
per survivor if the voice tier ever needs widening.
"""

INFECTION_PAGES: tuple[str, ...] = (
    "Зелёный грипп",
    "Носители",
    "АЧС",
    "Военные",
)

INFECTED_PAGES: tuple[str, ...] = (
    "Заражённые",
    "Обычные заражённые",
    "Необычные заражённые",
    "Особые заражённые",
    "Толстяк",
    "Охотник",
    "Курильщик",
    "Громила",
    "Плевальщица",
    "Жокей",
    "Танк",
    "Ведьма",
    "Крикун",
)
"""The infected pages that carry the facts about the mutations.

The Russian wiki keeps one article per special infected instead of the English
wiki's shared page, which is a straight upgrade for retrieval. Two traps, both
found by reading the wiki rather than the code: ``Необычные Заражённые`` is
capitalised as a redirect of the lowercase title, and the uncommon infected
(``Заражённый клоун``, ``Строитель``, ``Грязевик``…) are section redirects of
``Необычные заражённые`` — declaring them would re-fetch the whole parent page.
``Крикун`` is the cut Screamer, kept because it is real cut-content lore.
"""

CAMPAIGN_PAGES: tuple[str, ...] = (
    "Кампании",
    "Нет милосердию",
    "Роковой полёт",
    "Похоронный звон",
    "Смерть в воздухе",
    "Кровавая жатва",
    "Жертва",
    "Последний рубеж (Left 4 Dead)",
    "Последний рубеж (Left 4 Dead 2)",
    "Вымерший центр",
    "Переход",
    "Мрачный карнавал",
    "Болотная лихорадка",
    "Ужасный ливень",
    "Приход",
    "Холодный ручей",
    "Комикс «Жертва»",
)

WEAPON_PAGES: tuple[str, ...] = (
    "Оружие",
    "Боеприпасы",
    "Комплекты улучшений",
    "Оружие Counter-Strike",
    "Автомат Калашникова",
    "Дробовик",
    "Помповый дробовик",
    "Автоматический дробовик",
    "Боевой дробовик",
    "Хромированный дробовик",
    "Снайперская винтовка",
    "Охотничья винтовка",
    "Военная снайперская винтовка",
    "Пистолет-пулемёт",
    "Пистолет-пулемёт с глушителем",
    "Штурмовая винтовка",
    "Боевая винтовка",
    "Тяжёлый пулемёт",
    "Миниган",
    "Гранатомёт",
    "Пистолет «Магнум»",
    "Бензопила",
    "Катана",
    "Мачете",
    "Гитара",
    "Бейсбольная бита",
    "Фомка",
    "Пожарный топор",
    "Лопата",
    "Вилы",
    "Нож",
    "Полицейская дубинка",
    "Сковородка",
    "Крикетная бита",
    "Клюшка для гольфа",
)
"""The arsenal, missing from the English corpus entirely.

Only canonical titles are declared: the wiki has a redirect for almost every
weapon (``АК-47`` → ``Автомат Калашникова``, ``Монтировка`` → ``Фомка``,
``Сковорода`` → ``Сковородка``), and declaring a redirect both repeats the fetch
and files the text under a name the wiki does not use. ``Оружие``,
``Боеприпасы`` and ``Комплекты улучшений`` are the umbrella pages; the redirects
that point at them (``Основное оружие``, ``Фонарик``, ``Зажигательные
боеприпасы``…) are deliberately absent for the same reason.
"""

ITEM_PAGES: tuple[str, ...] = (
    "Предметы для лечения",
    "Аптечка",
    "Обезболивающие таблетки",
    "Шприц с адреналином",
    "Дефибриллятор",
    "Рвотная бомба",
    "Самодельная бомба",
    "Коктейль Молотова",
    "Канистра с бензином",
    "Пропановый баллон",
    "Кислородный баллон",
    "Взрывоопасная бочка",
    "Фейерверки",
    "Кола",
    "Гном Чомски",
    "Лазерный целеуказатель",
)

LOCATION_PAGES: tuple[str, ...] = (
    "Локации",
    "Госпиталь «Милосердие»",
    "Торговый центр",
    "Риверсайд",
    "Фэрфилд",
    "Милхейвен",
    "Метро",
    "Сахарный завод",
    "Отель «Харбор-Вью»",
    "Станция метро Холли-стрит",
    "Международный аэропорт Метро",
    "Округ Уитни",
)
"""The places of the outbreak.

The wiki holds some fifty more location stubs; the umbrella ``Локации`` page
summarises them, and the ones declared here are the stops a survivor would
actually name.
"""

GAME_MODE_PAGES: tuple[str, ...] = (
    "Игровые режимы",
    "Классификации игровых режимов (Left 4 Dead 2)",
    "Уровни сложности",
    "Финал",
    "Здоровье",
    "Режиссёр",
    "Дружественный огонь",
)
"""The rules that shape a run.

``Выживание``, ``Реализм``, ``Сражение`` and the rest of the named modes are
redirects into ``Игровые режимы``, which describes them all — so only the page
that holds the text is declared.
"""

ACHIEVEMENT_PAGES: tuple[str, ...] = ("Достижения",)

NPC_PAGES: tuple[str, ...] = ("Неигровые персонажи",)
"""Non-playable characters.

``Вирджил`` and ``Уитакер`` are redirects into this page.
"""

TACTIC_PAGES: tuple[str, ...] = (
    "Стратегии выживших",
    "Список официальных советов",
)

GRAFFITI_PAGE = "Граффити"
"""Holds the inscriptions of every safe room, grouped by campaign and chapter."""

SAFE_ROOM_PAGES: tuple[str, ...] = ("Убежище",)


def _sources() -> tuple[WikiSource, ...]:
    """Assemble the registry, keeping the declaration order as the fetch order."""
    sources: list[WikiSource] = [
        WikiSource(page, LoreTopic.INFECTION, ParserKind.ARTICLE) for page in INFECTION_PAGES
    ]
    sources.append(WikiSource("Выжившие", LoreTopic.SURVIVORS, ParserKind.ARTICLE))
    sources.extend(
        WikiSource(page, LoreTopic.SURVIVORS, ParserKind.ARTICLE, character_name=character)
        for page, character in CHARACTER_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.INFECTED, ParserKind.ARTICLE) for page in INFECTED_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.CAMPAIGNS, ParserKind.ARTICLE) for page in CAMPAIGN_PAGES
    )
    sources.extend(WikiSource(page, LoreTopic.WEAPONS, ParserKind.ARTICLE) for page in WEAPON_PAGES)
    sources.extend(WikiSource(page, LoreTopic.ITEMS, ParserKind.ARTICLE) for page in ITEM_PAGES)
    sources.extend(
        WikiSource(page, LoreTopic.LOCATIONS, ParserKind.ARTICLE) for page in LOCATION_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.GAME_MODES, ParserKind.ARTICLE) for page in GAME_MODE_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.ACHIEVEMENTS, ParserKind.ARTICLE)
        for page in ACHIEVEMENT_PAGES
    )
    sources.extend(WikiSource(page, LoreTopic.NPCS, ParserKind.ARTICLE) for page in NPC_PAGES)
    sources.extend(WikiSource(page, LoreTopic.TACTICS, ParserKind.ARTICLE) for page in TACTIC_PAGES)
    sources.extend(
        WikiSource(page, LoreTopic.SAFE_ROOMS, ParserKind.ARTICLE) for page in SAFE_ROOM_PAGES
    )
    sources.append(WikiSource(GRAFFITI_PAGE, LoreTopic.GRAFFITI, ParserKind.GRAFFITI))
    sources.extend(
        WikiSource(page, LoreTopic.QUOTES, ParserKind.SUBTITLES, character_name=character)
        for page, character in QUOTE_PAGES
    )
    return tuple(sources)


WIKI_SOURCES: Sequence[WikiSource] = _sources()
"""Every page the corpus is built from, in fetch order."""
