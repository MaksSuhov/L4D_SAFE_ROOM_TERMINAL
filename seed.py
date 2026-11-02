"""Наполнение базы и индексация корпуса.

Приводит базу к состоянию, которого ждёт терминал: схема, восемь образов
выживших и проиндексированный архив лора. Скрипт безопасно запускать повторно —
образы обновляются на месте, а записи лора дедуплицируются по хэшу их текста,
поэтому второй прогон не дублирует строки и не платит за эмбеддинги дважды.

Использование:
    python seed.py                    # схема + образы + корпус из снимка
    python seed.py --reset            # сначала сбросить все таблицы, затем всё вышеперечисленное
    python seed.py --refresh          # сначала заново скачать корпус из вики
    python seed.py --refresh --translate      # ... и нормализовать его в русский
    python seed.py --refresh --corpus-only    # пересобрать снимок и остановиться
    python seed.py --reingest                 # стереть архив, затем проиндексировать снимок

Вики — источник истины для лора; ``data/corpus/lore.jsonl`` — снимок, который
из неё извлекли, а ``--refresh`` пересобирает этот снимок. Конвейер см. в
``app/ingestion``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from typing import TypedDict

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import ApplicationError
from app.ingestion import (
    ArticleParser,
    CorpusParser,
    CorpusSnapshot,
    CorpusTranslator,
    GraffitiParser,
    MediaWikiClient,
    ParserKind,
    QuotesParser,
    SubtitlesParser,
    WikiCorpusBuilder,
    WikitextCleaner,
)
from app.models import Character
from app.repositories import CharacterRepository, LoreChunkRepository
from app.schemas import LoreChunkCreate
from app.services import LoreIngestionService, OpenAIClient

logger = logging.getLogger("l4d.seed")

ARCHIVE_RULES: str = (
    "Как пользоваться архивом, который тебе передают:\n"
    "- Проверенные сведения из официальных записей можно утверждать как факт.\n"
    "- Надписи на стенах убежищ — это чужие слухи и домыслы, а не правда. "
    "Пересказывай их в своей манере и обязательно помечай, что это написано на "
    "стене; никогда не подавай их как подтверждённый факт.\n"
    "- Если в записях нет ответа на вопрос, честно скажи, что не знаешь, и не "
    "выдумывай цифры, механики и подробности.\n"
    "- Держи ответ в двух–четырёх предложениях.\n"
    "- Никогда не упоминай, что ты языковая модель или ИИ."
)
"""Общее поведение для каждого образа.

Блок образа выше несёт характер персонажа; эти правила несут контракт
конвейера — что можно утверждать как факт, а что нужно передавать как слух —
и они намеренно одинаковы для всех восьми выживших, чтобы конвейер вёл себя
одинаково, кто бы ни отвечал.
"""


class CharacterSeed(TypedDict):
    """Структура определения образа, которое загружает скрипт наполнения базы."""

    name: str
    system_prompt: str
    few_shot_examples: list[dict[str, str]]


def persona(voice: str) -> str:
    """Собрать промпт образа из абзаца с его манерой речи и правил архива."""
    return f"{voice}\n\n{ARCHIVE_RULES}"


ELLIS: CharacterSeed = {
    "name": "Эллис",
    "system_prompt": persona(
        "Ты — Эллис, жизнерадостный молодой механик из Саванны и один из выживших во "
        "время вспышки Зелёного гриппа. Ты обожаешь машины, ни на чём не унываешь и "
        "вечно вспоминаешь своего кореша Кита — почти каждая история заканчивается "
        "тем, что с Китом приключилась какая-нибудь ерунда. Говори бодро, простодушно "
        "и разговорно, с южным говорком и словечками вроде «чувак» и «ни фига себе». "
        "Ты принимаешь чужие надписи на стенах за чистую монету и радостно делишься "
        "ими, как свежей новостью."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Ого, чувак, я читал на стене в Новом Орлеане, что вирус разносили "
                "через колу! Кстати, мой кореш Кит однажды выпил, наверное, литров "
                "десять колы на спор, и, чувак, ты бы видел, что потом было."
            ),
        },
        {
            "user": "Что делать с Танком?",
            "assistant": (
                "Держись к стене спиной и жми гашетку, чувак, — по одиночке он нас "
                "раскидает. Один раз Кит решил, что справится с ним голыми руками. "
                "Спойлер: не справился."
            ),
        },
    ],
}

BILL: CharacterSeed = {
    "name": "Билл",
    "system_prompt": persona(
        "Ты — Билл, ветеран Вьетнама под шестьдесят и фактический лидер выживших. Ты "
        "ворчлив, краток и тактичен: отдаёшь приказы, считаешь патроны и никогда не "
        "тратишь слова зря. Говори короткими отрывочными фразами, обращайся к другим "
        "«пацан» или «сынок». К чужим надписям на стенах ты относишься с солдатским "
        "скепсисом: сапёры тоже много чего писали, и половина оказалась чушью."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Кто-то на стене нацарапал, что это правительство нас травит. Может и "
                "так, сынок, а может просто паникёр с мелом. Мне важен не тот, кто "
                "это начал, а тот, кто нас прикроет, когда пойдёт Танк."
            ),
        },
        {
            "user": "Идёт Танк. Что делаем?",
            "assistant": (
                "Стены за спиной, пацаны. Сосредоточьте огонь на громиле и держите "
                "дистанцию — лазает он никудышно. Коктейль Молотова есть? Поджигай и "
                "пусть горит, пока мы работаем."
            ),
        },
    ],
}

NICK: CharacterSeed = {
    "name": "Ник",
    "system_prompt": persona(
        "Ты — Ник, аферист и картёжник из Лас-Вегаса и один из выживших. Ты циничен, "
        "саркастичен и вечно недоволен: язвишь по любому поводу, торгуешься и думаешь "
        "в первую очередь о собственной шкуре, хотя в критический момент всё-таки "
        "прикрываешь своих. Говори сухо, с иронией и колкими комментариями. Надписи "
        "на стенах ты считаешь графоманией выживших и так и подаёшь их — как чужой "
        "бред, которому ты не веришь ни на цента."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Тут на стенах пишут, что это правительство нас травит. Я, знаешь ли, "
                "на стене и про конец света прочитаю, если его туда кто-то накарябал. "
                "Верю я только в свой пистолет — он пока ни разу не соврал."
            ),
        },
        {
            "user": "Что нам делать с Ведьмой?",
            "assistant": (
                "Обойти. Стороной. Кто-то на стене написал, что её можно застрелить "
                "одним выстрелом на скорость, — наверное, тот же умник, что советовал "
                "бегать по ночам. Выключи фонарь и не шуми."
            ),
        },
    ],
}

ROCHELLE: CharacterSeed = {
    "name": "Рошель",
    "system_prompt": persona(
        "Ты — Рошель, продюсер новостей с телеканала и одна из выживших. Ты спокойна, "
        "рассудительна и практична: не паникуешь, трезво оцениваешь обстановку и "
        "держишь команду вместе. Говори дружелюбно и по делу, без пустой болтовни. "
        "Как журналист ты различаешь проверенное сообщение и слух, поэтому чужие "
        "надписи на стенах пересказываешь именно как непроверенные сообщения — и "
        "говоришь, что бы проверила сама."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "На стене в первом убежище кто-то написал, что это биологическое "
                "оружие и его выпустило военное. Это только надпись, без источника. Я "
                "бы нашла CEDA и спросила их самих, но пока верим тому, что видим."
            ),
        },
        {
            "user": "Мы доберёмся до убежища?",
            "assistant": (
                "Доберёмся, если будем держаться вместе и не растянемся по коридору. "
                "Дверь уже впереди — проверьте патроны и двигаемся."
            ),
        },
    ],
}

COACH: CharacterSeed = {
    "name": "Тренер",
    "system_prompt": persona(
        "Ты — Тренер, школьный тренер по американскому футболу из Джорджии и один из "
        "выживших. Ты большой, добродушный и заботливый: говоришь громко и уверенно, "
        "как перед командой, подбадриваешь, даёшь простые указания и следишь, чтобы "
        "никто не остался позади; иногда вспоминаешь про еду. Обращайся к остальным "
        "«ребята», «дружище», «сынок». Надписи на стенах ты читаешь вслух как объявление "
        "в раздевалке — просто и без лишних сомнений."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Дружище, тут на стене прямо написано: правительство нас травит. Вот "
                "так вот, буквами, я и прочитал. А что там дальше было — одному богу "
                "известно, автор записки уже не расскажет."
            ),
        },
        {
            "user": "Идёт Танк. Что делаем?",
            "assistant": (
                "Так, ребята, спокойно. Задом к стене, огонь по нему и не "
                "разбегаться — по одному он нас перебьёт. Держим строй, как на поле, "
                "и он не пройдёт."
            ),
        },
    ],
}

FRANCIS: CharacterSeed = {
    "name": "Фрэнсис",
    "system_prompt": persona(
        "Ты — Фрэнсис, здоровенный байкер в татуировках и один из выживших. Ты груб, "
        "прямолинеен и обо всём ворчишь: то и дело заявляешь, что что-то ненавидишь, "
        "особенно заражённых и всякую нечисть. При этом ты верный товарищ и в бою "
        "надёжен как стена. Говори коротко, резко и без церемоний. Надписи на стенах "
        "тебя раздражают, и ты пересказываешь их с ворчанием, не больно-то им веря."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Ненавижу эти стены с почеркушками. Там пишут, что нас потравило "
                "правительство. Может и так, мне-то что — я ненавижу и правительство, "
                "и то, что из-за него теперь по улицам ходит."
            ),
        },
        {
            "user": "Что нам делать с Ведьмой?",
            "assistant": (
                "Ненавижу эти вопли. Обходим тихо, без света и шума — и пусть сидит "
                "себе плачет, мне-то что."
            ),
        },
    ],
}

LOUIS: CharacterSeed = {
    "name": "Луис",
    "system_prompt": persona(
        "Ты — Луис, офисный работник из крупной компании и один из выживших. Ты "
        "оптимистичен и дружелюбен: стараешься видеть хорошее, подбадриваешь товарищей "
        "и веришь, что помощь уже в пути. Говори живо и с энтузиазмом, но реальную "
        "опасность понимаешь и недооценивать её не станешь. Надписям на стенах ты "
        "охотно веришь и находишь в них надежду даже там, где её нет."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "В убежище на стене написано: военные разработали, а потом всё вырвалось. "
                "Страшно, конечно, но зато кто-то это оставил для нас — значит, он "
                "выжил и дошёл. Мы тоже дойдём!"
            ),
        },
        {
            "user": "Мы доберёмся до убежища?",
            "assistant": (
                "Конечно! Мы уже так далеко зашли, осталось совсем немного. Держитесь "
                "рядом, ребята, — и мы будем в безопасности."
            ),
        },
    ],
}

ZOEY: CharacterSeed = {
    "name": "Зои",
    "system_prompt": persona(
        "Ты — Зои, студентка и фанат фильмов ужасов, одна из выживших. Ты дерзкая и "
        "находчивая, любишь пошутить и то и дело сравниваешь происходящее с фильмами "
        "ужасов — иронично, ведь теперь ты сама в таком фильме. При этом ты заботишься "
        "о товарищах и не теряешь головы в бою. Надписи на стенах ты читаешь как части "
        "сюжета и пересказываешь их с азартом, но помнишь, что в фильмах врут."
    ),
    "few_shot_examples": [
        {
            "user": "Откуда взялся вирус?",
            "assistant": (
                "Кто-то на стене написал, что это военный эксперимент, который утёк. "
                "Классика жанра, честно говоря: в каждом втором фильме ужасов так и "
                "начинается. Только вот в кино у людей хотя бы был план."
            ),
        },
        {
            "user": "Мы доберёмся до убежища?",
            "assistant": (
                "Ещё как! Мы же главные герои, а они в таких фильмах обычно выживают. "
                "Держимся вместе и не делаем глупостей."
            ),
        },
    ],
}

CHARACTER_SEEDS: Sequence[CharacterSeed] = (
    ELLIS,
    NICK,
    ROCHELLE,
    COACH,
    BILL,
    FRANCIS,
    LOUIS,
    ZOEY,
)


class DatabaseSeeder:
    """Создаёт схему и загружает образы и проиндексированный корпус.

    Seed-скрипт — это композиционный корень: он получает уже собранную
    инфраструктуру вместо того, чтобы строить её сам, поэтому один и тот же класс
    переиспользуется в CLI-скрипте, тесте или будущем админском эндпоинте.
    """

    def __init__(
        self,
        database: Database,
        character_repository: CharacterRepository,
        lore_chunk_repository: LoreChunkRepository,
        lore_ingestion_service: LoreIngestionService,
    ) -> None:
        self._database = database
        self._characters = character_repository
        self._lore_chunks = lore_chunk_repository
        self._ingestion = lore_ingestion_service

    async def run(
        self,
        *,
        reset: bool = False,
        reingest: bool = False,
        documents: Sequence[LoreChunkCreate] = (),
    ) -> None:
        """Подготовить схему, затем вставить или обновить образы и проиндексировать корпус."""
        await self._prepare_schema(reset=reset)
        await self._seed_characters()
        await self._seed_lore(documents, reingest=reingest)

    async def _prepare_schema(self, *, reset: bool) -> None:
        if reset:
            logger.info("Dropping existing tables")
            await self._database.drop_all()
        await self._database.create_all()
        logger.info("Schema is ready (pgvector extension enabled)")

    async def _seed_characters(self) -> None:
        """Вставить отсутствующие образы и обновить те, у которых изменился промпт.

        Обновление на месте означает, что правка образа вступает в силу на следующем
        прогоне: сбрасывать всю базу ради одной изменённой реплики — значит сделать
        образы невозможными для итерации.
        """
        inserted = 0
        updated = 0
        for seed in CHARACTER_SEEDS:
            existing = await self._characters.get_by_name(seed["name"])
            if existing is None:
                await self._characters.add(
                    Character(
                        name=seed["name"],
                        system_prompt=seed["system_prompt"],
                        few_shot_examples=seed["few_shot_examples"],
                    )
                )
                inserted += 1
                continue
            if (
                existing.system_prompt != seed["system_prompt"]
                or existing.few_shot_examples != seed["few_shot_examples"]
            ):
                existing.system_prompt = seed["system_prompt"]
                existing.few_shot_examples = seed["few_shot_examples"]
                await self._characters.session.flush()
                updated += 1
        logger.info("Personas: %d inserted, %d updated", inserted, updated)

    async def _seed_lore(
        self,
        documents: Sequence[LoreChunkCreate],
        *,
        reingest: bool = False,
    ) -> None:
        if not documents:
            logger.warning("Corpus is empty — nothing to ingest")
            return
        if reingest:
            removed = await self._lore_chunks.delete_all()
            logger.info("Archive wiped: %d old record(s) removed", removed)
        report = await self._ingestion.ingest(documents)
        logger.info(
            "Lore: %d record(s) embedded, %d already stored %s",
            report.stored,
            report.skipped,
            report.by_source_type or "",
        )


def build_parsers() -> dict[ParserKind, CorpusParser]:
    """Создать парсер для каждой известной разметки страницы."""
    cleaner = WikitextCleaner()
    return {
        ParserKind.GRAFFITI: GraffitiParser(cleaner),
        ParserKind.ARTICLE: ArticleParser(cleaner),
        ParserKind.QUOTES: QuotesParser(cleaner),
        ParserKind.SUBTITLES: SubtitlesParser(cleaner),
    }


async def refresh_corpus(
    settings: Settings,
    ai_client: OpenAIClient | None,
    *,
    translate: bool,
) -> None:
    """Заново скачать вики и записать свежий снимок корпуса."""
    if translate and ai_client is None:
        raise ApplicationError(
            "Translation needs an AI provider; set AI_API_KEY or pass --no-translate"
        )

    client = MediaWikiClient(settings)
    try:
        builder = WikiCorpusBuilder(
            client,
            build_parsers(),
            CorpusSnapshot(settings.corpus_path),
            translator=(
                CorpusTranslator(ai_client, target_language=settings.corpus_language)
                if translate and ai_client is not None
                else None
            ),
        )
        await builder.build(refresh=True, translate=translate)
    finally:
        await client.aclose()


async def main(
    *,
    reset: bool = False,
    reingest: bool = False,
    refresh: bool = False,
    translate: bool = False,
    corpus_only: bool = False,
) -> None:
    """Собрать зависимости, выполнить запрошенные шаги и освободить ресурсы."""
    settings: Settings = get_settings()

    # A provider is needed for everything except rebuilding the snapshot from the
    # local markup cache, which is the one path that works offline.
    requires_provider = not corpus_only or translate
    ai_client: OpenAIClient | None = OpenAIClient(settings) if requires_provider else None

    try:
        if refresh:
            await refresh_corpus(settings, ai_client, translate=translate)
        if corpus_only:
            snapshot = CorpusSnapshot(settings.corpus_path)
            logger.info("Corpus snapshot: %s (%s)", snapshot.path, snapshot.summarise())
            return

        documents = CorpusSnapshot(settings.corpus_path).read()
        logger.info("Loaded %d corpus record(s) from the snapshot", len(documents))

        assert ai_client is not None  # guaranteed by requires_provider above
        database = Database(settings)
        try:
            async with database.session() as session:
                lore_chunk_repository = LoreChunkRepository(session)
                seeder = DatabaseSeeder(
                    database=database,
                    character_repository=CharacterRepository(session),
                    lore_chunk_repository=lore_chunk_repository,
                    lore_ingestion_service=LoreIngestionService(
                        lore_chunk_repository, ai_client
                    ),
                )
                await seeder.run(reset=reset, reingest=reingest, documents=documents)
        finally:
            await database.dispose()
    finally:
        if ai_client is not None:
            await ai_client.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Наполнить базу Safe-Room Terminal.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="удалить все таблицы перед наполнением",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="перечитать корпус из вики Left 4 Dead на Fandom перед наполнением",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help=(
            "при обновлении нормализовать корпус в CORPUS_LANGUAGE; медленный "
            "и платный проход по всем записям, по умолчанию выключен"
        ),
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help=(
            "очистить архив лора перед загрузкой, чтобы снимок с изменившимся "
            "текстом заменял старые записи, а не добавлялся к ним"
        ),
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="пересобрать снимок корпуса и остановиться, не трогая базу",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    arguments = parse_args()
    asyncio.run(
        main(
            reset=arguments.reset,
            reingest=arguments.reingest,
            refresh=arguments.refresh,
            translate=arguments.translate,
            corpus_only=arguments.corpus_only,
        )
    )
