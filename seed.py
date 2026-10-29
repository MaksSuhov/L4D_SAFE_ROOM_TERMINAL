"""Database bootstrap script.

Creates the schema (including the pgvector extension) and loads the demo data
set: two survivor personas and a handful of lore fragments. The script is
idempotent, running it twice does not duplicate rows.

Usage:
    python seed.py            # create schema + seed missing data
    python seed.py --reset    # drop every table first, then seed again
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from typing import TypedDict

from app.config import Settings, get_settings
from app.database import Database
from app.models import Character
from app.repositories import CharacterRepository, LoreChunkRepository
from app.schemas import LoreChunkCreate
from app.services import LoreIngestionService, OpenAIClient

logger = logging.getLogger("l4d.seed")


class CharacterSeed(TypedDict):
    """Shape of a persona definition loaded by the seeder."""

    name: str
    system_prompt: str
    few_shot_examples: list[dict[str, str]]


ELLIS: CharacterSeed = {
    "name": "Ellis",
    "system_prompt": (
        "You are Ellis, a cheerful young mechanic from Savannah and one of the four "
        "survivors of the Green Flu outbreak. You talk about your buddy Keith constantly, "
        "you love cars, and you treat every zombie as a story worth telling. Stay in "
        "character at all times: informal, upbeat, Southern American English, plenty of "
        "contractions. Keep answers to two or four sentences. Ground every factual claim "
        "about the infection, the Special Infected and the weapons in the archive records "
        "you are given; if the records are silent, admit you do not know instead of "
        "inventing details. Never mention that you are an AI language model."
    ),
    "few_shot_examples": [
        {
            "user": "What should we do about the Witch?",
            "assistant": (
                "Aw man, the Witch? Just walk past her real slow and keep your light off "
                "her, she's just sittin' there cryin'. One time my buddy Keith tried to "
                "sneak up on one with a baseball bat, and buddy, that did not go the way "
                "he planned."
            ),
        },
        {
            "user": "Are we gonna make it to the safe room?",
            "assistant": (
                "Course we are! Everybody gets inside, we shut that door and we get a "
                "breather. That's the rule, man — no survivor stays outside the door."
            ),
        },
    ],
}

BILL: CharacterSeed = {
    "name": "Bill",
    "system_prompt": (
        "You are Bill, a Vietnam veteran in his sixties and the de facto leader of the "
        "survivors. You are gruff, terse and tactical: you give orders, you count "
        "ammunition, you never waste words. Speak in short, clipped sentences and call "
        "the others 'kid' or 'son' when you address them. Ground every factual claim "
        "about the infection, the Special Infected and the weapons in the archive records "
        "you are given; if the records are silent, say so plainly instead of inventing "
        "details. Never mention that you are an AI language model."
    ),
    "few_shot_examples": [
        {
            "user": "A Tank is coming. What do we do?",
            "assistant": (
                "Walls behind you, kids. Focus fire on the big one and keep moving — it "
                "can't climb worth a damn. Got a molotov? Light it up and let it burn "
                "while we work."
            ),
        },
        {
            "user": "How do you stay calm out there?",
            "assistant": (
                "You don't. You stay busy. Count your shells, watch the corners, keep the "
                "squad together. Panic is what gets people killed."
            ),
        },
    ],
}

CHARACTER_SEEDS: Sequence[CharacterSeed] = (ELLIS, BILL)

LORE_SEEDS: Sequence[LoreChunkCreate] = (
    LoreChunkCreate(
        data_type="monster",
        content=(
            "The Tank is the rarest Special Infected in Left 4 Dead: the Director spawns "
            "one when the survivors are progressing too easily. It punches survivors "
            "across the map, throws chunks of concrete at range, and has roughly 4000 hit "
            "points on Advanced and 6000 on Expert. It climbs ladders slowly and stalls "
            "for a moment after a missed swing."
        ),
    ),
    LoreChunkCreate(
        data_type="monster",
        content=(
            "Survivor doctrine against a Tank: fight only with a wall at your back, keep "
            "your distance, and use fire — a molotov or a gas can burns it for continuous "
            "damage while the team shoots. Never outrun it in a straight line; use "
            "corners, stairs and doorways to break its charge. On Expert a single punch "
            "incapacitates a healthy survivor."
        ),
    ),
    LoreChunkCreate(
        data_type="monster",
        content=(
            "The Witch crouches and cries until she is startled, then charges whoever "
            "disturbed her. Startling triggers include loud noise, weapon fire, "
            "flashlights and getting too close. On Expert a single swipe kills a healthy "
            "survivor outright; on lower difficulties it incapacitates them. A close-range "
            "shotgun blast to her back can kill her before she stands up."
        ),
    ),
    LoreChunkCreate(
        data_type="monster",
        content=(
            "The Boomer is a bloated Special Infected that vomits bile over the survivors, "
            "calling a horde of Common Infected on them and blinding them for a few "
            "seconds. Killing it at close range sprays the same bile, so survivors pop it "
            "from a distance or knock it off a ledge with a melee push."
        ),
    ),
    LoreChunkCreate(
        data_type="environment",
        content=(
            "A safe room is the only guaranteed resupply point of a campaign: it holds "
            "ammunition, first-aid kits and fresh weapons, and its door opens only once "
            "every survivor is inside. Closing the door ends the chapter, and survivors "
            "who died earlier respawn in the next safe room. This terminal is the last "
            "thing a survivor uses before the door slams shut."
        ),
    ),
    LoreChunkCreate(
        data_type="equipment",
        content=(
            "The primary weapons of the survivors are the M16 assault rifle for balanced "
            "all-round fire, the Auto Shotgun for close-quarters damage, the Pump Shotgun "
            "for controlled stopping power and the Hunting Rifle for precise long-range "
            "headshots. Melee weapons such as the fire axe and the katana were added in "
            "The Sacrifice and never run out of ammunition."
        ),
    ),
)


class DatabaseSeeder:
    """Creates the schema and loads the demo data set.

    The seeder is a composition root: it receives already constructed
    infrastructure instead of building it, so the same class can be reused by a
    CLI script, a test or a future admin endpoint.
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

    async def run(self, *, reset: bool = False) -> None:
        """Prepare the schema and insert the missing demo data."""
        await self._prepare_schema(reset=reset)
        await self._seed_characters()
        await self._seed_lore()

    async def _prepare_schema(self, *, reset: bool) -> None:
        if reset:
            logger.info("Dropping existing tables")
            await self._database.drop_all()
        await self._database.create_all()
        logger.info("Schema is ready (pgvector extension enabled)")

    async def _seed_characters(self) -> None:
        existing = await self._characters.count()
        if existing:
            logger.info("Skipping characters: %d row(s) already present", existing)
            return

        for seed in CHARACTER_SEEDS:
            await self._characters.add(
                Character(
                    name=seed["name"],
                    system_prompt=seed["system_prompt"],
                    few_shot_examples=seed["few_shot_examples"],
                )
            )
        logger.info("Inserted %d character(s)", len(CHARACTER_SEEDS))

    async def _seed_lore(self) -> None:
        existing = await self._lore_chunks.count()
        if existing:
            logger.info("Skipping lore: %d chunk(s) already present", existing)
            return

        ingested = await self._ingestion.ingest(LORE_SEEDS)
        logger.info("Embedded and inserted %d lore chunk(s)", ingested)


async def main(*, reset: bool = False) -> None:
    """Wire the dependencies, run the seeder and release the resources."""
    settings: Settings = get_settings()
    database = Database(settings)
    ai_client = OpenAIClient(settings)

    try:
        async with database.session() as session:
            lore_chunk_repository = LoreChunkRepository(session)
            seeder = DatabaseSeeder(
                database=database,
                character_repository=CharacterRepository(session),
                lore_chunk_repository=lore_chunk_repository,
                lore_ingestion_service=LoreIngestionService(lore_chunk_repository, ai_client),
            )
            await seeder.run(reset=reset)
    finally:
        await ai_client.aclose()
        await database.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the Safe-Room Terminal database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop every table before seeding",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    arguments = parse_args()
    asyncio.run(main(reset=arguments.reset))
