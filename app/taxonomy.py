"""Domain vocabulary of the lore archive.

The taxonomy is deliberately kept out of the ORM, the schemas and the ingestion
package: every layer needs to name the same concepts (what kind of evidence a
chunk is, which wiki section it came from), so the vocabulary lives on its own
and the layers depend on it instead of on each other.

Two axes describe a chunk:

* :class:`SourceType` — *how trustworthy* the record is, which is what the
  retrieval pipeline branches on. Official lore is verified encyclopaedia
  material, graffiti is hearsay scrawled on a safe-room wall, and voice samples
  are the recorded lines of a character.
* :class:`LoreTopic` — *what* the record is about, used for reporting and for
  narrowing a search when a caller wants a single subject.
"""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Evidence tier of a lore record.

    The values are stored verbatim in the database, so the members double as the
    values a SQL ``CHECK`` constraint and the API contract rely on.
    """

    OFFICIAL = "official"
    """Encyclopaedia lore from the Fandom wiki articles."""

    GRAFFITI = "graffiti"
    """Unverified messages left by survivors on safe-room walls."""

    VOICE = "voice"
    """Recorded dialogue of one character, used to imitate their manner of speech."""


class LoreTopic(StrEnum):
    """Subject area of a lore record, mirroring the wiki's own sections."""

    INFECTION = "infection"
    """The Green Flu: origin theories, CEDA, carriers, the military response."""

    SURVIVORS = "survivors"
    """Survivor biographies and facts."""

    INFECTED = "infected"
    """Common, uncommon and special infected, their mutations and behaviour."""

    CAMPAIGNS = "campaigns"
    """Campaign and chapter walkthroughs."""

    SAFE_ROOMS = "safe_rooms"
    """Safe-room mechanics and the terminal itself."""

    GRAFFITI = "graffiti"
    """The wall inscriptions of the safe rooms."""

    QUOTES = "quotes"
    """Dialogue lines of the survivors."""

    WEAPONS = "weapons"
    """The arsenal: firearms, melee weapons, ammunition and upgrades."""

    ITEMS = "items"
    """Pickups and deployables: healing, throwables, gas cans, props."""

    LOCATIONS = "locations"
    """Named places of the outbreak, from hospitals to fairgrounds."""

    GAME_MODES = "game_modes"
    """Game modes, difficulty levels and the rules that shape a run."""

    ACHIEVEMENTS = "achievements"
    """The achievements and the stunts they reward."""

    NPCS = "npcs"
    """Non-playable characters: Virgil, Whitaker, the military."""

    TACTICS = "tactics"
    """Advice and strategies the survivors wrote down for each other."""
