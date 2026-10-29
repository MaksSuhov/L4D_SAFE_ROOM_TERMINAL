#!/usr/bin/env python3
"""Derive the artwork that skins the terminal UI from the Left 4 Dead wiki.

The script mirrors the corpus pipeline in spirit: the sources are declared in
one table, every output is a pure function of the downloaded bytes, and the
results are committed so that building the UI never touches the network.

* **Survivor portraits.** The wiki hosts full-body renders, not head shots, so
  each survivor declares where their head sits in the render (as fractions of
  the bitmap). Cropping from the render keeps all eight avatars on the same
  footing instead of mixing the 43 px L4D1 HUD icons with the 66 px L4D2 ones.
* **Backdrop.** The Special Infected wallpaper; the bottom strip is trimmed to
  drop the fan-site watermark.
* **Mark.** The blood-splattered "4" of the official logo, isolated from the
  rest of the wordmark so the header can print a small L4D badge.

Run it with ``make assets`` (needs Pillow, see ``requirements-dev.txt``).

The artwork is Valve's property. It is used here as fan art in a
non-commercial portfolio project and is credited in the README.
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:  # pragma: no cover - imported lazily for a clear error message
    from PIL.Image import Image

WIKI_API_URL = "https://left4dead.fandom.com/api.php"
USER_AGENT = "L4DSafeRoomTerminal/1.0 (asset pipeline; non-commercial portfolio project)"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = REPOSITORY_ROOT / "frontend" / "public"

AVATAR_DIR = PUBLIC_DIR / "avatars"
BACKGROUND_PATH = PUBLIC_DIR / "backgrounds" / "specials.jpg"
MARK_PATH = PUBLIC_DIR / "brand" / "l4d-mark.png"

AVATAR_SIZE = 192
"""Side of a generated portrait; 4x the largest frame the UI draws it in."""

HEAD_SHARE_OF_FRAME = 0.58
"""Fraction of the avatar the head fills — the rest is hair and shoulders."""

HEADROOM_SHARE = 0.10
"""Space above the head, so a beret or a cap is not glued to the edge."""

HEAD_SHARE_OF_FIGURE = 0.118
"""Head height as a fraction of the survivor's own height.

Every render is a full-body pose, and a human head is consistently about an
eighth of the body, so one ratio frames all eight portraits at the same scale.
"""


@dataclass(frozen=True, slots=True)
class AvatarSource:
    """A survivor portrait and where their head sits inside it."""

    name: str
    """Name as the API returns it — the key the frontend resolves avatars by."""

    slug: str
    wiki_file: str
    head_center_x: float
    """Horizontal centre of the head, as a fraction of the bitmap's width.

    Only the horizontal position is declared: the renders are posed, so each
    survivor leans a different way, and no rule reads the head out of a
    silhouette reliably enough. The vertical extent comes from the measured
    bounding box of the figure, which needs no per-survivor tuning.
    """


# The tall L4D1 and L4D2 renders are 989 px high; Ellis is the one survivor the
# wiki only renders small (251x480), which is also why his portrait is softer.
AVATARS: tuple[AvatarSource, ...] = (
    AvatarSource("Билл", "bill", "Bill 1.png", 0.547),
    AvatarSource("Зои", "zoey", "Zoey 1.png", 0.765),
    AvatarSource("Фрэнсис", "francis", "Francis 1.png", 0.309),
    AvatarSource("Луис", "louis", "Louis 1.png", 0.334),
    AvatarSource("Тренер", "coach", "Render-coach.png", 0.603),
    AvatarSource("Эллис", "ellis", "Render-ellis.png", 0.351),
    AvatarSource("Ник", "nick", "Render-nick.png", 0.533),
    AvatarSource("Рошель", "rochelle", "Render-rochelle2.png", 0.597),
)

BACKGROUND_WIKI_FILE = "Specials 1920x1200 wallpaper.jpg"
BACKGROUND_HEIGHT = 1100
"""Rows kept from the top of the 1920x1200 wallpaper — trims the watermark."""

BACKGROUND_QUALITY = 86

LOGO_WIKI_FILE = "Left 4 Dead Logo.svg"
MARK_SIZE = 256
"""Square side of the generated badge — 8x the height the header draws it at.

The splatter is traced from a 790x400 render, so the badge is already close to
its native resolution; growing it further would only cost bytes.
"""

MARK_DRIP_RADIUS = 34
"""Blood droplets within this many pixels of the splat join the badge."""

RED_CHANNEL_MINIMUM = 90
RED_DOMINANCE = 1.5
"""A pixel counts as blood when red beats green and blue by this factor."""


class AssetError(RuntimeError):
    """Raised when an upstream asset cannot be fetched or rasterised."""


def _ensure_pillow() -> None:
    """Fail early with an actionable message when Pillow is not installed."""
    try:
        import PIL  # noqa: F401, PLC0415 - optional dependency
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise AssetError(
            "Pillow is required to build the artwork: "
            "pip install -r requirements-dev.txt"
        ) from error


def _wiki_image_url(client: httpx.Client, wiki_file: str) -> str:
    """Resolve a wiki file title to a direct CDN URL."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": f"File:{wiki_file}",
        }
    )
    response = client.get(f"{WIKI_API_URL}?{query}")
    response.raise_for_status()
    pages = response.json()["query"]["pages"].values()
    for page in pages:
        info = page.get("imageinfo")
        if info:
            return str(info[0]["url"])
    raise AssetError(f"the wiki has no file named {wiki_file!r}")


def download(client: httpx.Client, wiki_file: str) -> bytes:
    """Fetch the original bitmap of a wiki file."""
    url = _wiki_image_url(client, wiki_file)
    response = client.get(url)
    response.raise_for_status()
    return response.content


def _opaque_cover(image: Image) -> Image:
    """Return the bitmap with its transparency spread over black."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    canvas.paste(image, (0, 0), image)
    return canvas


def crop_avatar(image: Image, source: AvatarSource, size: int = AVATAR_SIZE) -> Image:
    """Cut the declared head out of a render and frame it as a square avatar."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    width, height = image.size
    figure = image.getchannel("A").point(lambda value: 255 if value > 40 else 0).getbbox()
    if figure is None:
        raise AssetError(f"the render for {source.name!r} is empty")
    figure_height = figure[3] - figure[1]

    frame = round(HEAD_SHARE_OF_FIGURE * figure_height / HEAD_SHARE_OF_FRAME)
    left = round(source.head_center_x * width - frame / 2)
    top = round(figure[1] - HEADROOM_SHARE * frame)

    cover = _opaque_cover(image)
    # A pose can lean far enough to push the square past the bitmap edge; the
    # transparent margin that appears is what the avatar's own frame hides.
    framed = cover.crop((left, top, left + frame, top + frame))
    return framed.resize((size, size), Image.Resampling.LANCZOS)


def trim_background(image: Image) -> Image:
    """Drop the watermark strip the fan-site wallpaper carries at the bottom."""
    return image.convert("RGB").crop((0, 0, image.width, min(BACKGROUND_HEIGHT, image.height)))


def _blood_components(image: Image) -> list[list[tuple[int, int]]]:
    """Group the red pixels into connected blobs, largest first."""
    import numpy as np  # noqa: PLC0415 - optional dependency, kept local

    pixels = np.asarray(image.convert("RGBA"), dtype=np.int16)
    red, green, blue, alpha = (pixels[:, :, index] for index in (0, 1, 2, 3))
    blood = (
        (alpha > 40)
        & (red > RED_CHANNEL_MINIMUM)
        & (red > green * RED_DOMINANCE)
        & (red > blue * RED_DOMINANCE)
    )

    height, width = blood.shape
    visited = np.zeros_like(blood)
    components: list[list[tuple[int, int]]] = []

    for start_y, start_x in zip(*np.nonzero(blood), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        queue = deque([(int(start_x), int(start_y))])
        blob: list[tuple[int, int]] = []
        while queue:
            current_x, current_y = queue.popleft()
            blob.append((current_x, current_y))
            for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_x, next_y = current_x + offset_x, current_y + offset_y
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and blood[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        components.append(blob)

    components.sort(key=len, reverse=True)
    return components


def crop_mark(image: Image, size: int = MARK_SIZE) -> Image:
    """Isolate the blood-splattered "4" from the logo and centre it on a square.

    The numeral itself is a knockout in the source vector, so the badge shows
    whatever sits behind it — on the dark header that reads as the ink of the
    glyph, which is exactly how the logo behaves when it is printed on black.
    """
    from PIL import Image  # noqa: PLC0415 - optional dependency

    components = _blood_components(image)
    if not components:
        raise AssetError("the logo contains no red ink to cut a badge from")

    splatter = components[0]
    left = min(x for x, _ in splatter)
    right = max(x for x, _ in splatter)
    top = min(y for _, y in splatter)
    bottom = max(y for _, y in splatter)

    # The droplets falling off the splatter are part of the mark's silhouette.
    selected = list(splatter)
    for blob in components[1:]:
        center_x = sum(x for x, _ in blob) / len(blob)
        center_y = sum(y for _, y in blob) / len(blob)
        if (
            left - MARK_DRIP_RADIUS <= center_x <= right + MARK_DRIP_RADIUS
            and top - MARK_DRIP_RADIUS <= center_y <= bottom + MARK_DRIP_RADIUS
        ):
            selected.extend(blob)

    xs = [x for x, _ in selected]
    ys = [y for _, y in selected]
    padding = 4
    box = (
        min(xs) - padding,
        min(ys) - padding,
        max(xs) + padding + 1,
        max(ys) + padding + 1,
    )
    ink = image.crop(box)
    side = max(ink.size)
    badge = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    badge.alpha_composite(ink, ((side - ink.width) // 2, (side - ink.height) // 2))
    return badge.resize((size, size), Image.Resampling.LANCZOS)


def rasterise_svg(data: bytes) -> Image:
    """Rasterise an SVG, using whichever renderer the machine provides."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    try:
        import cairosvg  # type: ignore[import-not-found]  # noqa: PLC0415 - optional

        with Image.open(io.BytesIO(cairosvg.svg2png(bytestring=data))) as rendered:
            return rendered.convert("RGBA")
    except ModuleNotFoundError:
        pass

    if shutil.which("rsvg-convert") is not None:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["rsvg-convert", "-f", "png"],  # noqa: S607 - resolved above
            input=data,
            capture_output=True,
            check=True,
        )
        with Image.open(io.BytesIO(completed.stdout)) as rendered:
            return rendered.convert("RGBA")

    if shutil.which("sips") is not None:
        # macOS ships sips, which rasterises through QuickLook but only reads
        # from disk, hence the temporary file.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "logo.svg"
            target = Path(directory) / "logo.png"
            source.write_bytes(data)
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["sips", "-s", "format", "png", str(source), "--out", str(target)],  # noqa: S607
                capture_output=True,
                check=True,
            )
            with Image.open(target) as rendered:
                return rendered.convert("RGBA")

    raise AssetError(
        "no SVG renderer found: install cairosvg (or librsvg) to rebuild the badge"
    )


def _write(image: Image, path: Path, **save_options: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **save_options)
    print(f"  {path.relative_to(REPOSITORY_ROOT)}  {image.width}x{image.height}")


def build_avatars(client: httpx.Client) -> None:
    """Cut every survivor's portrait out of their wiki render."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    print("survivor portraits")
    for source in AVATARS:
        with Image.open(io.BytesIO(download(client, source.wiki_file))) as render:
            avatar = crop_avatar(render.convert("RGBA"), source)
        _write(avatar, AVATAR_DIR / f"{source.slug}.png", optimize=True)


def build_background(client: httpx.Client) -> None:
    """Store the Special Infected wallpaper as the app backdrop."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    print("backdrop")
    with Image.open(io.BytesIO(download(client, BACKGROUND_WIKI_FILE))) as wallpaper:
        backdrop = trim_background(wallpaper)
    _write(backdrop, BACKGROUND_PATH, quality=BACKGROUND_QUALITY, optimize=True, progressive=True)


def build_mark(client: httpx.Client) -> None:
    """Cut the L4D badge out of the official logo."""
    print("terminal badge")
    logo = rasterise_svg(download(client, LOGO_WIKI_FILE))
    _write(crop_mark(logo), MARK_PATH, optimize=True)


BUILDERS = {
    "avatars": build_avatars,
    "background": build_background,
    "mark": build_mark,
}


def main(argv: list[str] | None = None) -> int:
    """Download the artwork and write every derived asset into ``public/``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(BUILDERS),
        help="build just this group of assets (repeatable); defaults to all",
    )
    arguments = parser.parse_args(argv)
    selected = arguments.only or sorted(BUILDERS)

    try:
        _ensure_pillow()
    except AssetError as error:
        print(error, file=sys.stderr)
        return 1

    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        try:
            for group in selected:
                BUILDERS[group](client)
        except (AssetError, httpx.HTTPError) as error:
            print(f"asset build failed: {error}", file=sys.stderr)
            return 1

    print("done — commit the refreshed files under frontend/public/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
