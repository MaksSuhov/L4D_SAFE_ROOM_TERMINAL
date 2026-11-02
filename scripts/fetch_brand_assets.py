#!/usr/bin/env python3
"""Готовит графику, которая оформляет интерфейс терминала, из вики Left 4 Dead.

По духу скрипт повторяет конвейер корпуса: источники объявлены в одной таблице,
каждый результат — чистая функция от скачанных байтов, а результаты закоммичены,
поэтому сборка интерфейса никогда не обращается к сети.

* **Портреты выживших.** В вики лежат рендеры в полный рост, а не головы, поэтому
  каждый выживший объявляет, где на рендере сидит его голова (в долях от
  битмапа). Обрезка из рендера держит все восемь аватаров в одном масштабе и не
  смешивает HUD-иконки L4D1 в 43 px с иконками L4D2 в 66 px.
* **Фон.** Обои Special Infected; нижняя полоса обрезана, чтобы убрать вотермарку
  фан-сайта.
* **Знак.** Забрызганная кровью «4» официального логотипа, выделенная из
  остального начертания, чтобы шапка могла напечатать маленький бейдж L4D.

Запуск: ``make assets`` (нужен Pillow, см. ``requirements-dev.txt``).

Графика принадлежит Valve. Здесь она используется как фанатское творчество в
некоммерческом портфолио-проекте и указана в README.
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
"""Сторона сгенерированного портрета; в 4 раза больше самого крупного кадра,
в котором его рисует интерфейс."""

HEAD_SHARE_OF_FRAME = 0.58
"""Доля аватара, которую занимает голова, — остальное волосы и плечи."""
HEADROOM_SHARE = 0.10
"""Отступ над головой, чтобы берет или кепка не прилипали к краю."""

HEAD_SHARE_OF_FIGURE = 0.118
"""Высота головы как доля от собственного роста выжившего.

Каждый рендер — это поза в полный рост, а человеческая голова всегда составляет
примерно восьмую часть тела, поэтому одно соотношение кадрирует все восемь
портретов в одном масштабе.
"""


@dataclass(frozen=True, slots=True)
class AvatarSource:
    """Портрет выжившего и то, где на нём сидит его голова."""

    name: str
    """Имя в том виде, в каком его возвращает API, — ключ, по которому фронтенд
    находит аватары."""

    slug: str
    wiki_file: str
    head_center_x: float
    """Горизонтальный центр головы, как доля от ширины битмапа.

    Объявлена только горизонтальная позиция: рендеры — это позы, поэтому каждый
    выживший наклонён по-своему, и ни одно правило не вычитывает голову из силуэта
    достаточно надёжно. Вертикальные границы берутся из измеренного ограничивающего
    прямоугольника фигуры, который не требует подгонки под каждого выжившего.
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
"""Сколько строк оставлено сверху от обоев 1920x1200 — так срезается вотермарка."""
BACKGROUND_QUALITY = 86

LOGO_WIKI_FILE = "Left 4 Dead Logo.svg"
MARK_SIZE = 256
"""Сторона квадрата сгенерированного бейджа — в 8 раз больше высоты, на которой его
рисует шапка.

Брызги обведены с рендера 790x400, поэтому бейдж уже близок к своему нативному
разрешению; дальнейшее увеличение только добавит байтов.
"""

MARK_DRIP_RADIUS = 34
"""Капли крови в пределах стольких пикселей от кляксы попадают в бейдж."""

RED_CHANNEL_MINIMUM = 90
RED_DOMINANCE = 1.5
"""Пиксель считается кровью, когда красный превышает зелёный и синий во столько раз."""


class AssetError(RuntimeError):
    """Возникает, когда вышестоящий ресурс не удаётся скачать или растеризовать."""


def _ensure_pillow() -> None:
    """Раньше упасть с понятным сообщением, если Pillow не установлен."""
    try:
        import PIL  # noqa: F401, PLC0415 - optional dependency
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise AssetError(
            "Pillow is required to build the artwork: "
            "pip install -r requirements-dev.txt"
        ) from error


def _wiki_image_url(client: httpx.Client, wiki_file: str) -> str:
    """Преобразовать название файла вики в прямой URL на CDN."""
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
    """Скачать исходный битмап файла вики."""
    url = _wiki_image_url(client, wiki_file)
    response = client.get(url)
    response.raise_for_status()
    return response.content


def _opaque_cover(image: Image) -> Image:
    """Вернуть битмап, разложив его прозрачность поверх чёрного фона."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    canvas.paste(image, (0, 0), image)
    return canvas


def crop_avatar(image: Image, source: AvatarSource, size: int = AVATAR_SIZE) -> Image:
    """Вырезать объявленную голову из рендера и обрамить её квадратным аватаром."""
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
    """Убрать полосу с вотермаркой, которую обои фан-сайта несут внизу."""
    return image.convert("RGB").crop((0, 0, image.width, min(BACKGROUND_HEIGHT, image.height)))


def _blood_components(image: Image) -> list[list[tuple[int, int]]]:
    """Сгруппировать красные пиксели в связные кляксы, сначала самые крупные."""
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
    """Выделить забрызганную кровью «4» из логотипа и отцентрировать её на квадрате.

    Сама цифра в исходном векторе — это вырез, поэтому бейдж показывает то, что
    стоит за ней, — на тёмной шапке это читается как краска глифа, и именно так
    ведёт себя логотип при печати по чёрному.
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
    """Растеризовать SVG тем рендерером, который есть на машине."""
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
    """Вырезать портрет каждого выжившего из его рендера в вики."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    print("survivor portraits")
    for source in AVATARS:
        with Image.open(io.BytesIO(download(client, source.wiki_file))) as render:
            avatar = crop_avatar(render.convert("RGBA"), source)
        _write(avatar, AVATAR_DIR / f"{source.slug}.png", optimize=True)


def build_background(client: httpx.Client) -> None:
    """Сохранить обои Special Infected как фон приложения."""
    from PIL import Image  # noqa: PLC0415 - optional dependency

    print("backdrop")
    with Image.open(io.BytesIO(download(client, BACKGROUND_WIKI_FILE))) as wallpaper:
        backdrop = trim_background(wallpaper)
    _write(backdrop, BACKGROUND_PATH, quality=BACKGROUND_QUALITY, optimize=True, progressive=True)


def build_mark(client: httpx.Client) -> None:
    """Вырезать бейдж L4D из официального логотипа."""
    print("terminal badge")
    logo = rasterise_svg(download(client, LOGO_WIKI_FILE))
    _write(crop_mark(logo), MARK_PATH, optimize=True)


BUILDERS = {
    "avatars": build_avatars,
    "background": build_background,
    "mark": build_mark,
}


def main(argv: list[str] | None = None) -> int:
    """Скачать графику и записать каждый производный ресурс в ``public/``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(BUILDERS),
        help="собрать только эту группу ассетов (можно повторять); по умолчанию — все",
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
