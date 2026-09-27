#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render the LABZ display art from the sources in brand/source.

Every logo, app icon and installer bitmap the product ships is derived here rather than
hand-committed, so there is one place that says what the artwork is made of and one command
that rebuilds it. Output paths and pixel dimensions are contracts: the NSIS bitmaps are pinned
by tests/studio/test_tauri_branding_contract.py, and the icon set is pinned by
tests/test_installer_shortcut_icons.py and the pyproject package_data that install.sh reads.

Sources, all RGBA with a real alpha channel:
  sticker-peel.png  500x500  the disc that carries the mark wherever it is round
  head.png          500x500  the head alone, for the monochrome tray silhouette
  gem.png           500x500  the character holding the gem, for the console placeholders
  logotext.png      865x289  the LABZ wordmark, black with a white keyline

The round mark is a disc, but a disc dropped straight into a square icon reads as a small dot
at taskbar sizes, so the app icon is a composite: the disc on a brand-green rounded square.
The disc's own white keyline is what separates the two at 16px, which is why the square is
this bright rather than a tint of the disc.

Platform containers (icon.ico, icon.icns and the PNG size set) are produced by `tauri icon`
rather than by Pillow. Pillow's ICO writer is fine, but its ICNS writer only emits ic07-ic14,
which silently drops the 16x16 and 32x32 1x entries macOS wants, and it stores every entry as
an uncompressed-depth RGBA PNG. `tauri icon` emits the full ten-entry set with the legacy
is32/l8mk chunks for the two small sizes, which is what the file this replaces contained.

Usage:
  python scripts/make_brand_assets.py           write every asset
  python scripts/make_brand_assets.py --check   fail if a committed byte is stale
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "brand/source"

TAURI_ICONS = REPO / "studio/src-tauri/icons"
BRANDING = REPO / "studio/src-tauri/windows/branding"
PUBLIC = REPO / "studio/frontend/public"

# The composited app icon. 1024 is what `tauri icon` wants for the top of its size ladder and
# what install.sh copies out as the Linux .desktop icon.
MASTER = 1024

# Sampled off the disc's own outer ring, so the square and the mark are the same green.
BRAND_GREEN = "#BDFB10"

# The disc nearly fills the square. Any smaller and it stops reading as an icon at 32px; any
# larger and the lime ring of the disc merges into the lime field behind it.
DISC_SCALE = 0.85

# Matches the squircle Windows and macOS both use for an app icon, so the icon does not sit in
# a differently-shaped hole once the OS masks it.
CORNER_RADIUS = 0.22

# 256 colours. Measured imperceptible against the full-depth master at 1:1 on the smooth metal
# and hair gradients, and it takes a third off the platform containers, because `tauri icon`
# re-encodes every entry as RGBA and deflate does better on fewer distinct values.
PALETTE = 256

# NSIS geometry. These are not free choices: MUI_HEADERIMAGE_BITMAP and
# MUI_WELCOMEFINISHPAGE_BITMAP are drawn at a fixed size, and
# tests/studio/test_tauri_branding_contract.py asserts all three numbers per file.
NSIS_HEADER = (300, 114)
NSIS_SIDEBAR = (328, 628)

# logotext.png is solid white glyphs on transparency, drawn to sit on a dark surface. Both the
# NSIS pages and the light-themed web chrome are light, so the mark is recoloured rather than
# recomposited: its alpha channel is already exact coverage, so filling it is lossless and the
# antialiased edges come through untouched. The brand green is far too light to carry a heavy
# display face on white, so this is a near-black, matching the black text the bitmaps used
# before and letting the disc supply the colour.
WORDMARK_INK = (17, 17, 17, 255)

# NSIS composites its pages over whatever the wizard paints, and neither MUI macro can blend
# an alpha channel, so the bitmaps have to carry their own background. White matches the
# wizard page these sit on.
NSIS_BACKGROUND = (255, 255, 255)

# The tray mark is a macOS template image: one colour plus alpha, and the OS recolours it for
# light and dark menu bars. So the shape is carried entirely by the alpha channel and the ink
# is only ever black or white.
TRAY_INK_LIGHT = (0, 0, 0, 255)
TRAY_INK_DARK = (255, 255, 255, 255)

# head.png is a filled, heavily textured illustration and a tray icon is 18px in one colour, so
# this is a reduction with a real cost: what survives at 18px is the outline, and head.png's
# outline is very nearly a circle. The blur welds the frizzy hair into one mass instead of
# leaving a speckle of detached strands, and the threshold is set high enough to drop the
# translucent hair tips so the result is a clean, deliberate oval rather than noise.
#
# Measured and rejected: thresholding the luminance for edges (the frizzy hair reads as a
# hairball), and cutting the gem out of gem.png (the hands occlude it into an unidentifiable
# speck). The sloth tray icon this replaces was line art, so its alpha carried interior detail
# that a filled illustration cannot. If this mark needs to be identifiable rather than merely
# clean, it wants a purpose-drawn one-colour glyph, not a reduction of a painting.
TRAY_BLUR_RADIUS = 2.0
TRAY_THRESHOLD = 200

# Sizes the app renders, chosen to match what each file already was so no CSS has to move.
# Each entry is (source art, pixel size).
PUBLIC_ASSETS: dict[str, tuple[str, int]] = {
    "sticker.png": ("sticker-peel.png", 512),
    "circle-logo-small.png": ("sticker-peel.png", 128),
    "favicon.png": ("sticker-peel.png", 96),
    "rounded.png": ("sticker-peel.png", 512),
    "rounded-512.png": ("sticker-peel.png", 512),
    # The two character cutouts, for the places the old art used a sloth pose: the rotating
    # greeting, the profile avatar picker, the empty states and the sign-in panel.
    "labz-gem.png": ("gem.png", 512),
    "labz-head.png": ("head.png", 512),
}
PUBLIC_LOGOTEXT_WIDTH = 960

# mascot-img.tsx bundles this one as a data URI so the fallback can never 404, which means it
# lives beside the source rather than in public/ and is the only asset here that is not a PNG.
# The disc is the right subject for it: it is the mark, so it still reads as the brand if it ever
# shows up next to something that failed to load.
FRONTEND_ASSETS = REPO / "studio/frontend/src/assets"
FALLBACK_NAME = "mascot-fallback.webp"
FALLBACK_SIZE = 128
FALLBACK_QUALITY = 90

# The Windows shortcut icon install.ps1 and install.sh drop on the desktop and Start menu. Six
# sizes, which is what unsloth.ico carried and what Explorer picks between.
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# What `tauri icon` emits, and the five of them this repository commits. The rest -- the
# android/ and ios/ trees, 64x64.png, 128x128@2x.png and the Square*Logo/StoreLogo set -- are
# already in .gitignore.
TAURI_KEEP = ("icon.png", "32x32.png", "128x128.png", "icon.ico", "icon.icns")

# `tauri icon` writes the ICNS in two runs over the same master with different bytes, so it
# cannot be held to a byte comparison the way everything else here is. See main().
NON_REPRODUCIBLE = frozenset({"icon.icns"})


def source(name: str) -> Image.Image:
    path = SOURCE / name
    if not path.exists():
        raise SystemExit(
            f"missing brand source {path}\n"
            "brand/source holds the only copy of the artwork this script renders from."
        )
    return Image.open(path).convert("RGBA")


def fit(img: Image.Image, size: int) -> Image.Image:
    """Square resize, from a source that is already square or nearly so."""
    width, height = img.size
    side = min(width, height)
    box = ((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2)
    return img.crop(box).resize((size, size), Image.LANCZOS)


def icon_master() -> Image.Image:
    """The composited 1024 master: the disc on the brand-green rounded square."""
    canvas = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rounded_rectangle(
        [0, 0, MASTER - 1, MASTER - 1],
        radius = int(MASTER * CORNER_RADIUS),
        fill = BRAND_GREEN,
    )
    disc = fit(source("sticker-peel.png"), int(MASTER * DISC_SCALE))
    canvas.alpha_composite(disc, ((MASTER - disc.width) // 2, (MASTER - disc.height) // 2))
    # FASTOCTREE rather than the default median cut: the artwork is mostly saturated lime and
    # black, and octree holds those without spending the palette on the skin tones.
    return canvas.quantize(colors = PALETTE, method = Image.FASTOCTREE, dither = Image.Dither.NONE)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format = "PNG", optimize = True)
    return buf.getvalue()


def bmp24_bytes(img: Image.Image) -> bytes:
    """A 24-bit BMP. NSIS reads the pixel format directly and cannot blend alpha, so an RGBA
    BMP here renders as black where the mask should be transparent."""
    if img.mode != "RGB":
        raise ValueError(f"NSIS bitmaps must be flattened to RGB, got {img.mode}")
    buf = io.BytesIO()
    img.save(buf, format = "BMP")
    return buf.getvalue()


def wordmark(width: int, ink: tuple[int, int, int, int] = WORDMARK_INK) -> Image.Image:
    """The LABZ wordmark at `width`, recoloured to `ink`."""
    mark = source("logotext.png")
    height = round(mark.height * width / mark.width)
    mask = mark.getchannel("A").resize((width, height), Image.LANCZOS)
    out = Image.new("RGBA", (width, height), ink[:3] + (0,))
    out.putalpha(mask)
    return out


def nsis_header() -> Image.Image:
    """The wizard header: the wordmark alone, centred in the 300x114 strip."""
    canvas = Image.new("RGB", NSIS_HEADER, NSIS_BACKGROUND)
    # 240 of 300 leaves the margin NSIS reserves for the page's own text column.
    mark = wordmark(240)
    canvas.paste(mark, ((NSIS_HEADER[0] - mark.width) // 2, (NSIS_HEADER[1] - mark.height) // 2), mark)
    return canvas


def nsis_sidebar() -> Image.Image:
    """The welcome/finish page sidebar: wordmark over the mark, on the 328x628 panel."""
    width, height = NSIS_SIDEBAR
    canvas = Image.new("RGB", NSIS_SIDEBAR, NSIS_BACKGROUND)

    mark = wordmark(264)
    canvas.paste(mark, ((width - mark.width) // 2, 96), mark)

    disc = fit(source("sticker-peel.png"), 248)
    canvas.paste(disc, ((width - disc.width) // 2, 232), disc)
    return canvas


def tray_icon(size: int, ink: tuple[int, int, int, int]) -> Image.Image:
    """A monochrome template mark reduced from head.png.

    A template image is one colour plus alpha and macOS recolours it for light and dark menu
    bars, so the shape has to be carried entirely by coverage. See TRAY_BLUR_RADIUS for what
    this reduction gives up.
    """
    alpha = source("head.png").getchannel("A")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius = TRAY_BLUR_RADIUS))
    # point() maps each level through a table, so this thresholds rather than scaling.
    solid = alpha.point(lambda v: 255 if v >= TRAY_THRESHOLD else 0)
    # Scaling a hard mask puts the antialiasing back, which is what a template image wants:
    # the OS recolours the ink and keeps the coverage.
    mask = solid.resize((size, size), Image.LANCZOS)
    shell = Image.new("RGBA", (size, size), ink[:3] + (0,))
    shell.putalpha(mask)
    return shell


def run_tauri_icon(master: Image.Image, workdir: Path) -> dict[str, bytes]:
    """Hand the master to `tauri icon` and collect back the five files this repo commits."""
    cli = REPO / "studio/node_modules/.bin/tauri.cmd"
    if not cli.exists():
        raise SystemExit(
            f"missing {cli}\nRun `npm ci --prefix studio` first: the platform icon containers are\n"
            "produced by the pinned Tauri CLI, not by Pillow."
        )
    source_png = workdir / "master.png"
    master.save(source_png, format = "PNG", optimize = True)

    out = workdir / "icons"
    result = subprocess.run(
        [str(cli), "icon", str(source_png), "-o", str(out)],
        capture_output = True,
        text = True,
    )
    if result.returncode != 0:
        raise SystemExit(f"tauri icon failed:\n{result.stdout}\n{result.stderr}")

    produced: dict[str, bytes] = {}
    for name in TAURI_KEEP:
        path = out / name
        if not path.exists():
            raise SystemExit(f"tauri icon did not produce {name}, which tauri.conf.json needs")
        produced[name] = path.read_bytes()

    # `tauri icon` writes icon.png at 512 even from a 1024 master, and the committed one has to
    # be 1024: install.sh prefers it over the 512px rounded variant for the Linux .desktop icon,
    # and tests/test_installer_shortcut_icons.py pins the dimensions off the IHDR. The master is
    # right here, so the file is written from it rather than taken from the CLI.
    produced["icon.png"] = png_bytes(master.convert("RGBA"))
    return produced


def icns_entry_sizes(data: bytes) -> set[tuple[int, int, int]]:
    """The (width, height, scale) set an ICNS actually offers, read back through Pillow.

    Used instead of a byte comparison because `tauri icon` does not encode the ICNS
    reproducibly -- two runs over the same master differ -- so the only stable thing to
    assert is that every size macOS will ask for is present.
    """
    from PIL import IcnsImagePlugin  # noqa: F401  (registers the reader)

    path = Path(tempfile.mkdtemp()) / "probe.icns"
    try:
        path.write_bytes(data)
        return set(Image.open(path).info.get("sizes", []))
    finally:
        shutil.rmtree(path.parent, ignore_errors = True)


# Every macOS icon slot `tauri icon` is expected to fill: 16 through 512 at 1x, plus the
# retina doubles. A missing one is a silently blurry icon in Finder or on a retina display.
ICNS_REQUIRED = {
    (16, 16, 1), (16, 16, 2), (32, 32, 1), (32, 32, 2),
    (128, 128, 1), (128, 128, 2), (256, 256, 1), (256, 256, 2),
    (512, 512, 1), (512, 512, 2),
}


def public_asset(name: str) -> bytes:
    art, size = PUBLIC_ASSETS[name]
    return png_bytes(fit(source(art), size))


def fallback_bytes() -> bytes:
    buf = io.BytesIO()
    fit(source("sticker-peel.png"), FALLBACK_SIZE).save(
        buf, format = "WEBP", quality = FALLBACK_QUALITY, method = 6
    )
    return buf.getvalue()


def build() -> dict[Path, bytes]:
    """Render every asset, keyed by the path it belongs at."""
    out: dict[Path, bytes] = {}

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        master = icon_master()

        for name, data in run_tauri_icon(master, workdir).items():
            out[TAURI_ICONS / name] = data

        # install.sh falls back to these when the packaged icns is unusable.
        out[PUBLIC / "labz.ico"] = _ico(master, ICO_SIZES)
        for name in PUBLIC_ASSETS:
            out[PUBLIC / name] = public_asset(name)
        out[PUBLIC / "logotext.png"] = png_bytes(wordmark(PUBLIC_LOGOTEXT_WIDTH))
        out[FRONTEND_ASSETS / FALLBACK_NAME] = fallback_bytes()

        out[BRANDING / "nsis-header.bmp"] = bmp24_bytes(nsis_header())
        out[BRANDING / "nsis-sidebar.bmp"] = bmp24_bytes(nsis_sidebar())

        out[TAURI_ICONS / "tray-icon.png"] = png_bytes(tray_icon(18, TRAY_INK_LIGHT))
        out[TAURI_ICONS / "tray-icon@2x.png"] = png_bytes(tray_icon(36, TRAY_INK_LIGHT))
        out[TAURI_ICONS / "tray-icon-light.png"] = png_bytes(tray_icon(36, TRAY_INK_LIGHT))
        out[TAURI_ICONS / "tray-icon-dark.png"] = png_bytes(tray_icon(36, TRAY_INK_DARK))

    return out


def _ico(master: Image.Image, sizes: list[tuple[int, int]]) -> bytes:
    buf = io.BytesIO()
    master.convert("RGBA").save(buf, format = "ICO", sizes = sizes)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--check",
        action = "store_true",
        help = "render into memory and fail if any committed asset differs",
    )
    args = parser.parse_args()

    assets = build()

    if args.check:
        stale: list[str] = []
        for path, data in sorted(assets.items()):
            if path.name in NON_REPRODUCIBLE:
                continue
            if not path.exists():
                stale.append(f"  missing  {path.relative_to(REPO)}")
            elif path.read_bytes() != data:
                stale.append(f"  stale    {path.relative_to(REPO)}")

        # `tauri icon` does not encode the ICNS reproducibly -- two runs over the same master
        # differ byte for byte -- so the only stable thing to assert about it is that every size
        # macOS will ask for is present. A missing entry is a silently blurry icon.
        icns = assets.get(TAURI_ICONS / "icon.icns")
        if icns is not None:
            missing = ICNS_REQUIRED - icns_entry_sizes(icns)
            if missing:
                stale.append(f"  icon.icns missing sizes {sorted(missing)}")

        if stale:
            print("brand assets are out of date:", file = sys.stderr)
            print("\n".join(stale), file = sys.stderr)
            print("\nRun: python scripts/make_brand_assets.py", file = sys.stderr)
            return 1
        print(f"brand assets are current ({len(assets) - len(NON_REPRODUCIBLE)} byte-exact, "
              f"icon.icns held to its size set)")
        return 0

    for path, data in sorted(assets.items()):
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(data)
        print(f"  {path.relative_to(REPO)}  ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
