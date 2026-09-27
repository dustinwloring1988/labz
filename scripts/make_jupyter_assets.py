#!/usr/bin/env python3
"""Render the JupyterLab branding assets from brand/source.

The desktop app has scripts/make_brand_assets.py; the JupyterLab image needs its
own three assets and they are rendered here rather than copied from the app's
output, because the sizes and the treatment differ:

  logo.png      the mark alone on transparency, for the top bar (24px), the
                About dialog (64px) and the login card (72px). Those sit on the
                "LABZ Dark" theme, so it must not depend on a light background.
  favicon.ico   16/32/48. Small enough that the mark needs the brand-green
                rounded square behind it, the same treatment as the app icon --
                the bare head is unreadable at 16px against browser chrome.
  logo.ts       logo.png as a data URI, so the labextension ships no runtime
                asset. The branding guard greps the built bundle for the
                data:image/png;base64 prefix, so the format is load-bearing.

Run from the repo root:  python scripts/make_jupyter_assets.py [--check]
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "brand" / "source"
OUT_JUPYTER = REPO / "docker" / "jupyter"
OUT_LOGO_TS = OUT_JUPYTER / "labext" / "src" / "logo.ts"

BRAND_GREEN = "#BDFB10"
CORNER_RADIUS = 0.22
MARK_SCALE = 0.86
LOGO_SIZE = 256
FAVICON_SIZES = [(16, 16), (32, 32), (48, 48)]
FAVICON_MASTER = 256
PALETTE = 256


def source(name: str) -> Image.Image:
    path = SOURCE / name
    if not path.exists():
        raise SystemExit(
            "missing brand source %s\n"
            "brand/source holds the only copy of the artwork this script renders from." % path
        )
    return Image.open(path).convert("RGBA")


def fit(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    box = ((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2)
    return img.crop(box).resize((size, size), Image.LANCZOS)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def logo() -> Image.Image:
    """The mark alone, on transparency.

    Quantized to a palette because this gets base64'd into logo.ts: the raw 256px
    RGBA encode is ~123kB and lands in the built bundle as ~165kB of JS, against
    ~19kB for the mark this replaces. The artwork is flat enough that 256 colours
    is visually identical at the 24/64/72px it is ever displayed at.
    """
    return fit(source("head.png"), LOGO_SIZE).quantize(
        colors=PALETTE, method=Image.FASTOCTREE, dither=Image.Dither.NONE
    )


def favicon_master() -> Image.Image:
    """The mark on the brand-green rounded square.

    Rendered once at FAVICON_MASTER and handed to Pillow with an explicit sizes
    list, rather than compositing each size by hand: passing pre-resized frames
    with append_images silently writes a single-entry ICO, which loses 32 and 48.
    """
    size = FAVICON_MASTER
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rounded_rectangle(
        [0, 0, size - 1, size - 1],
        radius=max(1, int(size * CORNER_RADIUS)),
        fill=BRAND_GREEN,
    )
    mark = fit(source("head.png"), int(size * MARK_SCALE))
    canvas.alpha_composite(mark, ((size - mark.width) // 2, (size - mark.height) // 2))
    return canvas.quantize(colors=PALETTE, method=Image.FASTOCTREE, dither=Image.Dither.NONE)


def ico_bytes(master: Image.Image) -> bytes:
    buf = io.BytesIO()
    master.save(buf, format="ICO", sizes=FAVICON_SIZES)
    return buf.getvalue()


def logo_ts(logo_png: bytes) -> str:
    """logo.png as ONE contiguous double-quoted literal.

    Deliberately not wrapped and not concatenated: branding.ts documents that the
    guard greps the built bundle for the data URI prefix, and webpack only keeps a
    string contiguous if it is a single literal. A folded or '+'-joined version
    still typechecks but splits the literal, so the guard's grep misses it.
    """
    b64 = base64.b64encode(logo_png).decode("ascii")
    return (
        "// SPDX-License-Identifier: AGPL-3.0-only\n"
        "// Copyright 2026-Present the LABZ team. See /studio/LICENSE.AGPL-3.0\n"
        "//\n"
        "// Auto-generated: the LABZ logo as a data URI, so the plugin has no runtime asset.\n"
        "// Regenerate with: python scripts/make_jupyter_assets.py\n"
        "export const LABZ_LOGO_DATA_URI =\n"
        '  "data:image/png;base64,' + b64 + '";\n'
    )


def build() -> dict[Path, bytes]:
    logo_png = png_bytes(logo())
    return {
        OUT_JUPYTER / "logo.png": logo_png,
        OUT_JUPYTER / "favicon.ico": ico_bytes(favicon_master()),
        OUT_LOGO_TS: logo_ts(logo_png).encode("utf-8"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed assets differ")
    args = parser.parse_args()

    outputs = build()
    stale = []
    for path, data in outputs.items():
        if args.check:
            current = path.read_bytes() if path.exists() else b""
            if current != data:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print("  wrote %-46s %7d bytes" % (path.relative_to(REPO), len(data)))

    if args.check:
        if stale:
            for path in stale:
                print("  STALE %s" % path.relative_to(REPO))
            return 1
        print("  jupyter branding assets are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
