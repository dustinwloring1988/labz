// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Curated list of the character cutouts offered as profile pictures.
//
// This used to be 21 sloth poses picked out of `public/Sloth emojis` (~38 PNGs), filtered to the
// subset that was effectively square and low on whitespace so each one filled the round avatar
// frame. There is no pose set for this character, only three cutouts, so the list is those three
// rather than three selected from a larger folder. All three are square, tightly cropped and
// transparent, so the frame crops none of them.
//
// Paths are relative to the public folder; resolve with `publicAssetUrl(...)` before using as an
// <img> src so subpath deploys are handled.
export const MASCOT_AVATARS: readonly string[] = [
  "labz-gem.png",
  "labz-head.png",
  "sticker.png",
];
