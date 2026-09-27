// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ComponentProps, useState } from "react";

// Bundled as a data URI so the fallback ships inside the JS bundle and can
// never 404, unlike the public-folder originals.
import fallbackMascot from "@/assets/mascot-fallback.webp?inline";

const LEADING_SLASH = /^\//;

// Resolve a public-folder asset against the deploy base (subpath mounts) and
// percent-encode anything in the path a URL cannot carry literally.
export function publicAssetUrl(path: string): string {
  return encodeURI(import.meta.env.BASE_URL + path.replace(LEADING_SLASH, ""));
}

type MascotImgProps = { src: string } & Omit<
  ComponentProps<"img">,
  "src" | "onError"
>;

// Keying on src remounts the inner component, so the retry state resets
// whenever the source changes (the greeting mark rotates).
export function MascotImg(props: MascotImgProps) {
  return <MascotImgInner key={props.src} {...props} />;
}

type Stage = "primary" | "retry" | "fallback";

// Decorative mark that degrades gracefully: a failed load retries once with
// a cache-buster (transient blips, stale caches), then swaps to the bundled
// fallback. Empty alt by default so a broken image never paints text.
function MascotImgInner({ src, alt = "", ...rest }: MascotImgProps) {
  const [stage, setStage] = useState<Stage>("primary");

  const url = publicAssetUrl(src);
  const effectiveSrc =
    stage === "primary"
      ? url
      : stage === "retry"
        ? `${url}${url.includes("?") ? "&" : "?"}retry=1`
        : fallbackMascot;

  return (
    <img
      draggable={false}
      {...rest}
      src={effectiveSrc}
      alt={alt}
      onError={() => setStage((s) => (s === "primary" ? "retry" : "fallback"))}
    />
  );
}
