// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { decodeJwtSubject } from "../src/features/profile/utils/jwt-subject.ts";
import { initialsFromName } from "../src/features/profile/utils/avatar-initials.ts";

test("account identity supplies the sidebar name and avatar unless a display name is set", () => {
  let token: string | null = null;
  const profile = { displayName: "", nickname: "", avatarDataUrl: null };
  const { useEffectiveProfile: renderProfile, OWNER_DISPLAY_NAME } = loadWithStubs<{
    useEffectiveProfile: () => {
      displayTitle: string;
      addressName: string;
      sessionSub: string | null;
    };
    OWNER_DISPLAY_NAME: string;
  }>(
    new URL(
      "../src/features/profile/hooks/use-effective-profile.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": { getAuthToken: () => token, OWNER_USERNAME: "unsloth" },
      "../utils/jwt-subject": { decodeJwtSubject },
      "../stores/user-profile-store": {
        useUserProfileStore: (select: (state: typeof profile) => unknown) =>
          select(profile),
      },
    },
  );

  // A managed account shows the username its owner chose, verbatim. The owner's own id is the
  // reserved literal "unsloth", an auth identifier rather than a name, so that ONE id displays as
  // OWNER_DISPLAY_NAME instead of leaking the id into the UI. sessionSub keeps the real subject
  // either way, and the avatar initials follow whatever is displayed.
  for (const [username, shown, initials] of [
    ["unsloth", OWNER_DISPLAY_NAME, OWNER_DISPLAY_NAME[0]],
    ["alice", "alice", "A"],
    ["bob", "bob", "B"],
  ]) {
    token = `test.${Buffer.from(JSON.stringify({ sub: username })).toString("base64url")}.test`;
    const effective = renderProfile();
    assert.equal(effective.displayTitle, shown);
    assert.equal(effective.addressName, shown);
    assert.equal(effective.sessionSub, username);
    assert.equal(initialsFromName(effective.displayTitle), initials.toUpperCase());
  }

  profile.displayName = "  Robert Smith  ";
  profile.nickname = "Rob";
  assert.equal(renderProfile().displayTitle, "Robert Smith");
  assert.equal(renderProfile().addressName, "Rob");

  profile.displayName = "  ";
  assert.equal(renderProfile().displayTitle, "bob");
  for (const invalidToken of [null, "invalid-token"]) {
    // No usable subject at all: the display still has to say something, and it has to be the
    // owner's name rather than the reserved id.
    token = invalidToken;
    assert.equal(renderProfile().displayTitle, OWNER_DISPLAY_NAME);
  }
});
