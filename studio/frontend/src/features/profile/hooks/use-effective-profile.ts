// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken, OWNER_USERNAME } from "@/features/auth";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { useUserProfileStore } from "../stores/user-profile-store";

// The owner's login id is the reserved literal "unsloth" -- that is an auth identifier, not a
// name, and it stays put: validate_account_username rejects it, the desktop-secret path and the
// owner fence both key on it, and changing it would strand every existing install's account row.
// So the name the owner is *shown* as is a separate constant, and it is what the sidebar row in
// the lower left falls back to when no display name has been set.
//
// Named rather than inlined at each site because every surface falling back to the login id has
// to agree on it; the settings profile preview reads the same one.
export const OWNER_DISPLAY_NAME = "Freaky";

// Maps the reserved owner id to the name to show. Only that id maps; a chosen username stays as
// chosen. Shared, not copied: every surface falling back to the login id must agree.
export function loginDisplayName(sessionSub: string | null): string {
  return sessionSub === OWNER_USERNAME ? OWNER_DISPLAY_NAME : (sessionSub ?? "");
}

export function useEffectiveProfile() {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);

  const sessionSub = decodeJwtSubject(getAuthToken());
  const dn = displayName.trim();
  const login = loginDisplayName(sessionSub);
  // Name to address the user by: nickname, else first name, else login id.
  const addressName = nickname.trim() || dn.split(/\s+/)[0] || login;
  return {
    sessionSub,
    displayTitle: dn || login || OWNER_DISPLAY_NAME,
    addressName,
    avatarDataUrl,
  };
}
