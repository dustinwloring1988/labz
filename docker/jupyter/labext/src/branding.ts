// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the LABZ team. See /studio/LICENSE.AGPL-3.0
//
// Canonical branding strings, mirrored from labz_branding.py, which greps the
// built bundle for them. Plain text only, never encoded.

export const PRODUCT = 'LABZ Docker Studio';
export const SHORT_LABEL = 'Built by the LABZ team';
export const SPLASH_LABEL = 'Loading LABZ Docker';
export const COPYRIGHT = 'Copyright 2026-Present the LABZ team';
export const AGPL_NOTICE = 'Licensed under Apache 2.0 and the GNU AGPLv3';
export const WEBSITE_URL = 'https://github.com/dustinwloring1988/labz';
export const DOCS_URL = 'https://unsloth.ai/docs';
export const SOURCE_URL = 'https://github.com/dustinwloring1988/labz';
export const LICENSE_URL = 'https://github.com/dustinwloring1988/labz#license';
export const AGPL_URL = 'https://www.gnu.org/licenses/agpl-3.0.html';
export const APACHE_URL = 'https://www.apache.org/licenses/LICENSE-2.0';

// Must equal PHRASE in labz_branding.py, which greps the built bundle for it.
// ONE literal, not a concatenation, so webpack keeps it contiguous.
export const PHRASE =
  'LABZ Docker Studio and JupyterLab image. Built by the LABZ team. Licensed under Apache 2.0 and the GNU AGPLv3. Source: https://github.com/dustinwloring1988/labz Website: https://github.com/dustinwloring1988/labz';
