# LABZ Docker Studio branding

The JupyterLab image carries the LABZ brand across the login page, the top bar,
the loading splash and the Help > About dialog. Unsloth granted permission for
the rename, so the product name and the attribution line are LABZ's. The licence
did not come with it: the image is still a derivative of the Unsloth Docker
Studio and JupyterLab image and is still conveyed under the GNU AGPLv3, and the
notices that obligation requires are still required. See [../NOTICE](../NOTICE)
and [/studio/LICENSE.AGPL-3.0](../../studio/LICENSE.AGPL-3.0).

## What must stay

- `Licensed under Apache 2.0 and the GNU AGPLv3`.
- The AGPLv3 licence file itself, at `share/jupyter/UNSLOTH_LICENSE.AGPL-3.0`.
- The Source, Unsloth reference, License, AGPLv3 and Apache links in the About
  dialog.
- The `LABZ` product name, the `LABZ Dark` theme, and the logo in the top bar
  and on the splash.

The attribution line itself is no longer load-bearing: `Built by the LABZ team`
and `Copyright 2026-Present the LABZ team` are branding, and the guard still
checks them because it checks the whole set, not because the licence requires
those particular words.

The canonical strings live in `labz_branding.py` and its TypeScript mirror
`labext/src/branding.ts`. The `PHRASE` literal must be byte-identical between
the two, because the guard greps the built labextension bundle for it.

## Where it lives

| File | Carries |
| --- | --- |
| `login.html` | JupyterLab login page and the notice footer. |
| `labext/src/branding.ts` | Canonical strings (TS mirror). |
| `labext/src/about.ts` | Help > About dialog and the license links. |
| `labext/src/splash.ts` | Loading-splash caption. |
| `labext/src/logo.ts` | Embedded LABZ logo data URI. Generated. |
| `labz_branding.py` | Canonical strings and the integrity guard. |

`logo.png`, `favicon.ico` and `labext/src/logo.ts` are generated. Regenerate
with `python scripts/make_jupyter_assets.py` and verify with `--check`; the
committed bytes must match or CI should fail.

## How it is enforced

`labz_branding.py` verifies the notices are present and unaltered in three
places (see [../Dockerfile.studio](../Dockerfile.studio) and
[../studio_launch.sh](../studio_launch.sh)):

1. **Build time:** `python -m labz_branding --verify` fails the image build if
   any notice asset is missing or altered.
2. **Whole image:** `studio_launch.sh` re-runs the same check before starting
   supervisord; a failure refuses to start the container.
3. **JupyterLab:** the module is also a `jupyter_server` extension that re-checks
   on load and refuses to serve JupyterLab if a notice was stripped after the
   container started.

The guard is a tripwire, not a lock. Anyone who forks the source controls the
build and can edit any of these files. It exists to make accidental removal fail
loudly and to make deliberate removal unambiguous. The notices it protects are
required by the AGPLv3 (see [../NOTICE](../NOTICE)), and removing them before
conveying or network-serving the image is a licence violation.
