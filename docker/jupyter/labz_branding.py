# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the LABZ team. See /studio/LICENSE.AGPL-3.0
"""LABZ Docker Studio branding + AGPLv3 integrity guard.

The plain-text source of truth for the branding strings AND the checker that
verifies they are still present, spread over several independent files so a
shallow find-and-replace cannot half-rebrand the image.

What this still guards is the licence, not the brand name. The AGPLv3 grant
survived the rebrand: the image is still a derivative of the Unsloth Docker
Studio and JupyterLab image, so it is still conveyed under the GNU AGPLv3, it
still has to offer its corresponding source to the people it serves over a
network, and it still has to display the licence notice and the upstream
component licences. Unsloth additionally granted permission for the rename, so
the product name and attribution line are now LABZ's rather than a notice the
guard protects. Deleting the licence file, the licence notice, or the source
and licence links is still a violation, and still fails the build.

Everything here is plain readable text: the only base64 blob is the logo data
URI.

Runs at image build time, from studio_launch.sh before supervisord, and as a
jupyter_server extension.
"""

import json
import os
import sys

# keep in sync with labext/src/branding.ts: the guard greps the bundle for these
PRODUCT = "LABZ Docker Studio"
SHORT_LABEL = "Built by the LABZ team"
SPLASH_LABEL = "Loading LABZ Docker"
COPYRIGHT = "Copyright 2026-Present the LABZ team"
AGPL_NOTICE = "Licensed under Apache 2.0 and the GNU AGPLv3"
WEBSITE_URL = "https://github.com/dustinwloring1988/labz"
DOCS_URL = "https://unsloth.ai/docs"
SOURCE_URL = "https://github.com/dustinwloring1988/labz"
LICENSE_URL = "https://github.com/dustinwloring1988/labz#license"
AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.html"
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
# ONE literal, byte-identical to PHRASE in labext/src/branding.ts
PHRASE = (
    "LABZ Docker Studio and JupyterLab image. Built by the LABZ team. "
    "Licensed under Apache 2.0 and the GNU AGPLv3. "
    "Source: https://github.com/dustinwloring1988/labz "
    "Website: https://github.com/dustinwloring1988/labz"
)

THEME_NAME = "LABZ Dark"
LABEXT_NAME = "labz-jupyterlab"
ABOUT_PLUGIN_ID = "labz-jupyterlab:about"
SPLASH_PLUGIN_ID = "labz-jupyterlab:splash"
LOGO_DATA_URI_PREFIX = "data:image/png;base64,iVBOR"


def resolve_paths(
    venv_share = None,
    jupyter_server_dir = None,
    config_dirs = None,
):
    """Installed locations of every checked branding asset; tests pass explicit roots."""
    if venv_share is None:
        venv_share = os.path.join(sys.prefix, "share", "jupyter")
    if jupyter_server_dir is None:
        import jupyter_server
        jupyter_server_dir = os.path.dirname(jupyter_server.__file__)
    labext_dir = os.path.join(venv_share, "labextensions", LABEXT_NAME)

    if config_dirs is None:
        try:
            from jupyter_core.paths import jupyter_config_path
            config_dirs = jupyter_config_path()
        except Exception:
            config_dirs = []
    page_configs = [os.path.join(venv_share, "lab", "settings", "page_config.json")]
    page_configs += [os.path.join(d, "labconfig", "page_config.json") for d in config_dirs]

    return {
        "license": os.path.join(venv_share, "UNSLOTH_LICENSE.AGPL-3.0"),
        "login": os.path.join(jupyter_server_dir, "templates", "login.html"),
        "overrides": os.path.join(venv_share, "lab", "settings", "overrides.json"),
        "labext_dir": labext_dir,
        "labext_pkg": os.path.join(labext_dir, "package.json"),
        "labext_static": os.path.join(labext_dir, "static"),
        "favicon": os.path.join(jupyter_server_dir, "static", "favicons", "favicon.ico"),
        "logo": os.path.join(jupyter_server_dir, "static", "logo", "logo.png"),
        "page_configs": page_configs,
    }


def _read(path):
    try:
        with open(path, encoding = "utf-8", errors = "replace") as f:
            return f.read()
    except OSError:
        return None


def _nonempty_file(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _bundle_text(static_dir):
    """Every built .js chunk: the production build minifies only identifiers, so the
    attribution literals survive verbatim in one of them."""
    if not os.path.isdir(static_dir):
        return ""
    parts = []
    for name in sorted(os.listdir(static_dir)):
        if name.endswith(".js"):
            text = _read(os.path.join(static_dir, name))
            if text:
                parts.append(text)
    return "\n".join(parts)


def verify_branding(paths = None):
    if paths is None:
        paths = resolve_paths()
    problems = []

    license_text = _read(paths["license"])
    if license_text is None:
        problems.append("missing AGPLv3 license file: " + paths["license"])
    elif "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        problems.append("AGPLv3 license file is not the GNU AGPL v3 text: " + paths["license"])

    login = _read(paths["login"])
    if login is None:
        problems.append("missing branded login page: " + paths["login"])
    else:
        for marker in (SHORT_LABEL, COPYRIGHT, SOURCE_URL, "AGPLv3"):
            if marker not in login:
                problems.append("login page missing attribution marker: " + marker)

    overrides = _read(paths["overrides"])
    if not overrides or THEME_NAME not in overrides:
        problems.append("overrides.json missing the '" + THEME_NAME + "' theme")

    pkg = _read(paths["labext_pkg"])
    if pkg is None:
        problems.append("missing labextension: " + paths["labext_pkg"])
    else:
        try:
            if json.loads(pkg).get("name") != LABEXT_NAME:
                problems.append("labextension package.json name is not " + LABEXT_NAME)
        except ValueError:
            problems.append("labextension package.json is not valid JSON")

    bundle = _bundle_text(paths["labext_static"])
    if not bundle:
        problems.append("missing built labextension bundle: " + paths["labext_static"])
    else:
        for marker in (
            PHRASE,
            SHORT_LABEL,
            COPYRIGHT,
            AGPL_URL,
            ABOUT_PLUGIN_ID,
            SPLASH_PLUGIN_ID,
            LOGO_DATA_URI_PREFIX,
        ):
            if marker not in bundle:
                problems.append("labextension bundle missing: " + marker)

    if not _nonempty_file(paths["favicon"]):
        problems.append("missing or empty favicon: " + paths["favicon"])
    if not _nonempty_file(paths["logo"]):
        problems.append("missing or empty logo: " + paths["logo"])

    # disabledExtensions leaves the bundle on disk but strips it at load
    for pc_path in paths.get("page_configs", []):
        text = _read(pc_path)
        if not text:
            continue
        try:
            disabled = json.loads(text).get("disabledExtensions", {})
        except ValueError:
            problems.append("page_config.json is not valid JSON: " + pc_path)
            continue
        if isinstance(disabled, dict):
            disabled_ids = [k for k, v in disabled.items() if v]
        elif isinstance(disabled, (list, tuple)):
            disabled_ids = list(disabled)
        else:
            disabled_ids = []
        for ident in disabled_ids:
            if not isinstance(ident, str):
                continue
            if ident == LABEXT_NAME or ident.startswith(LABEXT_NAME + ":"):
                problems.append(
                    "page_config.json disables LABZ branding '" + ident + "': " + pc_path
                )

    return problems


def banner(problems):
    lines = [
        "",
        "=" * 72,
        "ERROR: LABZ Docker Studio branding / license integrity check failed.",
        "",
        "This image is a derivative of the Unsloth Docker Studio and JupyterLab",
        "image and ships under the GNU AGPLv3. It will not start because required",
        "branding or license assets are missing or altered:",
        "",
    ]
    for p in problems:
        lines.append("  - " + p)
    lines += [
        "",
        SHORT_LABEL + ".  " + COPYRIGHT + ".",
        "Website: " + WEBSITE_URL,
        "Source:  " + SOURCE_URL,
        "License: GNU AGPLv3 (" + AGPL_URL + ")",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def _jupyter_server_extension_points():
    return [{"module": "labz_branding"}]


def _load_jupyter_server_extension(serverapp):
    problems = verify_branding()
    if not problems:
        return
    msg = banner(problems)
    print(msg, file = sys.stderr, flush = True)
    try:
        serverapp.log.critical(msg)
    except Exception:
        pass
    # studio_launch.sh refuses the container first; this backstops a direct run
    try:
        serverapp.exit(1)
    except Exception:
        pass
    raise SystemExit(1)


def main(argv = None):
    import argparse

    parser = argparse.ArgumentParser(description = "LABZ branding / license integrity check")
    parser.add_argument("--verify", action = "store_true", help = "verify and exit nonzero on failure")
    parser.add_argument("--venv-share", default = None)
    parser.add_argument("--jupyter-server-dir", default = None)
    args = parser.parse_args(argv)

    paths = resolve_paths(args.venv_share, args.jupyter_server_dir)
    problems = verify_branding(paths)
    if problems:
        print(banner(problems), file = sys.stderr, flush = True)
        return 1
    print("LABZ branding / license integrity check passed (" + PRODUCT + ", AGPLv3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
