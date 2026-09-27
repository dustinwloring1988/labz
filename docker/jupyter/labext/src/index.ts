// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the LABZ team. See /studio/LICENSE.AGPL-3.0

import {
  ILabShell,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { IThemeManager } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { LABZ_LOGO_DATA_URI } from './logo';
import aboutPlugin from './about';
import cellNavPlugin from './cellNav';
import colabTitlePlugin from './colabTitle';
import outputSelectPlugin from './outputSelect';
import splashPlugin from './splash';
import uiChromePlugin from './uiChrome';

/** A NAMED theme, so Settings > Theme and the adaptive light/dark switch see it. */
const themePlugin: JupyterFrontEndPlugin<void> = {
  id: 'labz-jupyterlab:theme',
  description: 'LABZ Dark (Monokai) theme.',
  autoStart: true,
  requires: [IThemeManager],
  activate: (app: JupyterFrontEnd, manager: IThemeManager): void => {
    const style = 'labz-jupyterlab/index.css';
    manager.register({
      name: 'LABZ Dark',
      isLight: false,
      themeScrollbars: true,
      load: () => manager.loadCSS(style),
      unload: () => Promise.resolve(undefined)
    });
  }
};

/**
 * The stock logo plugin is disabled + locked at build, so this is the only logo
 * widget. An <img> with inline styles, not a LabIcon, so it shows in any theme.
 */
const logoPlugin: JupyterFrontEndPlugin<void> = {
  id: 'labz-jupyterlab:logo',
  description: 'Replace the top-left Jupyter logo with the LABZ logo.',
  autoStart: true,
  requires: [ILabShell],
  activate: (app: JupyterFrontEnd, shell: ILabShell): void => {
    const logo = new Widget();
    const img = document.createElement('img');
    img.src = LABZ_LOGO_DATA_URI;
    img.alt = 'LABZ';
    img.style.height = '24px';
    img.style.width = 'auto';
    img.style.margin = '1px 6px 1px 8px';
    img.style.display = 'block';
    logo.node.appendChild(img);
    logo.node.style.display = 'flex';
    logo.node.style.alignItems = 'center';
    logo.id = 'jp-MainLogo';
    shell.add(logo, 'top', { rank: 0 });
  }
};

export default [
  themePlugin,
  cellNavPlugin,
  logoPlugin,
  colabTitlePlugin,
  outputSelectPlugin,
  uiChromePlugin,
  aboutPlugin,
  splashPlugin
];
