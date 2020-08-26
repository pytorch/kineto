import {enableProdMode} from '@angular/core';
import {platformBrowser} from '@angular/platform-browser';

import {AppModuleNgFactory} from 'org_xprof/frontend/app/app_module.ngfactory';

enableProdMode();

platformBrowser().bootstrapModuleFactory(AppModuleNgFactory);
