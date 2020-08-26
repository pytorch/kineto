import {NgModule} from '@angular/core';
import {StoreModule} from '@ngrx/store';

import {rootReducer} from './reducers';
import {STORE_KEY} from './state';

@NgModule({
  imports: [
    StoreModule.forFeature(STORE_KEY, rootReducer),
    StoreModule.forRoot({}),
  ],
})
export class RootStoreModule {
}
