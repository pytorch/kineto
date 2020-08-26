import {HttpClientModule} from '@angular/common/http';
import {NgModule} from '@angular/core';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {BrowserModule} from '@angular/platform-browser';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {EmptyPageModule} from 'org_xprof/frontend/app/components/empty_page/empty_page_module';
import {MainPageModule} from 'org_xprof/frontend/app/components/main_page/main_page_module';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {RootStoreModule} from 'org_xprof/frontend/app/store/store_module';

import {App} from './app';

/** The root component module. */
@NgModule({
  declarations: [App],
  imports: [
    BrowserModule,
    HttpClientModule,
    MatProgressBarModule,
    EmptyPageModule,
    MainPageModule,
    BrowserAnimationsModule,
    PipesModule,
    RootStoreModule,
  ],
  providers: [DataService],
  bootstrap: [App]
})
export class AppModule {
}
