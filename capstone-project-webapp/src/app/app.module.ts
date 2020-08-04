import { BrowserAnimationsModule } from "@angular/platform-browser/animations";
import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ToastrModule } from "ngx-toastr";

import { SidebarModule } from './sidebar/sidebar.module';
import { FooterModule } from './shared/footer/footer.module';
import { NavbarModule} from './shared/navbar/navbar.module';
import { FixedPluginModule} from './shared/fixedplugin/fixedplugin.module';

import { AppComponent } from './app.component';
import { AppRoutes } from './app.routing';

import { AdminLayoutComponent } from './layouts/admin-layout/admin-layout.component';
import {BrowserModule} from "@angular/platform-browser";
import {VgCoreModule} from "videogular2/compiled/src/core/core";
import {VgControlsModule} from "videogular2/compiled/src/controls/controls";
import {VgOverlayPlayModule} from "videogular2/compiled/src/overlay-play/overlay-play";
import {VgBufferingModule} from "videogular2/compiled/src/buffering/buffering";
import {AppSettings} from "./services/app-settings.client";
import {DataClientService} from "./services/data.client.service";
import {HttpClient, HttpClientModule} from "@angular/common/http";
import {FormsModule} from "@angular/forms";


@NgModule({
  declarations: [
    AppComponent,
    AdminLayoutComponent
  ],
  imports: [
    HttpClientModule,
    FormsModule,
    BrowserAnimationsModule,
    RouterModule.forRoot(AppRoutes,{
      useHash: true
    }),
    SidebarModule,
    NavbarModule,
    ToastrModule.forRoot(),
    FooterModule,
    FixedPluginModule,
    BrowserModule,
    VgCoreModule,
    VgControlsModule,
    VgOverlayPlayModule,
    VgBufferingModule
  ],
  providers: [
      AppSettings,
      DataClientService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
