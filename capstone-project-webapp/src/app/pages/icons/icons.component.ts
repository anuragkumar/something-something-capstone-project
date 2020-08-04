import { Component } from '@angular/core';
import {AppSettings} from '../../services/app-settings.client';
import {DataClientService} from '../../services/data.client.service';
import {Router} from '@angular/router';

@Component({
    selector: 'icons-cmp',
    moduleId: module.id,
    templateUrl: 'icons.component.html'
})

export class IconsComponent{
    videoList: [];
    url = AppSettings.API_ENDPOINT;

    constructor(private dataService: DataClientService, private router: Router) {
    }

    ngOnInit() {
        this.dataService.getVideoList().then(res => {
            console.log(res);
            this.videoList = res;
        });
    }

    _predictVideo = (vid, classes) => {
        console.info(vid);
        if (classes === 'four') {
            this.router.navigate(['/maps', vid]);
        } else if (classes === 'nine') {
            this.router.navigate(['/user', vid]);
        }
    }
}
