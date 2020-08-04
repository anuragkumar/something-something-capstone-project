import { Component, OnInit } from '@angular/core';
import {AppSettings} from '../../services/app-settings.client';
import {DataClientService} from '../../services/data.client.service';
import {ActivatedRoute} from '@angular/router';

@Component({
    selector: 'user-cmp',
    moduleId: module.id,
    templateUrl: 'user.component.html'
})

export class UserComponent implements OnInit{
    vid = '';
    result: {
        location: '1234'
    };
    url = AppSettings.API_ENDPOINT;
    images: [];
    videoURL = '';
    constructor(private dataService: DataClientService, private route: ActivatedRoute) {}

    ngOnInit() {
        this.route.params.subscribe(params => {
            console.log(params.vid);
            this.videoURL = this.url + '/data/videos/' + params.vid;
            console.log(this.videoURL);
            this.vid = params.vid.split('.')[0];
            console.log(this.vid);
        });

        this.dataService.getPrediction(this.vid, 'nine').then(res => {
            console.log(res);
            this.result = res;
            // this.url.concat(res.location);
        });

        this.dataService.getObjectDetectionImages(this.vid).then(res => {
            console.log(res);
            this.images = res;
        })
    }
}
