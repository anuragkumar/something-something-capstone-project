import { Component,OnInit } from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {DataClientService} from '../../services/data.client.service';
import {AppSettings} from '../../services/app-settings.client';

declare var google: any;

@Component({
    moduleId: module.id,
    selector: 'maps-cmp',
    templateUrl: 'maps.component.html'
})

export class MapsComponent implements OnInit {

    vid = '';
    result: {
        location: '1234'
    };
    url = AppSettings.API_ENDPOINT;
    images: [];
    videoURL = '';

    constructor(private route: ActivatedRoute, private dataService: DataClientService) { }

    ngOnInit() {
        this.route.params.subscribe(params => {
            console.log(params.vid);
            this.videoURL = this.url + '/data/videos/' + params.vid;
            console.log(this.videoURL);
            this.vid = params.vid.split('.')[0];
            console.log(this.vid);
        });

        this.dataService.getPrediction(this.vid, 'four').then(res => {
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
