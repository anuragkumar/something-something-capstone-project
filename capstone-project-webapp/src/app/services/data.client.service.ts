import {Injectable} from "@angular/core";
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {AppSettings} from "./app-settings.client";

const httpOptions = {
    headers: new HttpHeaders({'Content-Type': 'application/json'})
};

@Injectable()
export class DataClientService {
    url = AppSettings.API_ENDPOINT;

    constructor(private httpClient: HttpClient) {
    }

    getVideoList = () => {
        return fetch(this.url + '/api/videos').then(res => res.json());
    }

    getPrediction = (video, classes) => {
        return fetch(this.url + '/api/predict/' + video + '/' + classes).then(res => res.json());
    }

    getObjectDetectionImages = (videoid) => {
        return fetch(this.url + '/api/objectDetection/' + videoid).then(res => res.json());
    }

    uploadVideo = (file) => {
        const formData: FormData = new FormData();
        formData.append('file', file, file.name);
        return this.httpClient.post(this.url + '/api/upload', formData, httpOptions);
    }
}
