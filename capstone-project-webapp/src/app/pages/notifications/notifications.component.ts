import {Component, OnInit} from '@angular/core';
import { ToastrService } from "ngx-toastr";
import {FormGroup, FormControl, Validators} from '@angular/forms';
import {DataClientService} from '../../services/data.client.service';


@Component({
    selector: 'notifications-cmp',
    moduleId: module.id,
    templateUrl: 'notifications.component.html'
})

export class NotificationsComponent implements OnInit{

    myForm = new FormGroup({
        name: new FormControl('', [Validators.required, Validators.minLength(10000)]),
        file: new FormControl('', [Validators.required]),
        fileSource: new FormControl('', [Validators.required])
    });
    fileToUpload: File = null;
    label: '';

  constructor(private dataService: DataClientService) {}

    ngOnInit() {
    }

    get f() {
      return this.myForm.controls;
    }

    onFileChange = (file) => {
      this.fileToUpload = file.item(0);
        // if (event.target.files.length > 0) {
        //     console.log('something is present')
        //     const file = event.target.files[0];
        //     this.myForm.patchValue({
        //         fileSource: file
        //     });
        // }
    }

    submit() {
      // const formData = new FormData();
      // formData.append('file', this.myForm.get('fileSource').value);
      // // formData.append('label', this.label);
      // console.log(formData);

      this.dataService.uploadVideo(this.fileToUpload).subscribe(res => {
          console.log(res);
      }, error => {
          console.log(error);
      });
    }

}
