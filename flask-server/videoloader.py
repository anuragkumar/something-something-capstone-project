import torch
import av
import numpy as np
import torchvision
from video import Video
from config import config
from pytorch_implementation_new_way import ComposeMix, Scale, Augmentor


class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, videoPath):
        self.video = Video(videoPath)
        self.object = self.video.object
        self.classes = self.video.classes
        self.classes_dict = self.video.classes_dict

        self.transform_eval_pre = ComposeMix([
            [Scale(84), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(84), "img"]
        ])
        self.transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
        ])

        self.augmentor = Augmentor(None, None)

        self.clip_size = config["clip_size"]
        self.nclips = config["clip_size"]
        self.step_size = config["clip_size"]
        self.is_val = True
        self.video_item = self.getitem()

    def getitem(self):
        print(self.object)
        item = self.object
        reader = av.open(item.path)

        try:
            imgs = []
            imgs = [f.to_rgb().to_nd_array() for f in reader.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as exception:
            return '{}: WEBM reader cannot open {}. Empty list returned.'.format(type(exception).__name__, item.path)

        imgs = self.transform_eval_pre(imgs)
        imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)

        num_frames = len(imgs)
        target_idx = self.classes_dict[label]

        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]

        if len(imgs) < (self.clip_size * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.clip_size * self.nclips) - len(imgs)))

        # format data to torch
        data = torch.stack(imgs)
        data = data.permute(1, 0, 2, 3)
        return data, target_idx, item.id

