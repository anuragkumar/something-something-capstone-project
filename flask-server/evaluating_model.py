#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import sys
import importlib
import torch
import torchvision
import numpy as np
import av

sys.path.insert(0, "../")


# In[2]:


from pytorch_implementation_new_way import WebMDataset
from pytorch_implementation_new_way import MultiColumn, Model
from pytorch_implementation_new_way import ComposeMix, Scale, Augmentor
from pytorch_implementation_new_way import remove_module_from_checkpoint_state_dict


# In[3]:


config = {
    "model_name": "3D_model_4_classes",
    "output_dir": "D:/capstone-project-2-webapp/flask-server/data/trained_models/",

    "input_mode": "av",

    "data_folder": "D:/capstone-project-2-webapp/flask-server/data/videos/",

    "json_data_val": "D:/capstone-project-2-webapp/flask-server/data/files/validation_data_9_classes.json",

    "json_file_labels": "D:/capstone-project-2-webapp/flask-server/data/files/labels.json",

    "num_workers": 5,

    "num_classes": 4,
    "batch_size": 30,
    "clip_size": 72,
    
    "nclips_train": 1,
    "nclips_val": 1,

    "upscale_factor_train": 1.4,
    "upscale_factor_eval": 1.0,

    "step_size_train": 1,
    "step_size_val": 1,

    "lr": 0.008,
    "last_lr": 0.00001,
    "momentum": 0.9,
    "weight_decay": 0.00001,
    "num_epochs": -1,
    "print_freq": 100,

    "input_spatial_size": 84,

    "column_units": 512,
    "save_features": True
}


config2 = {
  "model_name": "3D_model_9_classes",
  "output_dir": "D:/capstone-project-2-webapp/flask-server/data/trained_models/",

  "input_mode": "av",

  "data_folder": "D:/capstone-project-2-webapp/flask-server/data/videos/",

  "json_data_val": "D:/capstone-project-2-webapp/flask-server/data/files/validation_data_9_classes.json",

  "json_file_labels": "D:/capstone-project-2-webapp/flask-server/data/files/labels_9_classes.json",

  "num_workers": 5,

  "num_classes": 9,
  "batch_size": 30,
  "clip_size": 72,

  "nclips_train": 1,
  "nclips_val": 1,

  "upscale_factor_train": 1.4,
  "upscale_factor_eval": 1.0,

  "step_size_train": 1,
  "step_size_val": 1,

  "lr": 0.008,
  "last_lr": 0.00001,
  "momentum": 0.9,
  "weight_decay": 0.00001,
  "num_epochs": -1,
  "print_freq": 100,

  "input_spatial_size": 84,

  "column_units": 512,
  "save_features": True
}


class VideoFolder(torch.utils.data.Dataset):
    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 get_item_id=False, is_test=False):
        self.dataset_object = WebMDataset(json_file_input, json_file_labels,
                                          root, is_test=is_test)
        self.dataset_object = WebMDataset(json_file_input, json_file_labels,
                                          root, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.get_item_id = get_item_id

    def get_video_id_data(self, video_id):
        for data in self.json_data:
            if str(video_id) == data.id:
                return data

    def __getitem__(self, video_id):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        item = self.get_video_id_data(video_id)

        # Open video file
        reader = av.open(item.path)

        try:
            imgs = []
            imgs = [f.to_rgb().to_nd_array() for f in reader.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))

        imgs = self.transform_pre(imgs)
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
        if self.get_item_id:
            return (data, target_idx, item.id)
        else:
            return (data, target_idx)

    def __len__(self):
        return len(self.json_data)


def predict(vid_id, classes):
    # set column model
    # column_cnn_def = importlib.import_module("{}".format(config['conv_model']))
    config_file = None
    if classes == "four":
        config_file = config
    elif classes == "nine":
        config_file = config2

    model_name = config_file["model_name"]

    print("=> Name of the model -- {}".format(model_name))

    # checkpoint path to a trained model
    checkpoint_path = os.path.join("../", config_file["output_dir"], config_file["model_name"], "model_best.pth.tar")
    print("=> Checkpoint path --> {}".format(checkpoint_path))

    # In[5]:

    model = MultiColumn(config_file['num_classes'], Model, int(config_file["column_units"]))
    model.eval();

    # In[6]:

    print("=> loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
        checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_path, checkpoint['epoch']))

    # In[7]:

    import json
    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
        [Scale(config_file['input_spatial_size']), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(config_file['input_spatial_size']), "img"]
    ])

    transform_post = ComposeMix([
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # default values for imagenet
            std=[0.229, 0.224, 0.225]), "img"]
    ])

    val_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config_file['json_data_val'],
                           json_file_labels=config_file['json_file_labels'],
                           clip_size=config_file['clip_size'],
                           nclips=config_file['nclips_val'],
                           step_size=config_file['step_size_val'],
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           get_item_id=True,
                           )
    dict_two_way = val_data.classes_dict

    # In[8]:

    len(val_data)

    # In[9]:

    dict_two_way

    # In[10]:

    type(val_data)

    # In[11]:

    selected_indx = np.random.randint(len(val_data))
    selected_indx = 48
    print(selected_indx)

    input_data, target, item_id = val_data[vid_id]

    input_data = input_data.unsqueeze(0)
    print("Id of the video sample = {}".format(item_id))
    print("True label --> {} ({})".format(target, dict_two_way[target]))

    # In[13]:

    if config_file['nclips_val'] > 1:
        input_var = list(input_data.split(config_file['clip_size'], 2))
        for idx, inp in enumerate(input_var):
            input_var[idx] = torch.autograd.Variable(inp)
    else:
        input_var = [torch.autograd.Variable(input_data)]

    # In[14]:

    output = model(input_var).squeeze(0)
    output = torch.nn.functional.softmax(output, dim=0)
    print(output)

    # In[37]:

    # compute top5 predictions
    pred_prob, pred_top5 = output.data.topk(4)
    pred_prob = pred_prob.numpy()
    pred_top5 = pred_top5.numpy()

    # In[38]:

    print(pred_prob)
    print(pred_top5)

    # In[39]:

    print("Id of the video sample = {}".format(item_id))
    print("True label --> {} ({})".format(target, dict_two_way[target]))
    print("\nTop-4 Predictions:")
    for i, pred in enumerate(pred_top5):
        print("Top {} :== {}. Prob := {:.2f}%".format(i + 1, dict_two_way[pred], pred_prob[i] * 100))

    pred_list = []
    for i, pred in enumerate(pred_top5):
        temp_dict = {
            "top": i + 1,
            "label": dict_two_way[pred],
            "probability": "{:.2f}%".format(pred_prob[i] * 100)
        }
        pred_list.append(temp_dict)

    result = {
        "video_id": item_id,
        "true_label": dict_two_way[target],
        "predictions": pred_list,
        "location": "/data/videos/" + item_id + ".webm"
    }
    return result

