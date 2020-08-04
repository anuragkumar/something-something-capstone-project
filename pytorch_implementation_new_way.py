#!/usr/bin/env python
# coding: utf-8

# In[87]:


config = {
    "model_name": "3D_model",
    "output_dir": "trained_models",
    
    "input_mode": "av",
    
    "data_folder": "D:/something-something-project/data/videos/20bn-something-something-v2/",
    "json_data_train": "D:/something-something-project/data/train_data.json",
    "json_data_val": "D:/something-something-project/data/validation_data.json",
    "json_data_test": "D:/something-something-project/data/something-something-v2-test.json",
    
    "json_data_labels": "D:/something-something-project/data/something-something-v2-mylabels.json",
    
    "num_workers": 8,                      # for parallel processing
    
    "num_classes": 4,                    # number of classes to classify
    "batch_size": 30,
    "clip_size": 30,
    
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
    "num_epochs": 30,
    "print_freq": 1,
    
    "conv_model": "models.model3d_1",
    "input_spatial_size": 84,
    
    "column_units": 512,
    "save_features": True
}


# In[29]:


# Load and read json data and construct a list containing video sample
# (name, id, label, path)

import os
import json
from collections import namedtuple


# In[30]:


ListData = namedtuple('ListData', ['id', 'label', 'path'])

# In[31]:


# defining a class to read json labels from <...>labels.json file provided in the dataset
class BaseDataset:
    "Read json data and construct a list containing video sample ids, label and path"
    def __init__(self, json_input_path, json_path_labels, data_root, extension, is_test=False):
        self.json_input_path = json_input_path
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test
        
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()
    
    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)
    
    def clean_template(self, template):
        """Replaces instances of '[something] --> 'something' """
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template
    
    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict
    
    def read_json_input(self):
        json_data = []
        if not self.is_test:
            with open(self.json_input_path, 'rb') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    label = self.clean_template(elem['template'])
                    if label not in self.classes:
                        continue
                        raise ValueError("Label mismatch! Please correct")
                    item = ListData(elem['id'], label, os.path.join(self.data_root + elem['id'] + self.extension))
                    json_data.append(item)
        else:
            with open(self.json_input_path, 'rb') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data


# In[32]:


# defining class specific to webm video format and inherit the base class
class WebMDataset(BaseDataset):
    def __init__(self, json_input_path, json_path_labels, data_root, is_test=False):
        EXTENSION = ".webm"
        super().__init__(json_input_path, json_path_labels, data_root, EXTENSION, is_test)


# In[33]:


# testing the read json class declared above and seeing how the data looks like
# webmdataset = WebMDataset("D:/something-something-project/data/something-something-v2-train.json",
#                        "D:/something-something-project/data/something-something-v2-labels.json",
#                        "D:/something-something-project/data/videos/20bn-something-something-v2/")
# webmdataset.display_data_members()


# In[34]:


# As you can see, we have created a named tuple (name, id, location) format for all the train data
# So, we just need to pass the train/validation/test.json file and we will get this list of named tuples for further processing


# In[35]:


import cv2
import torch
import numpy as np
import numbers
import collections
import random


# In[56]:


# transforming video

# we will define a class which composes several transformations together
class ComposeMix:
    """
    Composes several transformations together. It takes a list of transformations,
    where each element of transform is a list with two elemts.
    First being the transformation function itself, second being a string indicating "img" or "vid" transform
    
    Args:
        transforms (List[Transform, "<type>"]): list of transforms to compose. <type> = "img" | "vid"
        
    Example:
        >>> transforms.ComposeMix([
                                    [RandomCropVideo(84), "vid"],
                                    [torchvision.transforms.ToTensor(), "img"],
                                    [torchvision.transforms.Normalize(
                                                                        mean=[0.485, 0.456, 0.406], # default values for imagenet
                                                                        std=[0.229, 0.224, 0.225]
                                                                    ), "img"]
                                ])
    As you can see, we first randomly crop a video for 84x84 pixels, 
    then convert the cropped image into tensor and finally normalize the image using default values for imagenet.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    """
    What is __call__() method?
    class Example: 
        def __init__(self): 
            print("Instance Created") 
      
        # Defining __call__ method 
        def __call__(self): 
            print("Instance is called via special method") 
  
    # Instance created 
    e = Example() 
  
    # __call__ method will be called 
    e()
    
    Output:
    Instance Created
    Instance is called via special method
    """
    def __call__(self, imgs):
        for t in self.transforms:
            if t[1] == "img":
                for idx, img in enumerate(imgs):
                    imgs[idx] = t[0](img)
            elif t[1] == "vid":
                imgs = t[0](imgs)
            else:
                print ("Please specify the transform type")
                raise ValueError
        return imgs

class RandomCropVideo:
    """
    Crop the given video frames at a random location. Crop location is the same for all the frames.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an int instead of a sequence like
            (w, h), a square crop (size, size) is made
        padding: (cv2 constant): Method to be used for padding
    """
    
    def __init__(self, size, padding=0, pad_method=cv2.BORDER_CONSTANT):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        self.padding = padding
        self.pad_method = pad_method
    
    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Video to be cropped
            Returns:
                numpy.array: Cropped Video
        """
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        # Return random integers from low (inclusive) to high (exclusive)
        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         self.pad_method)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1 + th, x1:x1 + tw]
            imgs[idx] = img_crop
        return imgs

class RandomHorizontalFlipVideo:
    """Horizontally flip the given video frames randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            for idx, img in enumerate(imgs):
                imgs[idx] = cv2.flip(img, 1)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomReverseTimeVideo(object):
    """Reverse the given video frames in time randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            imgs = imgs[::-1]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotationVideo(object):
    """Rotate the given video frames randomly with a given degree.
    Args:
        degree (float): degrees used to rotate the video
    """

    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be rotated.
        Returns:
            numpy.array: Randomly rotated video.
        """
        h, w = imgs[0].shape[:2]
        degree_sampled = np.random.choice(
                            np.arange(-self.degree, self.degree, 0.5))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree_sampled, 1)

        for idx, img in enumerate(imgs):
            imgs[idx] = cv2.warpAffine(img, M, (w, h))

        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(degree={})'.format(self.degree_sampled)


class IdentityTransform(object):
    """
    Returns same video back
    """
    def __init__(self,):
        pass

    def __call__(self, imgs):
        return imgs


class Scale(object):
    r"""Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be scaled.
        Returns:
            numpy.array: Rescaled image.
        """
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if ow < w:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if oh < h:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
        else:
            return cv2.resize(img, tuple(self.size))


class UnNormalize(object):
    """Unnormalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel x std) + mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean).astype('float32')
        self.std = np.array(std).astype('float32')

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if isinstance(tensor, torch.Tensor):
            self.mean = torch.FloatTensor(self.mean)
            self.std = torch.FloatTensor(self.std)

            if (self.std.dim() != tensor.dim() or
                    self.mean.dim() != tensor.dim()):
                for i in range(tensor.dim() - self.std.dim()):
                    self.std = self.std.unsqueeze(-1)
                    self.mean = self.mean.unsqueeze(-1)

            tensor = torch.add(torch.mul(tensor, self.std), self.mean)
        else:
            # Relying on Numpy broadcasting abilities
            tensor = tensor * self.std + self.mean
        return tensor


# In[57]:


# data augmentor
class Augmentor:
    def __init__(self, augmentation_mappings_json=None,
                augmentation_types_todo=None,
                fps_jitter_factors=[1, 0.75, 0.5]):
        self.augmentation_mappings_json = augmentation_mappings_json
        self.augmentation_types_todo = augmentation_types_todo
        self.fps_jitter_factors = fps_jitter_factors

        # read json to get the mapping dict
        self.augmentation_mapping = self.read_augmentation_mapping(
                                        self.augmentation_mappings_json)
        self.augmentation_transforms = self.define_augmentation_transforms()
        
    def __call__(self, imgs, label):
        if not self.augmentation_mapping:
            return imgs, label
        else:
            candidate_augmentations = {"same": label}
            for candidate in self.augmentation_types_todo:
                if candidate == "jitter_fps":
                    continue
                if label in self.augmentation_mapping[candidate]:
                    if isinstance(self.augmentation_mapping[candidate], list):
                        candidate_augmentations[candidate] = label
                    elif isinstance(self.augmentation_mapping[candidate], dict):
                        candidate_augmentations[candidate] = self.augmentation_mapping[candidate][label]
                    else:
                        print("Something wrong with data type specified in "
                              "augmentation file. Please check!")
            augmentation_chosen = np.random.choice(list(candidate_augmentations.keys()))
            imgs = self.augmentation_transforms[augmentation_chosen](imgs)
            label = candidate_augmentations[augmentation_chosen]

            return imgs, label
        
    def read_augmentation_mapping(self, path):
        if path:
            with open(path, "rb") as fp:
                mapping = json.load(fp)
        else:
            mapping = None
        return mapping

    def define_augmentation_transforms(self, ):
        augmentation_transforms = {}
        augmentation_transforms["same"] = IdentityTransform()
        augmentation_transforms["left/right"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["left/right agnostic"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["reverse time"] = RandomReverseTimeVideo(1)
        augmentation_transforms["reverse time agnostic"] = RandomReverseTimeVideo(0.5)

        return augmentation_transforms

    def jitter_fps(self, framerate):
        if self.augmentation_types_todo and "jitter_fps" in self.augmentation_types_todo:
            jitter_factor = np.random.choice(self.fps_jitter_factors)
            return int(jitter_factor * framerate)
        else:
            return framerate


# In[58]:


import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np


# In[59]:


# utility functions

def load_args():
    parser = argparse.ArgumentParser(description='Smth-Smth example training')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true', 
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--gpus', '-g', help="GPU ids to use. Please"
                         " enter a comma separated list")
    parser.add_argument('--use_cuda', action='store_true',
                        help="to use GPUs")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return args


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config


def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def save_results(logits_matrix, features_matrix, targets_list, item_id_list,
                 class_to_idx, config):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx], f)


def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))
    from matplotlib import pylab as plt
    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))


def get_submission(logits_matrix, item_id_list, class_to_idx, config):
    top5_classes_pred_list = []

    for i, id in enumerate(item_id_list):
        logits_sample = logits_matrix[i]
        logits_sample_top5  = logits_sample.argsort()[-5:][::-1]
        # top1_class_index = logits_sample.argmax()
        # top1_class_label = class_to_idx[top1_class_index]

        top5_classes_pred_list.append(logits_sample_top5)

    path_to_save = os.path.join(
            config['output_dir'], config['model_name'], "test_submission.csv")
    with open(path_to_save, 'w') as fw:
        for id, top5_pred in zip(item_id_list, top5_classes_pred_list):
            fw.write("{}".format(id))
            for elem in top5_pred:
                fw.write(";{}".format(elem))
            fw.write("\n")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExperimentalRunCleaner(object):
    """
    Remove the output dir, if you exit with Ctrl+C and if there are less
    then 1 file. It prevents the noise of experimental runs.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, signal, frame):
        num_files = len(glob.glob(self.save_dir + "/*"))
        if num_files < 1:
            print('Removing: {}'.format(self.save_dir))
            shutil.rmtree(self.save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)


# In[60]:


# video parser

# PyAV is for direct and precise access to your media via containers, streams, packets, codecs, and frames.
# It exposes a few transformations of that data, and helps you get your data to/from other packages (e.g. Numpy and Pillow).
import av
import torch
import numpy as np
import torchvision


# In[61]:


class VideoFolder(torch.utils.data.Dataset):
    
    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                nclips, step_size, is_val, transform_pre=None, transform_post=None,
                augmentation_mappings_json=None, augmentation_types_todo=None,
                get_item_id=False, is_test=False):
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
    
    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """

        item = self.json_data[index]

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


# In[62]:


# testing VideoFolder class
"""
upscale_size = int(84 * 1.1)
transform_pre = ComposeMix([
                    [Scale(upscale_size), "img"],
                    [RandomCropVideo(84), "vid"]
                ])

transform_post = ComposeMix([
                        [torchvision.transforms.ToTensor(), "img"]
                    ])

loader = VideoFolder(root = "D:/something-something-project/data/videos/20bn-something-something-v2/",
                    json_file_input = "D:/something-something-project/data/something-something-v2-train.json",
                    json_file_labels = "D:/something-something-project/data/something-something-v2-labels.json",
                    clip_size = 36,
                    nclips = 1,
                    step_size = 1,
                    is_val = False,
                    transform_pre = transform_pre,
                    transform_post = transform_post)

import time
from tqdm import tqdm

# change the number of workers to 8 or something since jupyter notebook has some issues. Using 0 works fine
batch_loader = torch.utils.data.DataLoader(loader, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)

start = time.time()

for i, a in enumerate(tqdm(batch_loader)):
    if i > 100:
        break
    pass

print ("Size --> {}".format(a[0].size()))
print (time.time() - start)
"""

# In[63]:


import sys
import time
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from torch.optim.optimizer import Optimizer


###############################################################################
# TRAINING CALLBACKS
###############################################################################

class PlotLearning(object):
    def __init__(self, save_path, num_classes):
        self.accuracy = []
        self.val_accuracy = []
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.save_path_loss = os.path.join(save_path, 'loss_plot.png')
        self.save_path_accu = os.path.join(save_path, 'accu_plot.png')
        self.save_path_lr = os.path.join(save_path, 'lr_plot.png')
        self.init_loss = -np.log(1.0 / num_classes)

    def plot(self, logs):
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.learning_rates.append(logs.get('learning_rate'))

        best_val_acc = max(self.val_accuracy)
        best_train_acc = max(self.accuracy)
        best_val_epoch = self.val_accuracy.index(best_val_acc)
        best_train_epoch = self.accuracy.index(best_train_acc)

        plt.figure(1)
        plt.gca().cla()
        plt.ylim(0, 1)
        plt.plot(self.accuracy, label='train')
        plt.plot(self.val_accuracy, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accu)

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.ylim(0, self.init_loss)
        plt.plot(self.losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.savefig(self.save_path_loss)

        min_learning_rate = min(self.learning_rates)
        max_learning_rate = max(self.learning_rates)

        plt.figure(2)
        plt.gca().cla()
        plt.ylim(0, max_learning_rate)
        plt.plot(self.learning_rates)
        plt.title("max_learning_rate-{0:.6f}, min_learning_rate-{1:.6f}".format(max_learning_rate, min_learning_rate))
        plt.savefig(self.save_path_lr)


# Taken from keras.keras.utils.generic_utils
class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[64]:


# definig a model

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    - A 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1`, and is
      averaged at the end
    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, column_units):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # get convolution column features

        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        # print(x.size())

        # averaging features in time dimension
        x = x.mean(-1).mean(-1).mean(-1)

        return x


# In[65]:


# testing model

# num_classes = 174
# input_tensor = torch.autograd.Variable(torch.rand(5, 3, 72, 84, 84))
# model = Model(512).cuda()

# output = model(input_tensor.cuda())
# print(output.size())


# In[66]:


# multicolumn

import torch.nn as nn
import torch as th


class MultiColumn(nn.Module):

    def __init__(self, num_classes, conv_column, column_units,
                 clf_layers=None):
        """
        - Example multi-column network
        - Useful when a video sample is too long and has to be split into
          multiple clips
        - Processes 3D-CNN on each clip and averages resulting features across
          clips before passing it to classification(FC) layer
        Args:
        - Input: Takes in a list of tensors each of size
                 (batch_size, 3, sequence_length, W, H)
        - Returns: logits of size (batch size, num_classes)
        """
        super(MultiColumn, self).__init__()
        self.num_classes = num_classes
        self.column_units = column_units
        self.conv_column = conv_column(column_units)
        self.clf_layers = clf_layers

        if not self.clf_layers:
            self.clf_layers = th.nn.Sequential(
                                 nn.Linear(column_units, self.num_classes)
                                )

    def forward(self, inputs, get_features=False):
        outputs = []
        num_cols = len(inputs)
        for idx in range(num_cols):
            x = inputs[idx]
            x1 = self.conv_column(x)
            outputs.append(x1)

        outputs = th.stack(outputs).permute(1, 0, 2)
        outputs = th.squeeze(th.sum(outputs, 1), 1)
        avg_output = outputs / float(num_cols)
        outputs = self.clf_layers(avg_output)
        if get_features:
            return outputs, avg_output
        else:
            return outputs


# In[68]:


# num_classes = 174
# input_tensor = [th.autograd.Variable(th.rand(1, 3, 72, 84, 84))]
# model = MultiColumn(174, Model, 512)
# output = model(input_tensor)
# print(output.size())


# In[69]:


# training model

import os
import sys
import time
import signal
import importlib

import torch
import torch.nn as nn
import numpy as np
import torchvision


# In[72]:


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


# In[104]:


args = DictX ({
    "use_cuda": True,
    "gpus": "0",
    "resume": False,
    "eval_only": False,
    "start_epoch": 0
})


# In[105]:


config = config # defined in the first cell
file_name = config['conv_model']
# cnn_def = importlib.import_module("{}".format(file_name))


# def setup_cuda_devices(args):
#     device_ids = []
#     device = torch.device("cuda" if args.use_cuda else "cpu")
#     if device.type == "cuda":
#         device_ids = [int(i) for i in args.gpus.split(',')]
#     return device, device_ids


# setup cuda device - CPU or GPU
device, device_ids = setup_cuda_devices(args)

print(" > Using device: {}".format(device.type))
print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')

# dont need this as of now

# if config["input_mode"] == "av":
#     from data_loader_av import VideoFolder
# elif config["input_mode"] == "skvideo":
#     from data_loader_skvideo import VideoFolder
# else:
#     raise ValueError("Please provide a valid input mode")


# In[106]:


def main():
    global args, best_loss

    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    save_dir = os.path.join(output_dir, model_name)
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

    # create model
    print(" > Creating model ... !")
    model = MultiColumn(config['num_classes'], Model,
                        int(config["column_units"]))

    # multi GPU setting
    model = torch.nn.DataParallel(model, device_ids).to(device)

    # optionally resume from a checkpoint
    checkpoint_path = os.path.join(config['output_dir'],
                                   config['model_name'],
                                   'model_best.pth.tar')
    if args.resume:
        if os.path.isfile(checkpoint_path):
            print(" > Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))

    # define augmentation pipeline
    upscale_size_train = int(config['input_spatial_size'] * config["upscale_factor_train"])
    upscale_size_eval = int(config['input_spatial_size'] * config["upscale_factor_eval"])

    # Random crop videos during training
    transform_train_pre = ComposeMix([
            [RandomRotationVideo(15), "vid"],
            [Scale(upscale_size_train), "img"],
            [RandomCropVideo(config['input_spatial_size']), "vid"],
             ])

    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
            [Scale(upscale_size_eval), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"],
             ])

    # Transforms common to train and eval sets and applied after "pre" transforms
    transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                       mean=[0.485, 0.456, 0.406],  # default values for imagenet
                       std=[0.229, 0.224, 0.225]), "img"]
             ])

    train_data = VideoFolder(root=config['data_folder'],
                             json_file_input=config['json_data_train'],
                             json_file_labels=config['json_data_labels'],
                             clip_size=config['clip_size'],
                             nclips=config['nclips_train'],
                             step_size=config['step_size_train'],
                             is_val=False,
                             transform_pre=transform_train_pre,
                             transform_post=transform_post,
                             augmentation_mappings_json=None,
                             augmentation_types_todo=None,
                             get_item_id=False,
                             )

    print(" > Using {} processes for data loader.".format(
        config["num_workers"]))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config['json_data_val'],
                           json_file_labels=config['json_data_labels'],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_val'],
                           step_size=config['step_size_val'],
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           get_item_id=True,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    test_data = VideoFolder(root=config['data_folder'],
                            json_file_input=config['json_data_test'],
                            json_file_labels=config['json_data_labels'],
                            clip_size=config['clip_size'],
                            nclips=config['nclips_val'],
                            step_size=config['step_size_val'],
                            is_val=True,
                            transform_pre=transform_eval_pre,
                            transform_post=transform_post,
                            get_item_id=True,
                            is_test=True,
                            )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    print(" > Number of dataset classes : {}".format(len(train_data.classes)))
    assert len(train_data.classes) == config["num_classes"]

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = config["lr"]
    last_lr = config["last_lr"]
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if args.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        print(" > Evaluation DONE !")
        return

    # set callbacks
    plotter = PlotLearning(os.path.join(
        save_dir, "plots"), config["num_classes"])
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=0.5, patience=2, verbose=True)
    val_loss = float('Inf')

    # set end condition by num epochs
    num_epochs = int(config["num_epochs"])
    if num_epochs == -1:
        num_epochs = 999999

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epochs))
    start_epoch = args.start_epoch if args.resume else 0

    for epoch in range(start_epoch, num_epochs):

        lrs = [params['lr'] for params in optimizer.param_groups]
        print(" > Current LR(s) -- {}".format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(" > Training is DONE by learning rate {}".format(last_lr))
            sys.exit(1)

        # train for one epoch
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)

        # set learning rate
        lr_decayer.step(val_loss, epoch)

        # plot learning
        plotter_dict = {}
        plotter_dict['loss'] = train_loss
        plotter_dict['val_loss'] = val_loss
        plotter_dict['acc'] = train_top1 / 100
        plotter_dict['val_acc'] = val_top1 / 100
        plotter_dict['learning_rate'] = lr
        plotter.plot(plotter_dict)

        print(" > Validation loss after epoch {} = {}".format(epoch, val_loss))

        # remember best loss and save the checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Conv4Col",
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, config)


# In[107]:


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if config['nclips_train'] > 1:
            input_var = list(input.split(config['clip_size'], 2))
            for idx, inp in enumerate(input_var):
                input_var[idx] = inp.to(device)
        else:
            input_var = [input.to(device)]

        target = target.to(device)

        model.zero_grad()

        # compute output and loss
        output = model(input_var)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, item_id) in enumerate(val_loader):

            if config['nclips_val'] > 1:
                input_var = list(input.split(config['clip_size'], 2))
                for idx, inp in enumerate(input_var):
                    input_var[idx] = inp.to(device)
            else:
                input_var = [input.to(device)]

            target = target.to(device)

            # compute output and loss
            output, features = model(input_var, config['save_features'])
            loss = criterion(output, target)

            if args.eval_only:
                logits_matrix.append(output.cpu().data.numpy())
                features_matrix.append(features.cpu().data.numpy())
                targets_list.append(target.cpu().numpy())
                item_id_list.append(item_id)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["print_freq"] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if args.eval_only:
        logits_matrix = np.concatenate(logits_matrix)
        features_matrix = np.concatenate(features_matrix)
        targets_list = np.concatenate(targets_list)
        item_id_list = np.concatenate(item_id_list)
        print(logits_matrix.shape, targets_list.shape, item_id_list.shape)
        save_results(logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx, config)
        get_submission(logits_matrix, item_id_list, class_to_idx, config)
    return losses.avg, top1.avg, top5.avg


# In[108]:


if __name__ == '__main__':
    main()

