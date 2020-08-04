import os
import cv2
import torch
import torchvision
import numpy as np

from config import config
from videoloader import VideoLoader
from pytorch_implementation_new_way import MultiColumn, Model
from pytorch_implementation_new_way import remove_module_from_checkpoint_state_dict


class DataModel:
    def __init__(self):
        self.videoLoader = None
        self.item = None

        self.checkpoint_path = os.path.join(config["output_dir"], config["model_name"], "model_best.pth.tar")
        self.model = MultiColumn(config['num_classes'], Model, int(config["column_units"]))
        self.model.eval()
        self._load_model_checkpoint()

        self.dict_two_way = None
        self.input_data = None
        self.target = None
        self.item_id = None

    def _load_videoLoader(self, videoPath):
        self.videoLoader = VideoLoader(videoPath)
        self.item = self.videoLoader.video_item
        self.dict_two_way = self.videoLoader.classes_dict
        self.input_data, self.target, self.item_id = self.item

    def _load_model_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
            checkpoint['state_dict'])
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, videoPath):
        self._load_videoLoader(videoPath)
        self.input_data = self.input_data.unsqueeze(0)
        input_var = [torch.autograd.Variable(self.input_data)]
        output = self.model(input_var).squeeze(0)
        output = torch.nn.functional.softmax(output, dim=0)
        # compute top5 predictions
        pred_prob, pred_top5 = output.data.topk(4)
        pred_prob = pred_prob.numpy()
        pred_top5 = pred_top5.numpy()
        pred_list = []
        for i, pred in enumerate(pred_top5):
            temp_dict = {
                "top": i + 1,
                "label": self.dict_two_way[pred],
                "probability": "{:.2f}%".format(pred_prob * 100)
            }
            pred_list.append(temp_dict)

        result = {
            "video_id": self.item_id,
            "true_label": self.dict_two_way[self.target],
            "predictions": pred_list
        }
        return result
