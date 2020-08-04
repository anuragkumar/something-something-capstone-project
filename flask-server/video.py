import os
from collections import namedtuple
import json
from config import config


class Video:
    def __init__(self, video_id):
        self.ListData = namedtuple('ListData', ['id', 'label', 'path'])
        self.object = self.read_video_file(video_id)
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def read_json_labels(self):
        classes = []
        with open(config["json_file_labels"], 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def clean_template(self, template):
        """Replaces instances of '[something] --> 'something' """
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template

    def read_json_input(self, video_id):
        with open(config["json_data_val"], 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                if elem["id"] == video_id:
                    label = self.clean_template(elem["template"])
                    item = self.ListData(elem["id"], label, os.path.join(config["data_folder"] + elem["id"]) + ".webm")
                    return item

    def read_video_file(self, video_id):
        return self.read_json_input(video_id)

