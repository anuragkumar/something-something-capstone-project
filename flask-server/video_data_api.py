from flask import Blueprint
from flask import jsonify
import os
from config2 import config
import pandas as pd

video_api = Blueprint('video_api', __name__)

video_dir = "./data/videos"
substring_path = "/data/videos/"


def get_video_list():
    """
    get all the videos to construct the urls
    """
    video_data_list = []

    val_data = pd.read_json(config['json_data_val'], orient="records")
    vids = os.listdir(video_dir)
    videos = list(map(lambda x: substring_path + x, vids))
    for vid_id, vid_fullpath in zip(vids, videos):
        temp = []
        temp.append(vid_fullpath)
        temp.append(vid_id)
        selected_row = val_data.loc[val_data['id'] == int(vid_id.split(".")[0])]
        temp.append(selected_row["template"].values[0])
        video_data_list.append(temp)
    return video_data_list


@video_api.route("/videos")
def get_videos():
    return jsonify(get_video_list())

# @data_api.route("/datarecords")
# def get_data():
#     return data_obj.get_random_sample().to_json(orient='records')
#
#
# @data_api.route("/predictedDataRecords")
# def get_predicted_data():
#     return data_obj.get_predicted_random_sample().to_json(orient='records')


