from flask import Blueprint
from flask import render_template, send_from_directory
from flask import jsonify
from flask import request, Response
from model import DataModel
import evaluating_model
from object_detection import ObjectDetection
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tkinter import *
import matplotlib
from PIL import Image
from flask import send_file
import os

matplotlib.use('TkAgg')
# get_ipython().run_line_magic('matplotlib', 'inline')

predict_api = Blueprint('predict_api', __name__)

objectDetection = ObjectDetection()


@predict_api.route("/predict/<videoid>/<classes>")
def predict(videoid, classes):
    return jsonify(evaluating_model.predict(videoid, classes))


@predict_api.route("/objectDetection/<videoid>")
def object_detection(videoid):
    print("object detection started")
    images = objectDetection.get_object_detected_images(videoid)
    print("object detection completed")
    for idx, img in enumerate(images):
        im = Image.fromarray(img)
        if not os.path.exists("./data/images/" + str(videoid)):
            os.mkdir("./data/images/" + str(videoid))
        im.save("./data/images/" + videoid + "/" + videoid + "_" + str(idx) + ".jpg")
    print("images saved")
    image_dir = os.listdir("./data/images/" + videoid)
    image_dir = list(map(lambda x: "/data/images/" + videoid + "/" + x, image_dir))
    print("images retrieved")
    return jsonify(image_dir)
    # print(image_dir)
    # return render_template("image_template.html", results=image_dir)
    # fig = Figure()
    # plt.imsave()
    # plt.imshow(array_to_img(images[0]))
    # output = io.BytesIO()
    # FigureCanvasAgg(fig).print_png(output)
    # return send_file("./data/images/" + videoid + ".jpg", mimetype="image/png")

# @eda_api.route("/predict")
# def eda():
#     return "list of accounts"
#
#
# # @eda_api.route("/scatterHTML")
# # def send_scatter_html():
# #     return render_template('scattertext_benefits.html', password=u"You should know, right?")
#
#
# @eda_api.route("/industry", methods=['POST'])
# def get_industry_bar_plot_data():
#     req_data = request.get_json()
#     req = req_data.get('plotName')
#     flag = req_data.get('boolean')
#     return jsonify(data_obj.get_plot_data(req, flag))
#
#
# @eda_api.route("/origPlot", methods=['POST'])
# def get_orig_bar_plot_data():
#     req_data = request.get_json()
#     req = req_data.get('plotName')
#     flag = req_data.get('boolean')
#     return jsonify(data_obj.get_orig_plot_data(req, flag))


