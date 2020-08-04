from flask import Flask, send_from_directory, request
from video_data_api import video_api
from predict_api import predict_api
# from eda import eda_api
# from data_api import data_api
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "./data/videos/"

app.register_blueprint(video_api, url_prefix='/api')
app.register_blueprint(predict_api, url_prefix='/api')


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/data/<path:filepath>')
def data(filepath):
    return send_from_directory('data', filepath)


@app.route('/templates/<path:filepath>')
def template_files(filepath):
    return send_from_directory('templates', filepath)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    print("uploading file")
    if request.method == 'POST':
        print("inside post")
        f = request.files['file']
        try:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            print("file saved successfully")
        except Exception as e:
            print("failed to save file", e)
    return "file uploaded successfully"


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
