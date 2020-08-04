import shutil
import pandas as pd


source = "D:\\something-something-project\\data\\videos\\20bn-something-something-v2\\"
destination = "D:\\capstone-project-2-webapp\\flask-server\\data\\videos\\"


df = pd.read_json("D:\\capstone-project-2-webapp\\flask-server\\data\\files\\validation_data_9_classes.json", orient="records")


for index, row in df.iterrows():
    try:
        print("copying file-->{}, {}".format(row["template"], row["id"]))
        shutil.copyfile(source + str(row["id"]) + ".webm", destination + str(row["id"]) + ".webm")
    except Exception as e:
        print("Unable to copy file")
        print(e)

