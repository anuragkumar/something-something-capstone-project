import pandas as pd
import numpy as np
import matplotlib as plt
import os
import cv2
import multiprocessing


# directory where all the data is present
base_dir = "D:\\something-something-project\\data\\"

# loading only training and validation json files
train_vids_id = pd.read_json(base_dir + 'something-something-v2-train.json')
validation_vids_id = pd.read_json(base_dir + 'something-something-v2-validation.json')

# grouping the data based on following classes, grouping together all the data which belongs to one action
# we will be using below 9 classes:
classes = {'Dropping [something]' : [], 
           'Holding [something]' : [],
           'Moving [something]' : [], 
           'Picking [something]' : [],
           'Poking [something]' : [],
           'Pouring [something]' : [],
           'Putting [something]' : [],
           'Showing [something]' : [], 
           'Tearing [something]' : []}

validation_classes = {'Dropping [something]' : [], 
           'Holding [something]' : [],
           'Moving [something]' : [], 
           'Picking [something]' : [],
           'Poking [something]' : [],
           'Pouring [something]' : [],
           'Putting [something]' : [],
           'Showing [something]' : [], 
           'Tearing [something]' : []}


actual_class = {'Dropping [something]' : 'Dropping_something', 
           'Holding [something]' : 'Holding_something',
           'Moving [something]' : 'Moving_something', 
           'Picking [something]' : 'Picking_something',
           'Poking [something]' : 'Poking_something',
           'Pouring [something]' : 'Pouring_something',
           'Putting [something]' : 'Putting_something',
           'Showing [something]' : 'Showing_something', 
           'Tearing [something]' : 'Tearing_something'}


# utility function to return the class name from the template name
def get_class_name(template_name):
    for key in classes:
        if key in template_name:
            return key
    return None

# grouping the training data
# iterating the pd dataframe
for index, row in train_vids_id.iterrows():
    row_class = get_class_name(row['template'])
    if (row_class != None):
        # print ('Template: ', row['template'], '\tclass: ', row_class)
        classes[row_class].append((row['id'], row['placeholders'][0]))

# grouping the validation data
# iterating the pd dataframe
for index, row in validation_vids_id.iterrows():
    row_class = get_class_name(row['template'])
    if (row_class != None):
        # print ('Template: ', row['template'], '\tclass: ', row_class)
        validation_classes[row_class].append((row['id'], row['placeholders'][0]))


# directory where all the videos are saved
video_dir = 'D:\\something-something-project\\data\\videos\\20bn-something-something-v2\\'
# directory where images will be saved
output_dir_train = 'D:\\something-something-project\\train-images\\'
output_dir_validation = 'D:\\something-something-project\\validation-images\\'

count = 1

# function to extract frames and save it in respective class folder
def getFrame(sec, vidcap, output_dir, class_name, obj, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        if not os.path.exists(output_dir + actual_class[class_name]): 
            os.makedirs(output_dir + actual_class[class_name])
        
        os.chdir(output_dir + actual_class[class_name])
        cv2.imwrite(actual_class[class_name].split('_')[0] + '_' + obj + '_' + str(count) + ".jpg", image)     # save frame as JPG file
    return hasFrames

def generate_images(classes_dict, class_name, input_dir, output_dir, pname):
    global count
    count = 1
    print ("Current Process: ", pname)
    break_flag = False
    for vid in classes_dict[class_name]:
        vidcap = cv2.VideoCapture(input_dir + str(vid[0]) + '.webm')
        sec = 0
        fps = 1
        success = getFrame(sec, vidcap, output_dir, class_name, vid[1], count)
        while success:
            count = count + 1
            # limiting the number of images (frames) per class to 5000
#             if (count == 5000):
#                 break_flag = True
#                 break
            sec = sec + fps
            sec = round(sec, 2)
            success = getFrame(sec, vidcap, output_dir, class_name, vid[1], count)
#         if break_flag:
#             break
    print ("Process Finished: ", pname)

if __name__ == '__main__':
    processes=[]

    for key in classes:
        processes.append(multiprocessing.Process(target=generate_images, args=(classes, key, video_dir, output_dir_train, "training_imgs_" + key,)))

    for key in validation_classes:
        processes.append(multiprocessing.Process(target=generate_images, args=(validation_classes, key, video_dir, output_dir_validation, "validation_imgs_" + key,)))

    for t in processes:
        t.start()

    for t in processes:
        t.join()












