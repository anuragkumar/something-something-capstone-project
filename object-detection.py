#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


# In[2]:


def selective_search(image, method="fast"):
    # initialize OpenCV's selective search implementation and set the input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    
    # check to see if we are using "fast" but "less accurate" version of selective search
    if method == "fast":
        ss.switchToSelectiveSearchFast()
        
    # otherwise we are using "slower" but "more accurate" version
    else:
        ss.switchToSelectiveSearchQuality()
        
    # run selective search on the input image
    rects = ss.process()
    
    # return the region proposal bounding boxes
    return rects


# In[3]:


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


# In[4]:


args = DictX ({
    "image": "./stingray.jpg", # location of image
    "method": "fast",          # ["fast", "slow"]
    "conf": "0.9",             # minimum probabilty to consider a classification/detection
    "filter": None             # comma separated list of ImageNet labels to filter on
})


# In[5]:


# grab label filters
labelFilters = args.filter

# if the label filter is not empty, break it into a list
if labelFilters is not None:
    labelFilters = labelFilters.lower().split(",")


# In[6]:


# load ResNet (with weights pre-trained on ImageNet)
print ("Loading ResNet...")
model = ResNet50(weights="imagenet")

# load the input image from disk and grab its dimensions
image = cv2.imread(args.image)
(H, W) = image.shape[:2]


# In[7]:


print (type(image))


# In[8]:


print ("performing selective search with '{}' method".format(args.method))
rects = selective_search(image, method=args.method)
print ("{} regions found by selective search".format(len(rects)))

# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []


# In[9]:


# loop over the region proposal bounding box coordinates generated by
# running selective search
for (x, y, w, h) in rects:
    # if the width or height of the region is less than 10% of the
    # image width or height, ignore it (i.e., filter out small
    # objects that are likely false-positives)
    if w/float(W) < 0.1 or h/float(H) < 0.1:
        continue
        
    # extract the region from the input image, convert it from BGR to
    # RGB channel ordering, and them resize it to 224x224 (the input
    # dimesnsion required by our pretrained CNN)
    roi = image[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    
    # further processing by the ROI
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    
    # update our proposals and bounding boxes lists
    proposals.append(roi)
    boxes.append((x, y, w, h))


# In[10]:


# convert the proposals list into numpy array and show its dimensions
proposals = np.array(proposals)
print ("Proposal shape: {}".format(proposals.shape))

# classify each of the proposal ROIs using ResNet and then decode the
# predictions
print ("Classifying proposals...")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

# initialize a dictionary which maps class labels (keys) to any
# bounding box associated with that label (values)
labels = {}


# In[12]:


# loop over the predictions
for (i, p) in enumerate(preds):
    # grab the prediction information for the current region proposal
    (imagenetID, label, prob) = p[0]
    
    # only if the label filters are not empty "and" the label does not
    # exist in the list, then ignore it
    if labelFilters is not None and label not in labelFilters:
        continue
    
    # filter out weak detections by ensuring the predicted probability
    # is greater than the minimum probability
    if prob >= float(args.conf):
        # grab the bounding box associated with the prediction and
        # convert the coordinates
        (x, y, w, h) = boxes[i]
        box = (x, y, x + w, y + h)
        
        # grab the list of predictions for the label and add the
        # bounding box + probability to the list
        L = labels.get(labels, [])
        L.append((box, prob))
        labels[labels] = L


# In[13]:


# loop over the labels for each of detected objects in the image
for label in labels.keys():
    # clone the original image so that we can draw on it
    print("[INFO] showing results for '{}'".format(label))
    clone = image.copy()
    # loop over all bounding boxes for the current label
    for (box, prob) in labels[label]:
        # draw the bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # show the results *before* applying non-maxima suppression, then
    # clone the image again so we can display the results *after*
    # applying non-maxima suppression
    cv2.imshow("Before", clone)
    clone = image.copy()
    
    # extract the bounding boxes and associated prediction
    # probabilities, then apply non-maxima suppression
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    # loop over all bounding boxes that were kept after applying
    # non-maxima suppression
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box and label on the image
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # show the output after apply non-maxima suppression
    cv2.imshow("After", clone)
    cv2.waitKey(0)


# In[ ]:




