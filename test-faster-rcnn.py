import numpy as np
from keras_cv.models import FasterRCNN
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.applications import ResNet50
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend import epsilon
from matplotlib.pyplot import plt
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D

from sklearn.model_selection import cross_val_score, train_test_split
import joblib

from keras import backend as K

# Define the bounding box format
bounding_box_format = 'pascal_voc'

# Preprocess the data to create the necessary inputs for the model
def extract_data(data):
    image_filenames = data['filename'].unique()
    images = []
    boxes = []
    labels = []
    for filename in image_filenames:
        image_data = plt.imread(filename) # Load the image using Matplotlib
        images.append(image_data)
        image_boxes = data[data['filename'] == filename][['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes.append(image_boxes)
        image_labels = data[data['filename'] == filename]['class'].values
        labels.append(image_labels)

    # Convert the inputs to the appropriate format
    images = np.array(images)
    boxes = [np.array(image_boxes) for image_boxes in boxes]
    labels = [np.array(image_labels) for image_labels in labels]
    return images , boxes , labels

def define_backbone_alexnet(dimension):
    input = Input(shape = dimension)

    x = Conv2D(filters=96, kernel_size=11, strides=4, name='conv1', activation='relu')(input)
    x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(2)(x)

    # second layer
    x = Conv2D(filters=256, kernel_size=3, strides=1, name="conv2", activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(1)(x)

    # third layer
    x = Conv2D(filters=384, kernel_size=3, strides=1, name='conv3', activation='relu')(x)
    x = ZeroPadding2D(1)(x)

    # fourth layer
    x = Conv2D(filters=384, kernel_size=3, strides=1, name='conv4' , activation = 'relu')(x)
    x = ZeroPadding2D(1)(x)

    #fifth layer
    x = Conv2D(filters= 256 , kernel_size= 3, strides=1 , name = 'conv5', activation = 'relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool3")
    backbone = Model(inputs=input, outputs=x)
    return backbone

# Create the model


if __name__ =="__main__":
    while True:
            print('1.Backbone alexnet')
            choice = int(input('Choose choice: '))
            if choice == 1:
                folder = '.csv'
                data = pd.read_csv(folder , sep=',')
                classes = ['node']
                images, boxes, labels = extract_data(data= data)
                # Train the model
                model = FasterRCNN(classes=labels, bounding_box_format=bounding_box_format)
                model.fit(images, boxes, labels)
                model.save('faster-rcnn.h5')
            if choice == 2:
                pass
