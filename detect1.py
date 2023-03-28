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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, plot_confusion_matrix, recall_score, precision_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.datasets import mnist
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score, train_test_split
import joblib
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import backend as K
"""Note
Stretched dataset:
Total number of boxes: 34
Total of file : 23
Key 'image_13_stretched.jpg' has 1 boxes
Key 'image_17_stretched.jpg' has 1 boxes
Key 'image_21_stretched.jpg' has 1 boxes
Key 'image_22_stretched.jpg' has 1 boxes
Key 'image_27_stretched.jpg' has 1 boxes
Key 'image_29_stretched.jpg' has 2 boxes
Key 'image_36_stretched.jpg' has 2 boxes
Key 'image_43_stretched.jpg' has 1 boxes
Key 'image_45_stretched.jpg' has 1 boxes
Key 'image_46_stretched.jpg' has 1 boxes
Key 'image_52_stretched.jpg' has 2 boxes
Key 'image_56_stretched.jpg' has 2 boxes
Key 'image_75_stretched.jpg' has 2 boxes
Key 'image_77_stretched.jpg' has 1 boxes
Key 'image_7_stretched.jpg' has 1 boxes
Key 'image_84_stretched.jpg' has 2 boxes
Key 'image_89_stretched.jpg' has 2 boxes
Key 'image_90_stretched.jpg' has 3 boxes
Key 'image_91_stretched.jpg' has 2 boxes
Key 'image_92_stretched.jpg' has 1 boxes
Key 'image_93_stretched.jpg' has 2 boxes
Key 'image_94_stretched.jpg' has 1 boxes
Key 'image_9_stretched.jpg' has 1 boxes

"""
folder = 'output-stretched.csv'
df = pd.read_csv(folder , sep=',')
print(df)

# a dictionary where the keys are the image filenames
# and the values are lists of bounding box tuples
def bounding_boxes():
    boxes_dict = {}
    for index, row in df.iterrows():
        filename = row['filename']
        class_name = row['class']
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        box = (xmin,ymin, xmax, ymax)
        if filename in boxes_dict:
            boxes_dict[filename].append(box)
        else:
            boxes_dict[filename] =[box]
    return boxes_dict
boxes = bounding_boxes()
#number of files
print(len(boxes))#23
print(type(boxes))

num_boxes = 0

for key, value in boxes.items():
    if key in boxes:
        # Count the number of values (bounding boxes) for this key
        num_boxes_key = len(value)

        # Increment the total number of boxes counter
        num_boxes += num_boxes_key

        # Print the number of boxes for this key
        print("Key '{}' has {} boxes".format(key, num_boxes_key))
    else:
        print("Key '{}' does not exist in the dictionary".format(key))


print("Total number of boxes: {}".format(num_boxes))

def cnn_axlexnet(dimension):
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
    return x

def rpn_layer(base_layers, num_anchors):
    x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1,1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1,1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr]




def roi_pooling(conv_feature_map, rois, pool_size):
    """
    ROI pooling implementation using TensorFlow's crop_and_resize function.
    Args:
        conv_feature_map: The convolutional feature map of shape (batch_size, height, width, channels).
        rois: The regions of interest, as a tensor of shape (num_rois, 4) where each row represents [x_min, y_min, x_max, y_max].
        pool_size: The output size of the pooled region of interest, as a tuple of two integers (pool_height, pool_width).
    Returns:
        The ROI pooled features, as a tensor of shape (num_rois, pool_height, pool_width, channels).
    """
    num_rois = tf.shape(rois)[0]
    batch_size = tf.shape(conv_feature_map)[0]
    num_channels = tf.shape(conv_feature_map)[3]

    # Initialize output tensor
    pooled_features = tf.TensorArray(dtype=tf.float32, size=num_rois, dynamic_size=False)

    for i in range(num_rois):
        # Get the coordinates of the region of interest
        x_min, y_min, x_max, y_max = tf.unstack(rois[i])

        # Compute the height and width of the ROI
        roi_height = y_max - y_min
        roi_width = x_max - x_min

        # Compute the height and width of each pooling bin
        bin_size_h = tf.cast(tf.math.ceil(roi_height / pool_size[0]), dtype=tf.int32)
        bin_size_w = tf.cast(tf.math.ceil(roi_width / pool_size[1]), dtype=tf.int32)

        # Compute the row and column indices of the pooling bins
        rows = tf.range(y_min, y_max, delta=(roi_height / pool_size[0]))
        cols = tf.range(x_min, x_max, delta=(roi_width / pool_size[1]))

        # Convert the row and column indices to the format expected by crop_and_resize
        boxes = tf.transpose(tf.stack([rows, cols, rows + bin_size_h, cols + bin_size_w]))
        boxes = boxes / tf.constant([K.cast(tf.shape(conv_feature_map)[1], tf.float32),
                                      K.cast(tf.shape(conv_feature_map)[2], tf.float32),
                                      K.cast(tf.shape(conv_feature_map)[1], tf.float32),
                                      K.cast(tf.shape(conv_feature_map)[2], tf.float32)])

        # Crop and resize the feature map using the region of interest
        pooled = tf.image.crop_and_resize(conv_feature_map, boxes, tf.range(num_rois), pool_size)

        # Append the pooled feature map to the output tensor
        pooled_features = pooled_features.write(i, pooled)

    # Convert the tensor array to a tensor
    pooled_features = pooled_features.stack()

    # Reshape the output tensor to (num_rois, pool_height, pool_width, num_channels)
    pooled_features = tf.reshape(pooled_features, (num_rois, pool_size[0], pool_size[1], num_channels))

    return pooled_features


class RoiPoolingConv(layers):
    def __init__(self, pool_size, num_rois, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.num_rois = num_rois

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)

        # x[0] is the feature map with shape (batch_size, height, width, nb_channels)
        # x[1] is the ROI tensor with shape (batch_size, num_rois, 4)
        feature_map = x[0]
        rois = x[1]

        input_shape = K.shape(feature_map)
        outputs = []

        for roi_idx in range(self.num_rois):
            # Get the coordinates and dimensions of the ROI
            x_start, y_start, x_end, y_end = rois[0, roi_idx, :]

            # Compute the size of the ROI
            roi_width = x_end - x_start
            roi_height = y_end - y_start

            # Compute the size of a pooling bin in the ROI
            bin_size_h = roi_height / float(self.pool_size)
            bin_size_w = roi_width / float(self.pool_size)

            # Compute the indices of the pooling bins in the feature map
            bin_idxs_h = tf.range(self.pool_size, dtype=tf.float32) * bin_size_h
            bin_idxs_w = tf.range(self.pool_size, dtype=tf.float32) * bin_size_w

            # Add the coordinates of the ROI to the bin indices
            x_idxs = bin_idxs_w + x_start
            y_idxs = bin_idxs_h + y_start

            # Convert the bin indices to integers
            x_idxs = K.cast(K.round(x_idxs), dtype=tf.int32)
            y_idxs = K.cast(K.round(y_idxs), dtype=tf.int32)

            # Crop the feature map to the ROI and pool the values in each bin
            roi = tf.image.crop_and_resize(feature_map, boxes=tf.constant([[0, y_start, x_start, y_end, x_end]]),
                                            box_indices=tf.zeros((1,), dtype=tf.int32),
                                            crop_size=(self.pool_size, self.pool_size))
            outputs.append(roi)

        # Concatenate the ROI pooled features and return the result
        pooled_features = K.concatenate(outputs, axis=0)
        pooled_features = K.reshape(pooled_features, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return pooled_features


"""
import numpy as np
import pandas as pd

# read CSV file with bounding box information
df = pd.read_csv('bounding_boxes.csv')

# define base anchor box size and ratios
base_size = 16  # you can adjust this to match your needs
ratios = [0.5, 1, 2]

# compute anchor box widths and heights for each ratio
scales = [base_size * r for r in ratios]
widths = np.sqrt(scales)
heights = widths / ratios

# initialize list to store anchor boxes
anchor_boxes = []

# loop over bounding box information in the CSV file
for i in range(len(df)):
    # compute width and height of bounding box
    bbox_width = df['xmax'][i] - df['xmin'][i]
    bbox_height = df['ymax'][i] - df['ymin'][i]
    
    # compute center coordinates of bounding box
    bbox_center_x = df['xmin'][i] + 0.5 * bbox_width
    bbox_center_y = df['ymin'][i] + 0.5 * bbox_height
    
    # loop over anchor box sizes and ratios to generate anchor boxes
    for j in range(len(scales)):
        for k in range(len(ratios)):
            # compute anchor box width and height
            anchor_width = widths[j]
            anchor_height = heights[j]
            
            # compute anchor box center coordinates and convert to corner coordinates
            anchor_center_x = bbox_center_x + anchor_width * (2*k - 1) / 4
            anchor_center_y = bbox_center_y + anchor_height * (2*k - 1) / 4
            anchor_xmin = anchor_center_x - 0.5 * anchor_width
            anchor_ymin = anchor_center_y - 0.5 * anchor_height
            anchor_xmax = anchor_center_x + 0.5 * anchor_width
            anchor_ymax = anchor_center_y + 0.5 * anchor_height
            
            # append anchor box to list
            anchor_boxes.append([anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax])
            
# convert anchor box list to numpy array
anchor_boxes = np.array(anchor_boxes)


import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the CSV file
folder = 'output-stretched.csv'
with open(folder, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Group the bounding boxes by filename
boxes_dict = {}
for row in rows:
    filename = row['filename']
    class_name = row['class']
    xmin = int(row['xmin'])
    ymin = int(row['ymin'])
    xmax = int(row['xmax'])
    ymax = int(row['ymax'])
    box = (xmin, ymin, xmax, ymax)
    if filename in boxes_dict:
        boxes_dict[filename].append((class_name, box))
    else:
        boxes_dict[filename] = [(class_name, box)]

# Plot the bounding boxes for each image
for filename, boxes in boxes_dict.items():
    # Load the image
    image = plt.imread(filename)

    # Create a new figure
    fig, ax = plt.subplots(1)

    # Show the image
    ax.imshow(image)

    # Add the bounding boxes
    for class_name, box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], class_name, color='r')

    # Save the figure
    plt.savefig(filename + '.png')
"""
