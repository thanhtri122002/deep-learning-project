import cv2
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
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
#from keras_retinanet.utils.metrics import mean_average_precision
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

from keras import backend as K

# Define the bounding box format
bounding_box_format = 'pascal_voc'

# Preprocess the data to create the necessary inputs for the model


"""call iou in main
if choice == 1 or choice == 2:
    # Load test data
    test_data = pd.read_csv('test.csv', sep=',')
    test_images, test_boxes, test_labels = extract_data(test_data)

    # Test the model
    predictions = model.predict(test_images)

    # Calculate IOU
    iou_scores = calculate_iou(test_boxes, predictions)

    # Print IOU statistics
    print("IOU statistics:")
    print("Mean IOU:", np.mean(iou_scores))
    print("Median IOU:", np.median(iou_scores))
    print("Min IOU:", np.min(iou_scores))
    print("Max IOU:", np.max(iou_scores))

    # Visualize IOU histogram
    plt.hist(iou_scores, bins=20)
    plt.xlabel("IOU")
    plt.ylabel("Frequency")
    plt.show()

"""

def extract_data(data, validation_split):
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=validation_split)

    # Extract data from training set
    train_image_filenames = train_data['filename'].unique()
    train_images = []
    train_boxes = []
    train_labels = []
    for filename in train_image_filenames:
        image_data = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        train_images.append(image_data)
        image_boxes = train_data[train_data['filename'] == filename][[
            'xmin', 'ymin', 'xmax', 'ymax']].values
        train_boxes.append(image_boxes)
        image_labels = train_data[train_data['filename']
                                  == filename]['class'].values
        train_labels.append(image_labels)

    # Extract data from validation set
    val_image_filenames = val_data['filename'].unique()
    val_images = []
    val_boxes = []
    val_labels = []
    for filename in val_image_filenames:
        image_data = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        val_images.append(image_data)
        image_boxes = val_data[val_data['filename'] == filename][[
            'xmin', 'ymin', 'xmax', 'ymax']].values
        val_boxes.append(image_boxes)
        image_labels = val_data[val_data['filename']
                                == filename]['class'].values
        val_labels.append(image_labels)

    # Convert the inputs to the appropriate format
    train_images = np.array(train_images)
    train_boxes = [np.array(image_boxes) for image_boxes in train_boxes]
    train_labels = [np.array(image_labels) for image_labels in train_labels]
    val_images = np.array(val_images)
    val_boxes = [np.array(image_boxes) for image_boxes in val_boxes]
    val_labels = [np.array(image_labels) for image_labels in val_labels]

    return train_images, train_boxes, train_labels, val_images, val_boxes, val_labels


def define_backbone_alexnet(dimension):
    input = Input(shape = dimension)

    x = Conv2D(filters=96, kernel_size=11, strides=4, name='conv1', activation='relu')(input)
    x = MaxPooling2D(pool_size=3, strides=2, name='pool1',padding='same')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(2)(x)

    # second layer
    x = Conv2D(filters=256, kernel_size=3, strides=1, name="conv2", activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool2", padding='same')(x)
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
    x = MaxPooling2D(pool_size=3, strides=2, name="pool3", padding="same")(x)

    backbone = Model(inputs=input, outputs=x)
    return backbone

def calculate_iou(predictions, ground_truth):
    iou_scores = []
    for i in range(len(predictions)):
        pred_box = predictions[i]
        gt_box = ground_truth[i]
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1],gt_box[1])
        x2 = max(pred_box[2],gt_box[2])
        y2 = max(pred_box[3],gt_box[3])
        intersection = max(0,x2-x1 + 1) * max(0, y2-y1+1)

        pred_area = (pred_box[2] -pred_box[0] + 1) * (pred_box[3] - pred_box[1] +1)
        gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        union = pred_area + gt_area - intersection
        iou = intersection / union
        iou_scores.append(iou)  
    return iou_scores

def plot_iou_his(iou_scores,type_dataset):
    plt.hist(iou_scores,range=(0,1))
    plt.title(f'historgram of{type_dataset}')
    plt.xlabel('Iou score')
    plt.ylabel("Frequency")
    plt.show()
#call in main :plot_iou_histogram(iou_scores)


if __name__ =="__main__":
    while True:
            print('1.Train with enhanced dataset')
            print('2.Train with orginal dataset')
            choice = int(input('Choose choice: '))
            if choice == 1:
                folder = 'dataset-enhanced.csv'
                data = pd.read_csv(folder, sep=',')
                classes = ['node']

                # Extract data and split into training and validation sets
                train_images, train_boxes, train_labels, val_images, val_boxes, val_labels = extract_data(
                    data=data, validation_split=0.2)

                # Train the model with validation
                model = FasterRCNN(
                    classes=train_labels, bounding_box_format=bounding_box_format)
                early_stopping = EarlyStopping(
                    monitor='val_loss', patience=3, mode='min', verbose=1)
                model_checkpoint = ModelCheckpoint(
                    'faster-rcnn-enhanced.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
                model.fit(train_images, train_boxes, train_labels, validation_data=(
                    val_images, val_boxes, val_labels), epochs=10, callbacks=[early_stopping, model_checkpoint])
                model = load_model('faster-rcnn-enhanced.h5')
                predictions = model.predict(val_images)
                iou_scores = calculate_iou(predictions= predictions, ground_truth= val_boxes)
                plot_iou_his(iou_scores)
            if choice == 2:
                folder = 'output-og-anno.csv'
                data = pd.read_csv(folder, sep=',')
                classes = ['node']

                # Extract data and split into training and validation sets
                train_images, train_boxes, train_labels, val_images, val_boxes, val_labels = extract_data(
                    data=data, validation_split=0.2)

                # Train the model with validation
                model = FasterRCNN(
                    classes=train_labels, bounding_box_format=bounding_box_format)
                early_stopping = EarlyStopping(
                    monitor='val_loss', patience=3, mode='min', verbose=1)
                model_checkpoint = ModelCheckpoint(
                    'faster-rcnn-og.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
                model.fit(train_images, train_boxes, train_labels, validation_data=(
                    val_images, val_boxes, val_labels), epochs=10, callbacks=[early_stopping, model_checkpoint])
                model = load_model('faster-rcnn-og.h5')
                predictions = model.predict(val_images)
                iou_scores = calculate_iou(predictions= predictions, ground_truth= val_boxes)
                plot_iou_his(iou_scores)
            if choice == 3:
                break
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(model, X, y):
    # Predict class probabilities
    y_pred_proba = model.predict_proba(X)[:, 1] # we only need the probability of the positive class

    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_proba)

    # Print results
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("AUC-ROC: {:.4f}".format(auc_roc))

    # Return metrics as dictionary
    metrics = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1_score": f1,
               "auc_roc": auc_roc}
    return metrics

    


"""