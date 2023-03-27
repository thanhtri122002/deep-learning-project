import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.applications import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend import epsilon

# Define the configuration for the model


class Config:
    NAME = "node_detection"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + Node
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    TOP_DOWN_PYRAMID_SIZE = 256
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

# Define the dataset class


class NodeDataset(Sequence):
    def __init__(self, csv_file, batch_size, config):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.config = config
        self.image_ids = self.data["filename"].unique()
        self.num_images = len(self.image_ids)
        self.indexes = np.arange(self.num_images)

    def __len__(self):
        return int(np.ceil(self.num_images / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx *
                                     self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_gt_boxes = []
        batch_gt_class_ids = []
        for i in batch_indexes:
            image_id = self.image_ids[i]
            image_path = f"images/{image_id}"
            image = load_img(image_path, target_size=(
                self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM))
            image = img_to_array(image)
            image /= 255.0
            image -= 0.5
            image *= 2.0
            nodes = self.data[self.data["filename"] == image_id]
            boxes = nodes[["xmin", "ymin", "xmax", "ymax"]].values
            class_ids = np.ones((len(boxes),), dtype=np.int32)
            batch_images.append(image)
            batch_gt_boxes.append(boxes)
            batch_gt_class_ids.append(class_ids)
        batch_images = np.array(batch_images)
        batch_gt_boxes = np.array(batch_gt_boxes)
        batch_gt_class_ids = np.array(batch_gt_class_ids)
        return [batch_images, batch_gt_boxes, batch_gt_class_ids], np.zeros((self.batch_size, 1))

# Define the model architecture


def build_model(config):
    backbone = ResNet50(input_shape=(config.IMAGE_MAX_DIM,
                        config.IMAGE_MAX_DIM, 3), include_top=False)
    input_image = layers.Input(
        shape=(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3))
    _, C2, C3, C4, C5 = backbone(input_image)
    # Define the model architecture


def build_model(config):
    backbone = ResNet50(input_shape=(config.IMAGE_MAX_DIM,
                        config.IMAGE_MAX_DIM, 3), include_top=False)
    input_image = layers.Input(
        shape=(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3))
    _, C2, C3, C4, C5 = backbone(input_image)

    # Create the RPN
    rpn = build_rpn_model(config.RPN_ANCHOR_SCALES,
                          config.RPN_ANCHOR_RATIOS, config.BACKBONE_STRIDES)

    # Get the feature maps and anchors from the RPN
    rpn_feature_maps = rpn([C2, C3, C4, C5])
    anchors = rpn_feature_maps[1]

    # Create the classifier
    classifier = build_classifier_model(
        config.NUM_CLASSES, config.IMAGE_MAX_DIM, config.TOP_DOWN_PYRAMID_SIZE)

    # Connect the classifier to the RPN
    proposals = layers.RPNProposal(
        maximum_proposals=config.RPN_TRAIN_ANCHORS_PER_IMAGE)(rpn_feature_maps)
    classifier_input = layers.ROIAlign([config.POST_NMS_ROIS_TRAINING, config.POST_NMS_ROIS_TRAINING],
                                       name="roi_align_classifier")([proposals] + rpn_feature_maps)
    classifier_output = classifier(classifier_input)
    detections = layers.DetectionLayer(num_classes=config.NUM_CLASSES, name="roi_output")([
        proposals, classifier_output, anchors])

    # Create the model
    model = Model(inputs=input_image, outputs=[detections])
    return model


# Build and compile the model
config = Config()
model = build_model(config)
model.compile(optimizer=tf.keras.optimizers.Adam(
    epsilon=epsilon()), loss=[lambda y_true, y_pred: y_pred])
