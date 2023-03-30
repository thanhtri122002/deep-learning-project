import os
import cv2
import numpy as np
import pandas as pd
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 50
BATCH_SIZE = 16
NUM_CLASSES = 1
NUM_EPOCHS = 200

# Load and preprocess data


def load_data(annotations_file, img_folder):
    df = pd.read_csv(annotations_file)
    X = []
    y = []

    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(
            img_folder, row['filename']), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Convert grayscale to 3-channel
        img = np.stack([img, img, img], axis=-1)

        bbox = np.array([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        X.append(img)
        y.append(bbox)

    X = np.array(X)
    y = np.array(y)

    return X, y


# Load dataset and annotations
X, y = load_data("annotation.csv", "processed_img")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Feature extractor: VGG16
#vgg16 = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 1))
#vgg16.trainable = False

# Feature extractor: VGG16
vgg16 = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg16.trainable = False


x = vgg16(input_layer)
x = GlobalAveragePooling2D()(x)

# Regression head for bounding box coordinates
regression_head = Dense(128, activation='relu')(x)
regression_head = Dropout(0.5)(regression_head)
regression_head = Dense(64, activation='relu')(regression_head)
regression_head = Dropout(0.5)(regression_head)
regression_head = Dense(32, activation='relu')(regression_head)
regression_head = Dropout(0.5)(regression_head)
bbox_output = Dense(4, activation='linear',
                    name='bbox_output')(regression_head)

# Build the model
model = Model(inputs=input_layer, outputs=bbox_output)

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='mse')

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[
        ModelCheckpoint("model.h5", save_best_only=True, verbose=1),
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, verbose=1)
    ],
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {loss}")

# Save model
model.save("faster_rcnn.h5")
