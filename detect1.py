import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.applications import ResNet50

from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend import epsilon
folder = 'output-stretched.csv'
df = pd.read_csv(folder , sep=',')
print(df)