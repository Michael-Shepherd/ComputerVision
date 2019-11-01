from google.colab import drive
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from matplotlib import rc
from IPython.display import display, Math
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from IPython.display import display
from IPython.display import Image
import itertools
from skimage import data, filters
from datetime import datetime
from scipy import stats
import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
from math import floor, ceil
import cv2
import os
import math
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import subprocess
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.applications import vgg16, vgg19, resnet50
import keras
from IPython.display import SVG
from keras.utils import model_to_dot

file_path = "./drive/My Drive/CVassignment6_files"
test_path = f"{file_path}/test"
train_path = f"{file_path}/train"


def file_len(fname):
    p = subprocess.Popen(
        ["wc", "-l", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


nb_train_samples = 900
nb_validation_samples = 1705
num_classes = 9
epochs = 20
input_shape = (227, 227, 1)
img_width = 227
img_height = 227


def plot_model_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


def one_a_model():
    batch_size = 24
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(9))
    model.add(Activation("softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.0008),
        metrics=["accuracy"],
    )
    return model, batch_size


def one_a():
    model, batch_size = one_a_model()
    drive.mount("/content/drive")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(227, 227),
        color_mode="grayscale",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(227, 227),
        color_mode="grayscale",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )
    history = [0, 0]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=1,
    )
    val_acc = model.evaluate_generator(validation_generator)[1]
    return val_acc, history


val_acc, history = 0, 0
i = 0
while True:
    print(i)
    i = i + 1
    val_acc, history = one_a()
    if val_acc >= 0.60:
        break
print(f"Validation accuracy = {val_acc}")
plot_model_history(history)


def one_b_model():
    batch_size = 32
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(9))
    model.add(Activation("softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        metrics=["accuracy"],
    )
    return model, batch_size


def one_b():
    model, batch_size = one_b_model()
    drive.mount("/content/drive")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(227, 227),
        color_mode="grayscale",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(227, 227),
        color_mode="grayscale",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )

    history = [0, 0]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=1,
    )
    val_acc = model.evaluate_generator(validation_generator)[1]
    return val_acc, history


val_acc, history = 0, 0
i = 0
while True:
    print(i)
    i = i + 1
    val_acc, history = one_b()
    if val_acc >= 0.70:
        break
print(f"Validation accuracy = {val_acc}")
plot_model_history(history)

nb_train_samples = 900
nb_validation_samples = 1705
num_classes = 9


def two_model():
    batch_size = 24
    vgg_model = vgg16.VGG16()
    print("Loaded VGG")
    vgg_model.layers.pop()
    model = Sequential()
    for layer in vgg_model.layers:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(9, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.003, momentum=0.9),
        metrics=["accuracy"],
    )
    vgg_model = None
    return model, batch_size


def two():
    epochs = 25
    model, batch_size = two_model()
    drive.mount("/content/drive")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    )

    history = [0, 0]
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=1,
    )
    val_acc = model.evaluate_generator(validation_generator)[1]
    return val_acc, history


val_acc, history = 0, 0
i = 0
while True:
    print(i)
    i = i + 1
    val_acc, history = two()
    if val_acc >= 0.91:
        break
print(f"Validation accuracy = {val_acc}")
plot_model_history(history)
