from __future__ import absolute_import, division, print_function, unicode_literals

import os

import cv2
import numpy as np
from src.utils.faces import detect_faces


def load_dataset(directory, shape):
    x, y = [], []

    # enumerate foldersm on per class
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'

        if not os.path.isdir(path):
            continue

        faces = []
        for filename in os.listdir(path):
            f = path + filename
            if os.path.isfile(f):
                image = cv2.imread(f)
                faces.extend(detect_faces(image, shape))

        labels = [subdir for _ in range(len(faces))]

        x.extend(faces)
        y.extend(labels)

    return np.asarray(x), np.asarray(y)


def load_image(path, shape):
    image = cv2.imread(path)
    result = detect_faces(image, shape)

    return np.asarray(result)
