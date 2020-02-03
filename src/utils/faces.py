from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
from mtcnn import MTCNN

# create the detector, using default weights
detector = MTCNN()


def detect_faces(img, shape):
    # covert to RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect faces in the image
    results = detector.detect_faces(image)

    faces = []
    for result in results:
        # extract the bounding box
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, shape, interpolation=cv2.INTER_CUBIC)
        faces.append(face)

    return faces
