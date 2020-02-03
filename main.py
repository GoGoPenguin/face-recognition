from __future__ import absolute_import, division, print_function, unicode_literals

import io
import argparse

import joblib
import numpy as np
from sklearn import preprocessing

from src.models.svc import svc
from src.models.facenet import facenet
from src.utils.load import load_dataset, load_image


def train(image_size,
          batch_size,
          epochs,
          ):
    x_train, y_train = load_dataset('dataset/train/', (image_size, image_size))
    x_test, y_test = load_dataset('dataset/val/', (image_size, image_size))

    # convert to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    # normalize
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # create facenet model
    model = facenet((image_size, image_size, 3))
    # fit model
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
    )
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save_weights('assets/facenet.h5')

    # Evaluate the network
    results = model.predict(x_test)

    # Save test embeddings for visualization in projector
    # The vector and metadata files can be loaded and visualized here:
    # https://projector.tensorflow.org/
    np.savetxt("assets/vecs.tsv", results, delimiter='\t')

    out_m = io.open('assets/meta.tsv', 'w', encoding='utf-8')
    for labels in y_test:
        out_m.write(str(labels) + "\n")
    out_m.close()

    # calculate embedding
    embedding = model.predict(x_train)
    # label encoder
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    # create SVM model
    svm = svc(embedding, y_train)
    # save SVM model and labels
    joblib.dump(svm, 'assets/svm.joblib')
    np.save('assets/labels.npy', encoder.classes_)


def predict(path, image_size):
    # read image from given path
    faces = load_image(path, (image_size, image_size))
    faces.astype(np.float32)
    faces = faces / 255

    # load facenet model
    model = facenet((image_size, image_size, 3))
    model.load_weights('assets/facenet.h5')

    # calculate embedding
    embedding = model.predict(faces)
    # normalize input vector
    in_encoder = preprocessing.Normalizer(norm='l2')
    embedding = in_encoder.transform(embedding)
    # lable encoder
    out_encoder = preprocessing.LabelEncoder()
    out_encoder.classes_ = np.load('assets/labels.npy')
    # create input
    input_vector = np.expand_dims(embedding[0], axis=0)

    # svm model
    svm = joblib.load('assets/svm.joblib')
    result = svm.predict(input_vector)
    prob = svm.predict_proba(input_vector)
    print('Predicted: %s (%.3f)%%' %
          (out_encoder.inverse_transform(result)[0], prob[0, result[0]] * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process face recognition.')
    parser.add_argument(
        'action',
        choices=['train', 'predict'],
        help='What action to perform',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=36,
    )
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160,
    )
    parser.add_argument(
        '--image_path',
        help='Image for prediction.',
        default='dataset/val/elton_john/httpafilesbiographycomimageuploadcfillcssrgbdprgfacehqwMTEODAOTcxNjcMjczMjkzjpg.jpg',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of batches per epoch.',
        default=20,
    )

    args = parser.parse_args()

    if args.action == 'train':
        train(
            args.image_size,
            args.batch_size,
            args.epochs,
        )
    elif args.action == 'predict':
        predict(args.image_path, args.image_size)
