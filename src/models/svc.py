from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn import svm


def svc(data, label):
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(data, label)

    return model
