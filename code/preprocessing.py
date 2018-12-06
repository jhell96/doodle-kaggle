import ast
import json
import time
from os import listdir

import cv2
import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical

from constants import NCATS, BASE_SIZE, CATEGORIES_TO_INDEX

np.random.seed(seed=0)


class Database:
    def __init__(self, batchsize, size, lw, remove_unrecognized=False):
        self.batchsize = batchsize
        self.size = size
        self.lw = lw

        indir = "/home/doodle/pedro/data/training_data_grouped/"
        sorted_files = sorted(list(listdir(indir)), key=str.lower)
        sorted_names = [file.split('.')[0] for file in sorted_files]
        class_counts = json.load(open("/home/doodle/pedro/data/counts.json"))
        total = sum(class_counts.values())

        self.streams = [ClassStream(indir + file, remove_unrecognized) for file in sorted_files]
        self.prob_dist = [class_counts[name] / total for name in sorted_names]

    def _get_unprocessed_next_batch(self, batch_size):
        X, Y = [], []
        num_samples = 5
        for i in range(int(batch_size / num_samples)):
            class_stream = np.random.choice(self.streams, p=self.prob_dist)
            for j in range(num_samples):
                x, y = class_stream.get_next()
                X.append(x)
                Y.append(y)
            return X, Y
        return X, Y

    def processed_batch_generator(self):
        while True:
            x, y = self._get_unprocessed_next_batch(self.batchsize)
            yield get_image_array(x, self.size, self.lw), get_y_encoding(y)


class ClassStream:
    def __init__(self, file, remove_unrecognized, chunksize=1000):
        self.samples = []
        self.word = list(pd.read_csv(file, nrows=1)['word'])[0]
        self.sample_gen = self.sample_generator(chunksize)
        self.file = file
        self.remove_unrecognized = remove_unrecognized

    def get_next(self):
        if len(self.samples) == 0:
            self.samples.extend(next(self.sample_gen))
        return self.samples.pop(), self.word

    def sample_generator(self, chunksize):
        while (True):
            for chunk in pd.read_csv(self.file, chunksize=chunksize):
                if self.remove_unrecognized:
                    chunk = chunk[chunk['recognized']]
                yield chunk['drawing']


def draw_cv2(raw_strokes, size, lw):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            # colors = (255, 255, 255)
            # #color = 255
            # color = 255 - min(t, 10) * 13
            # colors = [0]*3
            # colors[0] = 255
            # colors[min(i,2)] = color
            # colors =  (255, 255, 255)
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), 255, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
        return img
    else:
        return img


def get_image_array(image_strings, size, lw):
    images = list(map(ast.literal_eval, image_strings))
    x = np.zeros((len(images), size, size, 1))
    for i, raw_strokes in enumerate(images):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = preprocess_input(x).astype(np.float32)
    return x


def get_y_encoding(words):
    y = [CATEGORIES_TO_INDEX[word] for word in words]
    return to_categorical(y, num_classes=NCATS)


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


if __name__ == "__main__":
    start = time.time()
    for i in range(100):
        a = next(gen)
    print(time.time() - start)
