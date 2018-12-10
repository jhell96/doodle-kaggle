import ast
import json
import time
from os import listdir

import cv2
import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical

from constants import NCATS, BASE_SIZE, CATEGORIES_TO_INDEX, NUM_DIFFICULTIES, E

np.random.seed(seed=0)


class Database:
    def __init__(self, batchsize, size, lw, remove_unrecognized=False):
        self.batchsize = batchsize
        self.size = size
        self.lw = lw

        indir = "/home/doodle/pedro/data/training_data_subsampled/"
        sorted_files = sorted(list(listdir(indir)), key=str.lower)
        sorted_files = [file for file in sorted_files if file.split('.')[0] in CATEGORIES_TO_INDEX]
        sorted_names = [file.split('.')[0] for file in sorted_files]
        class_counts = json.load(open("/home/doodle/pedro/data/counts.json"))
        total = sum([class_counts[name] for name in sorted_names])

        self.streams = [FileStream(indir + file, remove_unrecognized) for file in sorted_files]
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

    def update_prob_dist(self, prob_dist_updater):
        self.prob_dist = prob_dist_updater(self.prob_dist)

    def processed_batch_generator(self):
        while True:
            x, y = self._get_unprocessed_next_batch(self.batchsize)
            yield get_image_array(x, self.size, self.lw), get_y_encoding(y)


class DifficultyDatabase:
    def __init__(self, batchsize, size, lw):
        self.batchsize = batchsize
        self.size = size
        self.lw = lw
        indir = "/home/doodle/pedro/data/training_data_by_difficulty/"
        self.streams = [FileStream(indir + str(i + 1), remove_unrecognized=False) for i in range(NUM_DIFFICULTIES)]
        self.prob_dist = [0.1] * NUM_DIFFICULTIES

    def _get_unprocessed_next_batch(self, batch_size):
        X, Y = [], []
        if np.random.random() < E:
            index = np.random.choice(list(range(len(self.streams))), p=self.prob_dist)
        else:
            index = np.random.choice(list(range(len(self.streams))))
        for i in range(batch_size):
            x, y = self.streams[index].get_next()
            X.append(x)
            Y.append(y)
        return X, Y, index

    def processed_batch_generator(self):
        while True:
            x, y, index = self._get_unprocessed_next_batch(self.batchsize)
            yield (get_image_array(x, self.size, self.lw), get_y_encoding(y)), index


class UniformDatabase:
    def __init__(self, batchsize, size, lw):
        self.batchsize = batchsize
        self.size = size
        self.lw = lw
        indir = "/home/doodle/pedro/data/training_data_by_difficulty/"
        self.stream = FileStream(indir + "full.csv", remove_unrecognized=False)

    def processed_batch_generator(self):
        while True:
            X, Y = [], []
            for i in range(self.batchsize):
                x, y = self.stream.get_next()
                X.append(x)
                Y.append(y)
            yield get_image_array(X, self.size, self.lw), get_y_encoding(Y)


class FileStream:
    def __init__(self, file, remove_unrecognized, chunksize=1000):
        self.Xs = []
        self.Ys = []
        self.sample_gen = self.sample_generator(chunksize)
        self.file = file
        self.remove_unrecognized = remove_unrecognized

    def get_next(self):
        if len(self.Xs) == 0:
            xs, ys = next(self.sample_gen)
            self.Xs.extend(xs)
            self.Ys.extend(ys)
        return self.Xs.pop(), self.Ys.pop()

    def sample_generator(self, chunksize):
        while (True):
            for chunk in pd.read_csv(self.file, chunksize=chunksize, nrows=40000):
                if self.remove_unrecognized:
                    chunk = chunk[chunk['recognized']]
                yield chunk['drawing'], chunk["word"]


def draw_cv2(raw_strokes, size, lw):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            cv2.line(img, (stroke[0][i], stroke[1][i]),
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


def full_image_generator(size, batchsize, lw, proportion=1):
    while True:
        filename = "/home/doodle/pedro/data/sorted_traing_data.csv"
        df = pd.read_csv(filename)
        df = df[:int(len(df) * proportion)]
        tmp = "/home/doodle/pedro/data/tmp.csv"
        df.sample(frac=1).to_csv(tmp, index=False)
        for chunk in pd.read_csv(tmp, chunksize=batchsize):
            x = get_image_array(chunk["drawing"], size, lw)
            y = get_y_encoding(chunk["word"])
            yield x, y


if __name__ == "__main__":
    start = time.time()
    for i in range(100):
        a = next(gen)
    print(time.time() - start)
