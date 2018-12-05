import ast

import cv2
import numpy as np
import pandas as pd
import pickle
import time
from os import listdir
from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical

from constants import NCATS, BASE_SIZE, INPUT_FOLDER


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


def image_generator_xd(size, batchsize, ks, lw):
    while True:
        for k in np.random.permutation(ks):
            filename = INPUT_FOLDER + 'train_k{}.csv.gz'.format(k)
            for chunk in pd.read_csv(filename, chunksize=batchsize):
                x = df_to_image_array_xd(chunk, size, lw)
                y = to_categorical(chunk.y, num_classes=NCATS)
                yield x, y


def df_to_image_array_xd(df, size, lw):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = preprocess_input(x).astype(np.float32)
    return x


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


if __name__ == "__main__":
    start = time.time()
    gen = fast_image_generator(1000)
    for i in range(100):
        a = next(gen)
    print(time.time() - start)
