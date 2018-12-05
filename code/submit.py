import numpy as np
import pandas as pd
from keras.applications import MobileNet
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

from constants import NCATS, INPUT_FOLDER, CATEGORIES
from preprocessing import df_to_image_array_xd, preds2catids

size = 128
lw = 6

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy')
model.load_weights("weights/128x128_mobilenet_copy.hdf5")

test_predictions = []
test_path = INPUT_FOLDER + "test_simplified.csv"
for chunk in pd.read_csv(test_path, chunksize=50000):
    x_test = df_to_image_array_xd(chunk, size=size, lw=6)
    predictions = list(model.predict(x_test, batch_size=128, verbose=1))
    test_predictions.extend(predictions)

top3 = preds2catids(np.array(test_predictions))
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(CATEGORIES)}
top3cats = top3.replace(id2cat)

test = pd.read_csv(test_path)
test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv('../../submission.csv', index=False)
