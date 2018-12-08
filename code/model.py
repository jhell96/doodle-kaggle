import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import MobileNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from active_learning import BasicActiveLearner
from metrics import top_3_accuracy
from perf_logger import *
from preprocessing import Database, get_image_array, get_y_encoding
from constants import CATEGORIES_TO_INDEX

EXPERIMENT_NAME = "128x128_mobilenet_no_learner_keep_unrecognized_sample_easy"

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create())  # hook into the MongoDB
ex.captured_out_filter = apply_backspaces_and_linefeeds  # make output more readable

np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)


@ex.config
def track_params():
    batch_size = 512
    num_samples = 10000
    steps = num_samples // batch_size
    epochs = 50
    size = 128
    lw = 6
    saved_model = None  # "weights/128x128_mobilenet_copy.hdf5"


@ex.capture
def log_performance(_run, logs):
    _run.add_artifact("weights/{}_weights_{}.hdf5".format(EXPERIMENT_NAME, _run._id))
    _run.log_scalar("categorical_crossentropy", float(logs.get('categorical_crossentropy')))
    _run.log_scalar("categorical_accuracy", float(logs.get('categorical_accuracy')))
    _run.log_scalar("top_3_accuracy", float(logs.get('top_3_accuracy')))
    _run.log_scalar("val_categorical_crossentropy", float(logs.get('val_categorical_crossentropy')))
    _run.log_scalar("val_categorical_accuracy", float(logs.get('val_categorical_accuracy')))
    _run.log_scalar("val_top_3_accuracy", float(logs.get('val_top_3_accuracy')))
    _run.result = float(logs.get('val_top_3_accuracy'))


def log_probs(exp_id, probs):
    filename = "logs/class_probs/exp_{}.csv".format(exp_id)
    with open(filename, "a") as f:
        f.write(str(probs)[1:-1].replace(" ", "") + "\n")


@ex.automain
def main(_run, batch_size, epochs,
         steps, size, lw, saved_model):
    valid_df = pd.read_csv("/home/doodle/pedro/data/validation.csv")
    in_categories = [word in CATEGORIES_TO_INDEX for word in valid_df["word"]]
    valid_df = valid_df[in_categories]
    valid_df = valid_df[:1000]
    db = Database(batch_size, size, lw, remove_unrecognized=False)
    active_learner = BasicActiveLearner()
    x_valid = get_image_array(valid_df["drawing"], size, lw)
    y_valid = get_y_encoding(valid_df['word'])
    model = MobileNet(weights=None, input_shape=(size, size, 1), classes=30)
    model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy',
                  metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

    WEIGHTS_PATH = "weights/{}_weights_{}.hdf5".format(EXPERIMENT_NAME, _run._id)

    # this is a bug fix. we need to create the weights file before the mongo db tries to
    # access the file (otherwise, it throws an error because mongo tries to access before its
    # created by the training callback function)
    ##################################
    with open(WEIGHTS_PATH, 'w') as f:
        f.write(" ")
    #################################

    print("finished compiling model")
    if saved_model is not None:
        model.load_weights(saved_model)
    for _ in range(epochs):
        train_datagen = db.processed_batch_generator()
        model.fit_generator(
            train_datagen, steps_per_epoch=steps, epochs=1, verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[
                ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss',
                                save_best_only=True, mode='auto', period=10),
                LogPerformance(log_performance),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001),
                # LRFinder(batch_size * steps, batch_size, minimum_lr=1e-5, maximum_lr=1, lr_scale='exp',
                #          save_dir="learning_rate_losses/", verbose=True)
            ]
        )
        log_probs(_run._id, db.prob_dist)
        # y_valid_pred_prob = model.predict(x_valid)
        # db.prob_dist = active_learner.compute_new_prob_dist(y_valid, y_valid_pred_prob, db.prob_dist)
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]