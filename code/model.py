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

EXPERIMENT_NAME = "128x128_mobilenet_no_learner_removing_unrecognized"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create())  # hook into the MongoDB
ex.captured_out_filter = apply_backspaces_and_linefeeds  # make output more readable


np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)


@ex.config
def track_params():
    batch_size = 512
    num_samples = 100000
    steps = num_samples // batch_size
    epochs = 50
    size = 128
    lw = 6
    saved_model = None  # "weights/128x128_mobilenet_copy.hdf5"


@ex.capture
def log_performance(_run, logs):
    _run.add_artifact("weights/" + EXPERIMENT_NAME + "_weights_" + ".hdf5")
    _run.log_scalar("categorical_crossentropy", float(logs.get('categorical_crossentropy')))
    _run.log_scalar("categorical_accuracy", float(logs.get('categorical_accuracy')))
    _run.log_scalar("top_3_accuracy", float(logs.get('top_3_accuracy')))
    _run.log_scalar("val_categorical_crossentropy", float(logs.get('val_categorical_crossentropy')))
    _run.log_scalar("val_categorical_accuracy", float(logs.get('val_categorical_accuracy')))
    _run.log_scalar("val_top_3_accuracy", float(logs.get('val_top_3_accuracy')))
    _run.result = float(logs.get('val_top_3_accuracy'))


@ex.automain
def main(batch_size, epochs,
         steps, size, lw, saved_model):
    valid_df = pd.read_csv("/home/doodle/pedro/data/validation.csv", nrows=10000)

    db = Database(batch_size, size, lw, remove_unrecognized=True)
    active_learner = BasicActiveLearner()
    x_valid = get_image_array(valid_df["drawing"], size, lw)
    y_valid = get_y_encoding(valid_df['word'])

    model = MobileNet(weights=None, input_shape=(size, size, 1), classes=340)
    model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy',
                  metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
    print("finished compiling model")
    if saved_model is not None:
        model.load_weights(saved_model)
    for _ in range(epochs):
        train_datagen = db.processed_batch_generator()
        model.fit_generator(
            train_datagen, steps_per_epoch=steps, epochs=1, verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[
                ModelCheckpoint("weights/" + EXPERIMENT_NAME + "_weights_" + ".hdf5", monitor='val_loss',
                                save_best_only=True, mode='auto', period=10),
                LogPerformance(log_performance),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001),
                # LRFinder(batch_size * steps, batch_size, minimum_lr=1e-5, maximum_lr=1, lr_scale='exp',
                #          save_dir="learning_rate_losses/", verbose=True)
            ]
        )
        # y_valid_pred_prob = model.predict(x_valid)
        # db.prob_dist = active_learner.compute_new_prob_dist(y_valid, y_valid_pred_prob, db.prob_dist)

    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]
#
#
# if __name__ == "__main__":
#     batch_size = 512
#     num_samples = 5000000
#     steps = num_samples // batch_size
#     epochs = 40
#     size = 128
#     lw = 6
#     saved_model = None  # "weights/128x128_mobilenet_copy.hdf5"
#     main(batch_size=batch_size, steps=steps, epochs=epochs, lw=lw, size=size, saved_model=saved_model)
