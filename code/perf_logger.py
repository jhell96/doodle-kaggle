from keras.callbacks import Callback

class LogPerformance(Callback):
    def __init__(self, log_func):
        Callback.__init__(self)
        self.log_func = log_func

    def on_epoch_end(self, _, logs={}):
        self.log_func(logs=logs)
