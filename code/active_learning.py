import numpy as np
from constants import NCATS

class BasicActiveLearner:
    def compute_new_prob_dist(self, y_actual, y_predict_prob, old_prob):
        new_prob_dist = [0]*NCATS
        for act_array, pred_prob in zip(y_actual, y_predict_prob):
            pred = np.argmax(pred_prob)
            act = np.argmax(act_array)
            if act != np.argmax(pred):
                new_prob_dist[act] += 1
        total = sum(new_prob_dist)
        return [prob/total for prob in new_prob_dist]


