import numpy as np
from constants import NCATS

class BasicActiveLearner:
    def compute_new_prob_dist(self, y_actual, y_predict_prob, old_prob):
        new_prob_dist = [0]*NCATS
        for act_array, pred_prob in zip(y_actual, y_predict_prob):
            pred = int(np.argmax(pred_prob))
            act = int(np.argmax(act_array))
            if int(act) != pred:
                new_prob_dist[int(act)] += 1
        total = sum(new_prob_dist)
        new_prob = [prob/total for prob in new_prob_dist]
        return new_prob


