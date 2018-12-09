import numpy as np
from constants import NCATS
from math import e
from bisect import bisect_right


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

class DifficultyLearner:
    def __init__(self):
        self.weights = [0]*10
        self.progresses = [0]

    def compute_new_prob_dist(self, i, progress, prob_i):
        if len(self.progresses) > 50:
            self.progresses = self.progresses[:49]
        rank = bisect_right(sorted(self.progresses), progress)
        self.progresses.append(progress)
        self.weights[i] += (rank-50)/100/prob_i
        exp_weights = [e**weight for weight in self.weights]
        sum_weights = sum(exp_weights)
        return [weight/sum_weights for weight in exp_weights]

