import numpy as np
from constants import NCATS, E
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
        self.threshold = 100
        self.weights = np.zeros(10)
        self.progresses = []

    def compute_new_prob_dist(self, i, progress, old_prob_dist):
        if len(self.progresses) > self.threshold:
            rank = bisect_right(sorted(self.progresses), progress)
            self.progresses.append(progress)
            weight_diff = (2*rank-len(self.progresses))/len(self.progresses)/10
            assert 1 >= weight_diff >= -1
            prob = (old_prob_dist[i])*E+(1-E)/10
            assert 0 <= prob <= 1
            self.weights[i] += weight_diff/prob
            exp_weights = [e**weight for weight in self.weights]
            sum_weights = sum(exp_weights)
            return [weight/sum_weights for weight in exp_weights]
        else:
            self.progresses.append(progress)
            return old_prob_dist
