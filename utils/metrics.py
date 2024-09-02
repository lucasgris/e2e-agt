import numpy as np
import torch
from jiwer import cer, wer, mer, wil 
import mir_eval
from .util import tab_to_hz_mir_eval

def f_measure(predicted, target, offset_ratio=True, num_strings=6, ignore_empty=False):
    if len(predicted.shape) == 2:
        predicted = np.expand_dims(predicted, axis=0)
        target = np.expand_dims(target, axis=0)
    batch_precision = []
    batch_recall = []
    batch_f_measure = []
    for i in range(predicted.shape[0]):
        ref_intervals, ref_pitches = tab_to_hz_mir_eval(target[i])
        est_intervals, est_pitches = tab_to_hz_mir_eval(predicted[i])
        for s in range(num_strings):
            if ref_intervals[s].shape[0] == 0 or est_intervals[s].shape[0] == 0:
                precision = recall = f_measure = 0
                if ignore_empty:
                    continue
            else:
                precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals[s], ref_pitches[s], est_intervals[s], est_pitches[s],
                    offset_ratio=offset_ratio
                )
            batch_precision.append(precision)
            batch_recall.append(recall)
            batch_f_measure.append(f_measure)
    return np.mean(batch_f_measure), np.mean(batch_precision), np.mean(batch_recall)

def f_measure_(predicted, target):
    tp = fn = fp = 0
    if len(predicted.shape) == 2:
        predicted = np.expand_dims(predicted, axis=0)
        target = np.expand_dims(target, axis=0)
    assert predicted.shape == target.shape, f"Shapes do not match: {predicted.shape} != {target.shape} (Batch size, n_strings, n_timesteps)"
    for i in range(predicted.shape[0]):
        for s in range(predicted.shape[1]):
            for t in range(predicted.shape[2]):
                if predicted[i, s, t] == target[i, s, t] != 0:
                    tp += 1
                elif predicted[i, s, t] != 0 and target[i, s, t] == 0:
                    fp += 1
                elif predicted[i, s, t] == 0 and target[i, s, t] != 0:
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return f_measure, precision, recall


class TabMetrics:

    def tab_to_words(tab: np.ndarray, collapse_silences=True, sil='-', real_class_shift=-1, replace_chars=True):
        words = []
        for t in range(len(tab[0])):
            word = ""
            for s in range(len(tab)):
                if tab[s][t] == 0:
                    word += "-"
                else:
                    if replace_chars:
                        # replace the fret positions with greek letters to compute the error rates
                        word += str(chr(int(tab[s][t]) + real_class_shift + 945))
                    else:
                        word += str(int(tab[s][t]) + real_class_shift)
            if collapse_silences: 
                if len(words) == 0:
                    words.append(word)
                else:
                    if word.replace(sil, '') != '':
                        words.append(word)
                    elif words[-1].replace(sil, '') != '':
                        words.append(word)
            else:
                words.append(word)
        return " ".join(words)

    def tab_error_rate(predicted, target, **kwargs):
        """
        Computes the error rate between two tablatures.
        In practice, the tablature error rate is the word error rate considering each time step as a word.

        For example:
        |--1--------- ... --|
        |--1--------- ... --|
        |-----3------ ... --|
        |------------ ... --|
        |---------4-- ... --|
        |----------5- ... --|
        In this case, we would have the words: ------ 11---- ------ --3--- ------ ----4- ------ -----5 ...  ------
        Note that we colapse the silence words to compute the error rate.

        Args:
            predicted and target: a matrix of shape (n_strings, n_timesteps) containing the predicted tablature, where the value at each position is the fret number.
        """
        return wer(
            TabMetrics.tab_to_words(target, **kwargs),
            TabMetrics.tab_to_words(predicted, **kwargs),
        )

    def fret_error_rate(predicted, target, **kwargs):
        """
        Computes the fret error rate between two tablatures.
        In practice, the fret error rate is the character error rate considering each time step as a word.

        For example:
        |--1--------- ... --|
        |--1--------- ... --|
        |-----3------ ... --|
        |------------ ... --|
        |---------4-- ... --|
        |----------5- ... --|
        In this case, we would have the words: ------ 11---- ------ --3--- ------ ----4- ------ -----5 ...  ------
        Note that we colapse the silence words to compute the error rate.

        Args:
            predicted and target: a matrix of shape (n_strings, n_timesteps) containing the predicted tablature, where the value at each position is the fret number.
        """
        return cer(
            TabMetrics.tab_to_words(target, **kwargs),
            TabMetrics.tab_to_words(predicted, **kwargs),
        )


class TabCNNMetrics:
    # Adapted from https://github.com/andywiggins/tab-cnn/blob/master/model/Metrics.py
    # The tab parameter is expected to have the values of the predictions of the network.
    # Please notice that we used 0 for silence in the output of the network, therefore
    # the real value of the fret should be subtracted by 1. You don't need to
    # precompute these values. The class will take care of it.

    # For metrics computation, you can pass either a batch of single frame predictions or
    # a frame-wise prediction. Do not pass a batch of frame-wise predictions.

    # NOTE: This class does not handle well onset predictions. 
    # It expects frame-level predictions, which is a simpler approach for
    # music transcription where the onset detection is not a concern. 
    # See. Benetos, Emmanouil, et al. "Automatic music transcription: 
    # An overview." IEEE Signal Processing Magazine 36.1 (2018): 20-30. 
    # Using this class with onsets will lead to inf and nan values.
    def __init__(self, closed_string_label=0, num_strings=6, fretboard_size=23):
        self.closed_string_label = closed_string_label
        self.fretboard_size = fretboard_size
        self.num_strings = num_strings

    def tab2pitch(self, tab):
        # This method does not support batch
        pitch_vector = np.zeros(44)
        string_pitches = [40, 45, 50, 55, 59, 64]
        for string_num in range(self.num_strings):
            # fret_vector = tab[string_num]
            # fret_class = np.argmax(fret_vector, -1)
            fret_class = tab[string_num]
            if fret_class != self.closed_string_label:
                pitch_num = fret_class + string_pitches[string_num] - 40
                pitch_vector[pitch_num] = 1
        return pitch_vector

    def tab2bin(self, tab):
        # This method does not support batch
        tab_arr = np.zeros((self.num_strings, self.fretboard_size))
        for string_num in range(self.num_strings):
            # fret_vector = tab[string_num]
            # fret_class = np.argmax(fret_vector, -1)
            fret_class = tab[string_num]
            if fret_class != self.closed_string_label:
                fret_num = fret_class
                tab_arr[string_num][fret_num] = 1
        return tab_arr

    def pitch_precision(self, pred, gt):
        pitch_pred = np.array(list(map(self.tab2pitch, pred)))
        pitch_gt = np.array(list(map(self.tab2pitch, gt)))
        numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
        denominator = np.sum(pitch_pred.flatten())
        return (1.0 * numerator) / denominator

    def pitch_recall(self, pred, gt):
        pitch_pred = np.array(list(map(self.tab2pitch, pred)))
        pitch_gt = np.array(list(map(self.tab2pitch, gt)))
        numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
        denominator = np.sum(pitch_gt.flatten())
        return (1.0 * numerator) / denominator

    def pitch_f_measure(self, pred, gt):
        p = self.pitch_precision(pred, gt)
        r = self.pitch_recall(pred, gt)
        f = (2 * p * r) / (p + r)
        return f

    def tab_precision(self, pred, gt):
        # get rid of "closed" class, as we only want to count positives
        tab_pred = np.array(list(map(self.tab2bin, pred)))
        tab_gt = np.array(list(map(self.tab2bin, gt)))
        numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
        denominator = np.sum(tab_pred.flatten())
        return (1.0 * numerator) / denominator

    def tab_recall(self, pred, gt):
        # get rid of "closed" class, as we only want to count positives
        tab_pred = np.array(list(map(self.tab2bin, pred)))
        tab_gt = np.array(list(map(self.tab2bin, gt)))
        numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
        denominator = np.sum(tab_gt.flatten())
        return (1.0 * numerator) / denominator

    def tab_f_measure(self, pred, gt):
        p = self.tab_precision(pred, gt)
        r = self.tab_recall(pred, gt)
        f = (2 * p * r) / (p + r)
        return f

    def tab_disamb(self, pred, gt):
        tp = self.tab_precision(pred, gt)
        pp = self.pitch_precision(pred, gt)
        return tp / pp
