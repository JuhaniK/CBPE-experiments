"""
Copyright (c) 2025 Juhani Kivimäki
Released under the MIT License. See LICENSE file in project root for details.

This module includes the shortcut estimators. 
"""

import numpy as np
from estimators import binary_splitter

def estimate_confusion_matrix(confidences):
    positive, negative = binary_splitter(confidences)
    tp = np.sum(positive)
    fp = np.sum(1 - positive)
    fn = np.sum(negative)
    tn = np.sum(1 - negative)
    return tp, fp, tn, fn


def s_accuracy(confidences):
    n =  len(confidences)
    tp, _, tn, _ = estimate_confusion_matrix(confidences)
    accuracy = (tp + tn) / n
    return accuracy


def s_precision(confidences):
    tp, fp, _, _ = estimate_confusion_matrix(confidences)
    precision = tp / (tp + fp)
    return precision


def s_recall(confidences):
    tp, _, _, fn = estimate_confusion_matrix(confidences)
    recall = tp / (tp + fn)
    return recall


def s_f1(confidences):
    tp, fp, _, fn = estimate_confusion_matrix(confidences)
    f1 = 2 * tp / (2*tp + fp + fn)
    return f1

def s_all(confidences):
    n =  len(confidences)
    tp, fp, tn, fn = estimate_confusion_matrix(confidences)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2*tp + fp + fn)
    return [accuracy, precision, recall, f1]