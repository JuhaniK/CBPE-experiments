"""
Copyright (c) 2025 Juhani Kivimäki
Released under the MIT License. See LICENSE file in project root for details.

This module includes the actual estimators along with some utility functions. 
"""


import warnings
import numpy as np

from time import time
from poibin import PoiBin
from collections import defaultdict


# Helper functions
def checksum(results):
    """
    A helper function used to sum all the probabilities of a given distribution (should be equal to 1) and printing the sum.
    
    Arguments:
        results (dict): A dictionary representing a discrete probability distribution (keys as variable values, values as probabilities)
    Returns:
        None
    """
    checksum = 0.0
    for value in results.values():
        checksum += value
    print("Checksum is", checksum)


def printout(metric, results):    
    """
    A helper function creating a printout of the expectation and CIs for the desired metric.
    
    Arguments:
        metric (str): Name of the metric
        results (dict): A dictionary representing a discrete probability distribution (keys as variable values, values as probabilities)
    Returns:
        None
    """
    print(f"\nThe distribution of {metric} is:")
    expectation = calculate_expectation(results, printout=True)
    print(f"with expected value of {expectation:.4f}.")
    cis = calculate_CI(results, 0.05)
    print(f"and with 95% CI bounds [{cis[0]:.2f}, {cis[1]:.2f}].")
    checksum(results)


# Functions used to form the Poisson binomial distributions
def binary_splitter(confidences, threshold=0.5):
    """
    Splits a Numpy array of confidence scores into two arrays. First array contains values above and second below a given threshold.
    
    Arguments:
        confidences (NumPy array): An array with confidence scores
        threshold (float): Defines the splitting point
    Returns:
        positive (NumPy array): An array holding all the confidence scores above or equal to the threshold
        negative (NumPy array): An array holding all the confidence scores below the threshold
    """
    pos_mask = confidences >= threshold
    positive = confidences[pos_mask]
    negative = confidences[~pos_mask]
    return positive, negative


def positives(positive_confidences, number_of_values):
    """
    Returns the Poisson binomial distributions for true and false positives.
    
    Arguments:
        positive_confidences (NumPy array): An array with confidence scores for the positive class
        number_of_values (int): The number of values in the above array
    Returns:
        tp_pmf (NumPy array): An array with the full distribution of probabilities for the numbers of true positives
        fp_pmf (NumPy array): An array with the full distribution of probabilities for the numbers of false positives
    """    
    pb_tp = PoiBin(positive_confidences)
    tp_pmf = pb_tp.pmf(list(range(number_of_values)))
    fp_pmf = tp_pmf[::-1]
    return tp_pmf, fp_pmf


def negatives(negative_confidences, number_of_values):
    """
    Returns the Poisson binomial distributions for true and false negatives.
    
    Arguments:
        negative_confidences (NumPy array): An array with confidence scores for the positive class
        number_of_values (int): The number of values in the above array
    Returns:
        tn_pmf (NumPy array): An array with the full distribution of probabilities for the numbers of true negatives
        fn_pmf (NumPy array): An array with the full distribution of probabilities for the numbers of false negatives
    """        
    reversed_confidences = 1 - negative_confidences
    pb_tn = PoiBin(reversed_confidences)
    tn_pmf = pb_tn.pmf(list(range(number_of_values)))
    fn_pmf = tn_pmf[::-1]
    return tn_pmf, fn_pmf


# Functions used to calculate statistics
def calculate_expectation(results, printout=False):
    """
    Calculates the expected value for a given discrete probability distribution.

    Arguments:
        results (dict): A dictionary representing a discrete probability distribution (keys as variable values, values as probabilities)
        printout (bool): If this is True, a printout of the input distribution is given.
    Returns:
        expectation (float): The expected value of the input distribution.
    """
    expectation = 0.0
    for item in sorted(results.items()):
        x, p = item
        expectation += x*p
        if printout is True:
            print(f"{x:.2f}: {p*100:.2f}%")
    return expectation


def calculate_CI(results, alpha=0.05, type="hdci"):
    """
    Calculates the (1-alpha) credible interval of a given ditribution.

    Arguments:
        results (dict): A dictionary representing a discrete probability distribution (keys as variable values, values as probabilities)
        alpha (float): Signals which CI is to be formed as 1-alpha. Default is 95% CI.
        type (str): Which CI is to be formed. Current options are highest-density credible interval (hdci). The default is "hdci".
    Returns:
        limits (list): A list of two values, a lower and an upper bound for the CI respectively. The bounds are included in the interval, which is the 
                       smallest (connected) interval holding at least (1-alpha) of the probability mass of the distribution.

    """
    if type == "hdci":
        # sorted_items = np.array(sorted(results.items()))
        sorted_items = sorted(results.items())
        a = 0
        b = len(sorted_items)-1
        tail_coverage = 0
        bounds_not_found = True 
        
        while bounds_not_found is True:
            low_p = sorted_items[a][1]
            high_p = sorted_items[b][1]
            if low_p < high_p:
                if tail_coverage + low_p < alpha:
                    tail_coverage += low_p
                    a += 1
                else:                    
                    bounds_not_found = False                
            else:
                if tail_coverage + high_p < alpha:
                    tail_coverage += high_p
                    b -= 1
                else:
                    bounds_not_found = False
        limits = [sorted_items[a][0], sorted_items[b][0]]
        return limits
    else:
        raise NotImplementedError(f"CI type {type} has not (yet) been implemented.")


# Functions to derive different metrics
def accuracy(confidences, zero_division='warn', timeit=False):
    """
    Forms the full distribution of accuracy based on the estimated confusion matrix.

    Arguments:
        confidences (NumPy array): An array with confidence scores
        zero_division ('warn' or float): What to return for an empty list of confidence scores. The default value 'warn' means that 
                                         zero accuracy is assumed but a UserWarning is also raised. Otherwise, the argument is converted
                                         as a floating point number and full probability mass is assigned on the specified value.
        timeit (bool): If this is set to True, the function returns time 
    Returns:
        accuracy_distribution (dict): A dictionary with keys as variable values and values as corresponding probabilities.
        Optionally:
            FFT_time (float): Time spent deriving the Poisson binomial distribution via FFT.
            metric_time (float): Time spent deriving the distribution metric.
    """
    n = len(confidences)
    if n == 0:
        if zero_division == 'warn':
            msg = ("\n\nEmpty list of confidence scores lead to zero division. Accuracy considered to be 0.0 in this case."+
                "\nYou may change this behaviour by setting the parameter 'zero_division' value to 1,"+
                "\nor suppress this warning by setting the parameter value to 0.")
            warnings.warn(msg, UserWarning)
            return {0.0: 1.0}
        else:
            try:
                user_arg = float(zero_division)
            except ValueError as e:
                print(e)
                raise ValueError(f"\nError: {zero_division} cannot be converted to float.")
            if user_arg <= 1.0 and user_arg >= 0.0:
                return {user_arg: 1.0}
            else:
                raise ValueError(f"\nError: {zero_division} outside of range [0.0, 1.0]")

    pos_confs = np.where(confidences >= 0.5, confidences, 1-confidences)
    
    if timeit is True:
        start = time()

    pb = PoiBin(pos_confs)

    if timeit is True:
        mid = time()

    k_values = list(range(n+1))
    pmf = pb.pmf(k_values)    
    accuracy_distribution = {}
    for k in k_values:
        accuracy_distribution[k/n] = pmf[k]
    
    if timeit is True:
        end = time()
        FFT_time = mid-start
        metric_time = end-mid
        return (accuracy_distribution, FFT_time, metric_time)    

    return accuracy_distribution


def precision(confidences, zero_division='warn', timeit=False):
    """
    Forms the full distribution of precision based on the estimated confusion matrix.

    Arguments:
        confidences (NumPy array): An array with confidence scores
        zero_division ('warn' or float): What to return for an empty list of confidence scores. The default value 'warn' means that 
                                         zero precision is assumed but a UserWarning is also raised. Otherwise, the argument is converted
                                         as a floating point number and full probability mass is assigned on the specified value.
        timeit (bool): If this is set to True, the function returns time                                 
    Returns:
        precision_distribution (dict): A dictionary with keys as variable values and values as corresponding probabilities.
        Optionally:
            FFT_time (float): Time spent deriving the Poisson binomial distribution via FFT.
            metric_time (float): Time spent deriving the distribution metric.
    """
    pos_confs, _ = binary_splitter(confidences)
    n = len(pos_confs)
    
    if timeit is True:
        start = time()

    tp_pmf, _ = positives(pos_confs, n+1)

    if timeit is True:
        mid = time()

    precision_distribution = defaultdict(float)

    # Deal with potential zero division according to user preference
    if n==0:
        if zero_division == 'warn':
            msg = ("\nTP and FP are both 0 leading to zero division. Precision is considered to be 0.0 in this case."+
                    " You may change this behaviour by setting the parameter 'zero_division' value to ,"+
                    " or suppress this warning by setting it to 0.")
            warnings.warn(msg, UserWarning)
            precision_distribution[0.0] = 1.0
        else: 
            try:
                user_arg = float(zero_division)
            except ValueError as e:
                print(e)
                raise ValueError(f"\nError: {zero_division} cannot be converted to float.")
            if user_arg <= 1.0 and user_arg >= 0.0:
                precision_distribution[user_arg] = 1.0
            else:
                raise ValueError(f"\nError: {zero_division} outside of range [0.0, 1.0]")
        return precision_distribution
    
    for k in range(n+1):
        precision_distribution[k/n] += tp_pmf[k]

    if timeit is True:
        end = time()
        FFT_time = mid-start
        metric_time = end-mid
        return (precision_distribution, FFT_time, metric_time)    

    return precision_distribution


def recall(confidences, zero_division='warn', timeit=False):
    """
    Forms the full distribution of recall based on the estimated confusion matrix.

    Arguments:
        confidences (NumPy array): An array with confidence scores
        zero_division ('warn' or float): How to deal with possible zero division situations. The default value 'warn' means that 
                                         zero recall is assumed but a UserWarning is also raised. Otherwise, the argument is converted
                                         as a floating point number and probability mass is assigned on the specified value.
        timeit (bool): If this is set to True, the function returns time 
    Returns:
        recall_distribution (dict): A dictionary with keys as variable values and values as corresponding probabilities.
        Optionally:
            FFT_time (float): Time spent deriving the Poisson binomial distribution via FFT.
            metric_time (float): Time spent deriving the distribution metric.
    """
    pos_confs, neg_confs = binary_splitter(confidences)
    n = len(pos_confs)
    m = len(neg_confs)

    if timeit is True:
        start = time()

    tp_pmf, _ = positives(pos_confs, n+1)
    _, fn_pmf = negatives(neg_confs, m+1)

    if timeit is True:
        mid = time()

    recall_distribution = defaultdict(float)
    recall_distribution[0.0] = tp_pmf[0]
    recall_distribution[1.0] = fn_pmf[0]

    # Deal with potential zero division according to user preference
    duplicate = tp_pmf[0] * fn_pmf[0]  

    if zero_division == 'warn':
        msg = ("\n\nTP and FN might both be 0 leading to zero division. Recall considered to be 0.0 in this case."+
                "\nYou may change this behaviour by setting the parameter 'zero_division' value to 1,"+
                "\nor suppress this warning by setting the parameter value to 0.")
        warnings.warn(msg, UserWarning)
        recall_distribution[1.0] -= duplicate
    else: 
        try:
            user_arg = float(zero_division)
        except ValueError as e:
            print(e)
            raise ValueError(f"\nError: {zero_division} cannot be converted to float.")
        if user_arg == 1.0 or user_arg == 0.0:
            recall_distribution[-1*(user_arg-1)] -= duplicate
        else:
            raise ValueError(f"\nError: {zero_division} has to be either 0.0 or 1.0")

    for i in range(1, n+1):
        for j in range(1, m+1):
            recall_distribution[i/(i+j)] += tp_pmf[i] * fn_pmf[j]

    recall_distribution = dict(recall_distribution)

    if timeit is True:
        end = time()
        FFT_time = mid-start
        metric_time = end-mid
        return (recall_distribution, FFT_time, metric_time)    

    return recall_distribution


def f1(confidences, zero_division='warn', timeit=False):
    """
    Forms the full distribution of the F_1 metric based on the estimated confusion matrix.

    Arguments:
        confidences (NumPy array): An array with confidence scores
        zero_division ('warn' or float): How to deal with possible zero division situations. The default value 'warn' means that 
                                         zero F_1 is assumed but a UserWarning is also raised. Otherwise, the argument is converted
                                         as a floating point number and probability mass is assigned on the specified value.
        timeit (bool): If this is set to True, the function returns time 
    Returns:
        f1_distribution (dict): A dictionary with keys as variable values and values as corresponding probabilities.
        Optionally:
            FFT_time (float): Time spent deriving the Poisson binomial distribution via FFT.
            metric_time (float): Time spent deriving the distribution metric.
    """
    f1_distribution = defaultdict(float)

    # Deal with potential zero division according to user preference
    if len(confidences) == 0:
        if zero_division == 'warn':
            msg = ("\n\nEmpty list of confidence scores lead to zero division. F1 considered to be 0.0 in this case."+
                "\nYou may change this behaviour by setting the parameter 'zero_division' value to 1,"+
                "\nor suppress this warning by setting the parameter value to 0.")
            warnings.warn(msg, UserWarning)
            f1_distribution[0.0] = 1.0
        else: 
            try:
                user_arg = float(zero_division)
            except ValueError as e:
                print(e)
                raise ValueError(f"\nError: {zero_division} cannot be converted to float.")
            if user_arg <= 1.0 and user_arg >= 0.0:
                f1_distribution[user_arg] = 1.0
            else:
                raise ValueError(f"\nError: {zero_division} outside of range [0.0, 1.0]")
        return f1_distribution        

    pos_confs, neg_confs = binary_splitter(confidences)
    n = len(pos_confs)
    m = len(neg_confs)

    if timeit is True:
        start = time()
    
    tp_pmf, _ = positives(pos_confs, n+1)
    _, fn_pmf = negatives(neg_confs, m+1)
    
    if timeit is True:
        mid = time()

    f1_distribution[0.0] = tp_pmf[0]
    
    for i in range(1, n+1):
        for j in range(0, m+1):
            f1_distribution[2*i/(i+j+n)] += tp_pmf[i] * fn_pmf[j]

    if timeit is True:
        end = time()
        FFT_time = mid-start
        metric_time = end-mid
        return (f1_distribution, FFT_time, metric_time)    
    
    return f1_distribution


# A quick test script
def main():
    confs = np.array([0.1, 0.3, 0.7, 0.9, 0.8, 0.25, 0.4, 0.67, 0.51, 0.86])
    
    accuracy_result = accuracy(confs)
    printout("accuracy", accuracy_result)

    precision_result = precision(confs)
    printout("precision", precision_result)

    recall_result = recall(confs)
    printout("recall", recall_result)

    f1_result = f1(confs)
    printout("f1", f1_result)


if __name__ == "__main__":
    main()