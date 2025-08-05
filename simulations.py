"""
Copyright (c) 2025 Juhani Kivimäki
Released under the MIT License. See LICENSE file in project root for details.

This script runs experiments 
4.1 "Convergence of the Shortcut Estimators" and 
4.2 "Quality of the Confidence Intervals" 
presented in the CBPE paper.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from estimators import accuracy, precision, recall, f1, calculate_expectation, calculate_CI
from shortcut_estimators import s_recall, s_f1  
from utils import setstyle
from time import time

import warnings
warnings.filterwarnings("ignore")

TRIALS = 10000  # The number of trials to perform

rng = np.random.default_rng(seed=49)
setstyle()

sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])  # The array defining the monitoring window sizes

ns = len(sizes)
hits = np.zeros((ns, 4, 2))
results = np.zeros((TRIALS, ns, 2))  # A placeholder for the data gathered in all trials for Experiment 4.1
final_results = np.zeros((ns, 4))    # A placeholder for the aggregated data used to plot Figure 1 in Experiment 4.1
times = np.zeros((TRIALS, 2))        # A (redundant) placeholder for timing data
genesis = time()
beta_lower = 0.1  # Lower bound for the Beta parameter range
beta_upper = 10   # Upper bound for the Beta parameter range

max_recall = 0
min_recall = 0
max_f1 = 0
min_f1 = 0

for s, size in enumerate(sizes):
    print(f"\nAverage processing times (s) for chunk size {size}:")
    for trial in range(TRIALS):
        # Sample Beta distribution parameters for this trial
        a, b = rng.uniform(beta_lower, beta_upper, size=2)
        
        # Sample $size$ confidence score instances from the Beta distribution
        batch = rng.beta(a=a, b=b, size=size)

        # Sample labels for each instance using the confidence score as the parameter
        labels = rng.binomial(n=1, p=batch, size=size)

        # Convert confidence scores to predictions (using threshold t=0.5)
        preds = np.round(batch, decimals=0)


        # Estimate the 90% and 95% confidence bounds for each metric and check whether the true value falls within the estimated confidence interval

        # accuracy
        true_metric = accuracy_score(labels, preds)
        dist = accuracy(batch)
        limits_95 = calculate_CI(dist)        
        limits_90 = calculate_CI(dist, alpha=0.1)        
        if true_metric >= limits_95[0] and true_metric <= limits_95[1]:
            hits[s, 0, 0] += 1
        if true_metric >= limits_90[0] and true_metric <= limits_90[1]:
            hits[s, 0, 1] += 1

        # precision
        true_metric = precision_score(labels, preds)
        dist = precision(batch)
        limits_95 = calculate_CI(dist)        
        limits_90 = calculate_CI(dist, alpha=0.1)        
        if true_metric >= limits_95[0] and true_metric <= limits_95[1]:
            hits[s, 1, 0] += 1
        if true_metric >= limits_90[0] and true_metric <= limits_90[1]:
            hits[s, 1, 1] += 1

        # For recall and f1, also measure the error of the shortcut estimator to keep track of convergence

        # recall
        true_metric = recall_score(labels, preds)
        start = time()
        sr = s_recall(batch)  
        dist = recall(batch)
        metric = calculate_expectation(dist)
        dif = metric - sr
        end = time()
        times[trial, 0] = end-start
        results[trial, s, 0] = dif
        limits_95 = calculate_CI(dist)        
        limits_90 = calculate_CI(dist, alpha=0.1)        
        if true_metric >= limits_95[0] and true_metric <= limits_95[1]:
            hits[s, 2, 0] += 1
        if true_metric >= limits_90[0] and true_metric <= limits_90[1]:
            hits[s, 2, 1] += 1
        if dif < min_recall:
            min_recall = dif
        if dif > max_recall:
            max_recall = dif   

        # f1
        true_metric = f1_score(labels, preds)
        start = time()
        sf1 = s_f1(batch)  
        dist = f1(batch)
        metric = calculate_expectation(dist)
        limits = calculate_CI(dist)
        dif = metric - sf1
        end = time()
        times[trial, 1] = end-start
        results[trial, s, 1] = dif
        limits_95 = calculate_CI(dist)        
        limits_90 = calculate_CI(dist, alpha=0.1)        
        if true_metric >= limits_95[0] and true_metric <= limits_95[1]:
            hits[s, 3, 0] += 1
        if true_metric >= limits_90[0] and true_metric <= limits_90[1]:
            hits[s, 3, 1] += 1
        if dif < min_f1:
            min_f1 = dif
        if dif > max_f1:
            max_f1 = dif
        
    print(f"\tRecall = {np.mean(times[:,0]):.2f}")
    print(f"\tF_1 = {np.mean(times[:,1]):.2f}")
    
    # Aggregate results for trial 4.1

    final_results[s, 0] = np.mean(results[:, s, 0])
    final_results[s, 1] = np.std(results[:, s, 0])
    final_results[s, 2] = np.mean(results[:, s, 1])
    final_results[s, 3] = np.std(results[:, s, 1])
    
grande_finale = time()
print(f"\nTotal time = {(grande_finale - genesis)//60} min. {(grande_finale - genesis)%60} s.")

print("Numerical values for recall:")
for i, value in enumerate(final_results[:, 0]):
    print(f"N = {sizes[i]}, Value = {value:.6f}")

print("Numerical values for F1:")
for i, value in enumerate(final_results[:, 2]):
    print(f"N = {sizes[i]}, Value = {value:.6f}")

print(f"Recall approximation error bounds are [{min_recall:.6f}, {max_recall:.6f}]")
print(f"F1 approximation error bounds are [{min_f1:.6f}, {max_f1:.6f}]")

# Plot the results

ticks = [s for s in sizes for _ in range(TRIALS)]

recall_dist = ["recall" for _ in ticks]
f1_dist = [r"F$_1$" for _ in ticks]

recall_dataf = pd.DataFrame({"ticks": ticks, "errors": results[:, :, 0].flatten('F'), "Distribution": recall_dist})
f1_dataf = pd.DataFrame({"ticks": ticks, "errors": results[:, :, 1].flatten('F'), "Distribution": f1_dist})
dataf = pd.concat((recall_dataf, f1_dataf))

ax = sns.lineplot(data=dataf, x="ticks", y="errors", estimator="mean", hue="Distribution", errorbar=("sd", 1))
ax.set(xlabel="Monitoring window size", ylabel="Approximation error", xscale="log", xlim=[10, 1000], xticks=sizes, ylim=[-0.01, 0.01])

plt.hlines(y=0, xmin=10, xmax=1000, colors="black", linestyle='--')
plt.title(f"Mean approximation error of the shortcut estimator") # \n(over {TRIALS} trials)")
plt.legend()
plt.show()
plt.close()

setstyle()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(sizes[9:], 100 * hits[9:, 0, i] / TRIALS, label="Accuracy")
    plt.plot(sizes[9:], 100 * hits[9:, 1, i] / TRIALS, label="Precision")
    plt.plot(sizes[9:], 100 * hits[9:, 2, i] / TRIALS, label="Recall")
    plt.plot(sizes[9:], 100 * hits[9:, 3, i] / TRIALS, label=r"F$_1$")
    if i == 0:
        plt.hlines(y=95, xmin=100, xmax=1000, colors="black", linestyle='--', label="Target (95%)")
        plt.ylim([90, 100])
    else:
        plt.hlines(y=90, xmin=100, xmax=1000, colors="black", linestyle='--', label="Target (90%)")
        plt.ylim([80, 100])

    plt.xlabel("Monitoring window size")
    plt.ylabel("Actual coverage (%)")
    plt.xlim([100, 1000])
    plt.xticks(sizes[9:])
    plt.title(f"{95-5*i}% CI")
    plt.legend()

plt.suptitle(f"Validity of the confidence intervals") #\n(over {TRIALS} trials)")
plt.show()
