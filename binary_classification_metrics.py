""" 
Copyright (c) 2025 Juhani Kivimäki
Released under the MIT License. See LICENSE file in project root for details.

This script runs an experiment with simulated data (not included in the CBPE paper).
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from betacal import BetaCalibration
from xgboost import XGBClassifier
from shortcut_estimators import s_all
from utils import calculate_ese, calculate_correlations_all

# Global variables 
TRIALS = 1000
WINDOW = 1000
BINS = 20
PLOT = 'dataset'      # Set this to 'dataset' to visualize the created datasets and 'models' to plot the posterior densities of the models.
TYPE = 'circle'  # To generate data for the linear case, use 'gradient' and for non-linear case, use 'circle'.
RANDOM_SEED = 49
SAVE = True

# Set the label scaling parameter lambda:
LAMBDA = np.log(np.sqrt(2))  

# Initialize the random number generator
rng = np.random.default_rng(RANDOM_SEED)


def setstyle():
    """
    This function is used to set the plotting style.
    """
    plt.style.use("seaborn-v0_8")

    plt.rc("figure", figsize=(10, 10))
    plt.rc("image", cmap='coolwarm')

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12) 
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)



def spherical_to_cartesian(theta):
    """
    Converts n-dimensional spherical coordinates to Cartesian coordinates.
    
    Arguments:
        theta (list or np.array): An n-dimensional list/array of angles in radians. The first element is the radius.
    Returns:
        np.array: An n-dimensional Cartesian coordinate vector.
    """
    n = len(theta)  # Total dimensions
    r = theta[0]
    cartesian = np.zeros(n)
    
    # Compute Cartesian coordinates using the transformation formulas
    cartesian[0] = r * np.cos(theta[1])  # x_1 = r * cos(θ1)
    
    sin_prod_so_far = 1
    for i in range(1, n-1):
        sin_prod_so_far *= np.sin(theta[i])
        cartesian[i] = r * sin_prod_so_far * np.cos(theta[i])
    
    cartesian[-1] = r * sin_prod_so_far * np.sin(theta[-1])  # Last coordinate (x_n)
    
    return cartesian


def create_dataset(n=10000, dim=2, easy_bias=0.5, label_bias=0.5):
    """
    This function is used to create synthetic data in $dim dimensions.

    Arguments:
        n (int): The number of points to sample.
        dim (int): The number of dimensions.
        easy_bias (float): The percentage of easy-to-predict points out of all points. 
        
    Returns:
        X (NumPy array): Coordinates for the generated points.
        y (NumPy array): Labels for the generated points.
        y_prob (NumPy array): Confidence scores used to generate labels for the generated points.
    """
    # Sample distance from the origin for each point using a mixture of five components
    r1 = np.abs(rng.normal(loc = 0, scale = 0.5, size = int(n * easy_bias * (1 - label_bias) * 2/3)))
    r2 = np.abs(rng.normal(loc = 3, scale = 0.5, size = int(n * easy_bias * label_bias)))
    r3 = np.abs(rng.normal(loc = 6, scale = 0.5, size = int(n * easy_bias * (1 - label_bias) * 1/3)))
    r4 = np.abs(rng.normal(loc = np.sqrt(2), scale = 0.5, size = int(n * (1 - easy_bias) // 2)))
    r5 = np.abs(rng.normal(loc = 3 + np.sqrt(2), scale = 0.5, size = int(n * (1 - easy_bias) // 2)))
        
    rs = np.concatenate((r1, r2, r3, r4, r5))
    
    # Sample directional angles
    angles = rng.uniform(0, 2*np.pi, size = (len(rs), dim-1))

    # Change from spherical to cartesian coordinates
    thetas = np.column_stack((rs.T, angles))
    X = np.apply_along_axis(spherical_to_cartesian, axis=1, arr=thetas)

    # Calculate (absolute) distance d from the circle x^2+y^2=9 for each sampled X
    lengths = np.linalg.norm(X, axis=1)
    d = np.abs(lengths - 3)    

    # Map distance d as the probability of positive class using a scaled sigmoid
    y_prob = np.exp(-LAMBDA * d**2)

    # Assign a label for each instance by sampling from a Bernoulli distribution
    y = np.array([rng.binomial(1, prob) for prob in y_prob])

    return X, y, y_prob 


def tableformat(value, total_digits=3):
    # Compute the integer part's length
    value = 100*value
    integer_part = int(value)
    integer_length = len(str(abs(integer_part)))
    
    # Calculate the number of decimal places to ensure total digits
    decimal_places = max(0, total_digits - integer_length)
    
    # Format the number with the calculated decimal places
    formatted_value = f"{value:.{decimal_places}f}"
    
    return formatted_value


# Control the amount of easy and hard to predict samples respectively
portions = [(80000, 20000),   # This is the original distribution used to train the model and calibration mapping
            (20000, 80000)]  # The rest represent shifted distributions with increasing portion of hard samples
            
# Initialize all models used in the experiment
dimensions = [2, 10, 100, 1000]
n_dims = len(dimensions)
splitmap = {0: "ID", 1: "OoD"}

# Reserve containers for temporary data storage
calib_rows = []
uncalib_rows = []
calibration_mappings = []
calibs = np.zeros(n_dims)
og_accs = np.zeros(n_dims)
uc_og_confs = np.zeros(n_dims)
c_og_confs = np.zeros(n_dims)
results_tensor = np.zeros((len(portions), 12, n_dims, 2))

ue_metrics = np.zeros((TRIALS, 4))
ce_metrics = np.zeros((TRIALS, 4))

#------------------#
# BEGIN EXPERIMENT #
#------------------#

setstyle()

for i, dim in enumerate(dimensions):
    print("\nprosessing trials for feature dimensions:", dim)
    for j, portion in enumerate(portions):
        n, m = portion
        eb = n/(n+m)
        X, y, y_prob = create_dataset(n+m, dim=dim, easy_bias=eb, label_bias=0.2)
        if j == 0:
            print(f"Label bias for original data is {np.mean(y)}")
        else:
            print(f"\tfor shifted data is {np.mean(y)}")
        
        if PLOT == 'dataset' and dim == 2:
            plt.subplot(1, 2, j+1)
            positive = y==1
            plt.scatter(x=20, y=20, c='red', label="1")
            plt.scatter(x=10, y=10, c='blue', label="0")
            plot_idx = rng.permutation(X.shape[0])
            colormap = {0: "blue", 1: "red"}
            vcmap = np.vectorize(colormap.get)

            plt.scatter(x=X[plot_idx, 0], y=X[plot_idx, 1], c=vcmap(y[plot_idx]), zorder=1, alpha=0.025)
                                    
            plt.xlim([-8.5, 8.5])
            plt.ylim([-8.5, 8.5])
            
            # Draw the true decision boundary (where p(y|x) = 0.5)
            if TYPE == 'gradient':
                plt.axline((0,0), (1,1), c='black', alpha=0.8, ls='--', label="True decision boundary")
            elif TYPE =='circle':
                angle = np.linspace( 0 , 2 * np.pi , 150 ) 
                delta = np.sqrt(-np.log(0.5) / LAMBDA) 
                r1 = 3 - delta
                r2 = 3 + delta
    
                x1 = r1 * np.cos( angle ) 
                y1 = r1 * np.sin( angle )       
                x2 = r2 * np.cos( angle ) 
                y2 = r2 * np.sin( angle )       

                plt.plot(x1, y1, c='black', alpha=0.8, ls='--', label="True decision boundary")
                plt.plot(x2, y2, c='black', alpha=0.8, ls='--')

            if j == 0:
                plt.title(f"Easy={n}, Hard={m} (no shift)")
            else:
                plt.title(f"Easy={n}, Hard={m}")
            plt.legend(loc="upper left")
        
            if j+1 == 2:
                plt.suptitle("Visualization of covariate shift within the experiment")
                plt.show()

        if j == 0:
        # Split training data, 60 % training, 20 % validation, 20 % testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
            X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED, stratify=y_test)
        else:
            X_test = X
            y_test = y
        n_test_samples = len(y_test)
        test_index = list(range(n_test_samples))

        # Calculate the accuracy of a Bayes optimal classifier
        if TYPE == 'gradient':
            y_hat_bayes = (X_test[:,0] > X_test[:,1]).astype(int)
        elif TYPE == 'circle':
            distances = np.linalg.norm(X_test, axis=1)
            delta = np.sqrt(-np.log(0.5) / LAMBDA) 
            r1 = 3 - delta
            r2 = 3 + delta
            y_hat_bayes = ((distances > r1) & (distances < r2))

        correct_bayes = y_hat_bayes == y_test
        print("\tThe accuracy of the Bayes optimal classifier is", np.mean(correct_bayes))
        ece_bayes = calculate_ese(y, y_prob)
        print("\tThe adaECE of the Bayes optimal classifier is", ece_bayes)

        # To draw the decicion boundaries for each classifier
        if PLOT == 'models' and j==0:
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = plt.subplot(2, 4, 1)
            ax.set_title("Input data")

            # Plot the data points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.1, edgecolors="k")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())

        if j==0:
            model = XGBClassifier(seed=RANDOM_SEED, verbosity=0)
            model.fit(X_train, y_train)
            og_accs[i] = model.score(X_test, y_test)
            
            # To draw the decicion boundaries for each classifier
            if PLOT == 'models':
                ax = plt.subplot(2, 4, i+2)
                DecisionBoundaryDisplay.from_estimator(model, X, grid_resolution=1000, cmap=cm, alpha=0.8, ax=ax, eps=0.5)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())                
                ax.set_title(dimensions[i])
                # ax.set_title(dimensions[i])

        # Calculate actual values for the metrics over the test set
        preds = model.predict(X_test)
        metrics = np.array([accuracy_score(y_test, preds),
                            precision_score(y_test, preds),
                            recall_score(y_test, preds),
                            f1_score(y_test, preds)])

        # Calculate (uncalibrated) probabilities for the positive class
        p_valid = model.predict_proba(X_valid)        
        p_test = model.predict_proba(X_test)
        probas_valid = p_valid[:, 1]
        probas_test = p_test[:, 1]

        if j==0:      
            # Train a calibration mapping
            c_map = IsotonicRegression(out_of_bounds='clip')
            # c_map = BetaCalibration(parameters='abm')
            
            try:
                c_map.fit(probas_valid, y_valid)
            except:
                c_map.fit(probas_valid.reshape(-1, 1), y_valid)
            calibration_mappings.append(c_map)        
                        
        c_map = calibration_mappings[i]
        
        ece_uncalib = calculate_ese(y_test, probas_test, scheme='dynamic')
        calib = c_map.predict(probas_test)
        ece_calib = calculate_ese(y_test, calib, scheme='dynamic')

        for k in range(TRIALS):
            sampled_indices = rng.choice(test_index, size=WINDOW)
            sample_X = X_test[sampled_indices]
            sample_y = y_test[sampled_indices]

            p_test = model.predict_proba(sample_X)
            probas_test = p_test[:, 1]
            calib = c_map.predict(probas_test)       

            ue_metrics[k, :] = s_all(probas_test) 
            ce_metrics[k, :] = s_all(calib)
            
        results_tensor[j,0:12:3,i,0] = metrics        
        results_tensor[j,1:12:3,i,0] = np.mean(ue_metrics, axis=0) - metrics        
        results_tensor[j,2:12:3,i,0] = np.mean(ce_metrics, axis=0) - metrics            
        results_tensor[j,1:12:3,i,1] = np.std(ue_metrics)
        results_tensor[j,2:12:3,i,1] = np.std(ce_metrics)

        results_tensor[j,0,i,1] = ece_uncalib
        results_tensor[j,3,i,1] = ece_calib

        if PLOT == 'models' and j==0:
            plt.tight_layout()
            plt.show()

        uncalib_row = {'n_features': dim, 
                'split': splitmap[j],
                'ACE': tableformat(results_tensor[j,0,i,1]),
                'Acc': tableformat(results_tensor[j,0,i,0]),
                'a_MAE': "$" + tableformat(results_tensor[j,1,i,0]) + "\\pm" + tableformat(2*results_tensor[j,1,i,1]) + "$",
                'Prec': tableformat(results_tensor[j,3,i,0]),
                'p_MAE': "$" + tableformat(results_tensor[j,4,i,0]) + "\\pm" + tableformat(2*results_tensor[j,4,i,1]) + "$",
                'Rec': tableformat(results_tensor[j,6,i,0]),
                'r_MAE': "$" + tableformat(results_tensor[j,7,i,0]) + "\\pm" + tableformat(2*results_tensor[j,7,i,1]) + "$",
                'F$_1$': tableformat(results_tensor[j,9,i,0]),
                'f_MAE': "$" + tableformat(results_tensor[j,10,i,0]) + "\\pm" + tableformat(2*results_tensor[j,10,i,1]) + "$"
                }

        calib_row = {'n_features': dim, 
                'split': splitmap[j],
                'ACE': tableformat(results_tensor[j,3,i,1]),
                'Acc': tableformat(results_tensor[j,0,i,0]),
                'a_MAE': "$" + tableformat(results_tensor[j,2,i,0]) + "\\pm" + tableformat(2*results_tensor[j,2,i,1], total_digits=2) + "$",
                'Prec': tableformat(results_tensor[j,3,i,0]),
                'p_MAE': "$" + tableformat(results_tensor[j,5,i,0]) + "\\pm" + tableformat(2*results_tensor[j,5,i,1], total_digits=2) + "$",
                'Rec': tableformat(results_tensor[j,6,i,0]),
                'r_MAE': "$" + tableformat(results_tensor[j,8,i,0]) + "\\pm" + tableformat(2*results_tensor[j,8,i,1], total_digits=2) + "$",
                'F$_1$': tableformat(results_tensor[j,9,i,0]),
                'f_MAE': "$" + tableformat(results_tensor[j,11,i,0]) + "\\pm" + tableformat(2*results_tensor[j,11,i,1], total_digits=2) + "$"
                }

        uncalib_rows.append(uncalib_row)
        calib_rows.append(calib_row)

uncalib_results = pd.DataFrame(uncalib_rows)
calib_results = pd.DataFrame(calib_rows)

corr_columns = ['ACE', 'a_MAE', 'p_MAE', 'r_MAE', 'f_MAE']

u_corr_data = uncalib_results.loc[:, corr_columns].to_numpy()
c_corr_data = calib_results.loc[:, corr_columns].to_numpy()

u_corrs = calculate_correlations_all(u_corr_data)
c_corrs = calculate_correlations_all(c_corr_data)

print(f"Uncalibrated correlations:\n\t{u_corrs}")    
print(f"Calibrated correlations:\n\t{c_corrs}")

if SAVE is True:
    uncalib_results.to_csv("sim_u_results.csv", index=False)
    calib_results.to_csv("sim_c_results.csv", index=False)

print("All done!")