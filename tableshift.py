"""
Copyright (c) 2025 Juhani Kivimäki
Released under the MIT License. See LICENSE file in project root for details.

This script runs the TableShift benchmark (Section 4.3) presented in the CBPE paper.
"""

import os
import pickle
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from pathlib import Path
from sklearnex import patch_sklearn
patch_sklearn()

from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
# from betacal import BetaCalibration

from shortcut_estimators import s_all  # For faster computation times, we use the shortcut estimators
from utils import setstyle, calculate_ese, calculate_correlations_all

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import warnings
warnings.filterwarnings("ignore")  

# Global variables
PATH = Path.cwd()
CORES = multiprocessing.cpu_count() - 1
CV = 6
N_ITER = 120
SEED = 42

OPTIMIZE = False   # Set this to True to enable Bayesian hyperparameter optimization
TRAIN = False      # Set this to True to enable model training 
CALIBRATE = False  # Set this to True to tarin calibration mappings
SAVE = False       # Set this to True to save resulting parameter grid/model/calibrator/results 
TRIALS = 1000      # Set this to 0 to estimate performance over the whole dataset 
WINDOW = 500


# Define hyperparameter grid to be used in Bayesian optimization
XGB_GRID = {'learning_rate': Real(1e-5, 1.0, prior='log-uniform'),
          'max_depth': Integer(3, 10),
          'min_child_weight': Real(1e-8, 1e5, prior='log-uniform'),
          'subsample': Real(0.1, 1.0),
          'colsample_bytree': Real(0.5, 1.0),
          'colsample_bylevel': Real(0.5, 1.0),
          'gamma': Real(1e-8, 1e2, prior='log-uniform'),
          'reg_lambda': Real(1e-8, 1e2, prior='log-uniform'),
          'reg_alpha': Real(1e-8, 1e2, prior='log-uniform'),
          'max_bin': Categorical([128, 256, 512])
          }

"""
LGBM_GRID = {'learning_rate': Real(1e-5, 1.0, prior='log-uniform'),
          'min_child_samples': Categorical([1, 2, 4, 8, 16, 32, 64]),
          'min_child_weight': Real(1e-8, 1e5, prior='log-uniform'),
          'subsample': Real(0.5, 1.0),
          'max_depth': Categorical([-1] + list(range(1, 32))),
          'colsample_bytree': Real(0.5, 1.0),
          'colsample_bynode': Real(0.5, 1.0),
          'reg_lambda': Real(1e-8, 1e2, prior='log-uniform'),
          'reg_alpha': Real(1e-8, 1e2, prior='log-uniform'),
          }
"""
          
DATASETS = ['acsfoodstamps',
            'acsincome',
            'acsunemployment',
            'assistments',
            'brfss_blood_pressure',
            'brfss_diabetes',
            'college_scorecard',
            'diabetes_readmission']

SPLITS = ['id', 'ood']

RNG = np.random.default_rng(seed = SEED)


def tableformat(value, total_digits=3):
    """This is a utility function used to format the results in Table 1"""
    # Compute the integer part's length
    value = 100*value
    integer_part = int(value)
    integer_length = len(str(abs(integer_part)))
    
    # Calculate the number of decimal places to ensure total digits
    decimal_places = max(0, total_digits - integer_length)
    
    # Format the number with the calculated decimal places
    formatted_value = f"{value:.{decimal_places}f}"
    
    return formatted_value


def optimize(X_train, y_train, X_valid, y_valid, dataset, save=SAVE):  
    """This function is used in the Bayesian hyperparameter optimization"""
    np = CORES//CV
    nw = CV*np
    header = f"\n\nBegin parameter search for dataset {dataset}. Searching for {N_ITER} iterations (candidates):"                  
    print(header)
    print("-" * (len(header)-2))
    print(f"\nUsing {CV*np}/{nw+1} CPU threads in model training, with")
    print(f"{CV}-fold cross-validation and sampling {np} parameter sets per iteration.")
    
    cv = StratifiedKFold(n_splits=CV, shuffle=True)
    acc_scorer = make_scorer(accuracy_score)
    search = BayesSearchCV(# estimator=LGBMClassifier(seed=SEED, 
                                                    # early_stopping_round=3,                                                   
                                                    # verbosity=-1),                        
                        estimator=XGBClassifier(objective="binary:logistic",
                                                    eval_metric="logloss",
                                                    seed=SEED,
                                                    early_stopping_rounds=5,
                                                    tree_method="hist", 
                                                    verbosity=0),
                        search_spaces=XGB_GRID,
                        n_iter=N_ITER,
                        scoring=acc_scorer,
                        n_jobs=nw,
                        n_points=np,
                        cv=cv,
                        verbose=0)    
    search.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    new_params = search.best_params_
    print("\nBest parameters are:")
    for item in new_params.items():
        print("\t", item[0], ":", item[1])
    print(f"with (CV) accuracy: {search.best_score_}, found on iteration {search.best_index_}")
    if save is True:
        print("Saving optimal parameters.")
        param_path = PATH / "XGBoost_params" / f"{dataset}.pkl"
        with open(param_path, 'wb') as f:
            pickle.dump(new_params, f)      
    return new_params


def train(params, X_train, y_train, dataset, save=SAVE):
    """This function is used to train the models"""
    header = f"\nTraining model for dataset {dataset}."                  
    print(header)
    # model = LGBMClassifier(seed=SEED,
    #                        verbosity=0,
    #                        **params)
    model = XGBClassifier(objective="binary:logistic",
                          eval_metric="logloss",
                          tree_method="hist",     
                          verbosity=0,                     
                          **params)
    
    # Retrain the model with all available data and save as a pickle object
    model.fit(X_train, y_train)
    if save is True:
        print("Saving trained model.")
        model_path = PATH / "XGBoost_models" / f"{dataset}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model


def main():
    # Initialize containers to collect all data
    u_row_list = []
    c_row_list = []

    # Mappings used in text formatting
    splitmap = {'id': 'ID', 'ood': 'OoD'}
    ds_map = {'acsfoodstamps': 'Food Stamps',
                'acsincome': 'Income',
                'acsunemployment': 'Unemployment',
                'assistments': 'ASSISTments',
                'brfss_blood_pressure': 'Hypertension',
                'brfss_diabetes': 'Diabetes',
                'college_scorecard': 'College Scorecard',
                'diabetes_readmission': 'Hospital Readmission'}    

    # Iterate over all available datasets
    for i, ds in enumerate(DATASETS):
        print("\nProcessing dataset:", ds)

        # Train the models if they are not already trained
        if TRAIN is True:
            X_train = pd.read_csv(f"tableshift_data/{ds}_train_features.csv", index_col=0)
            y_train = pd.read_csv(f"tableshift_data/{ds}_train_labels.csv", index_col=0).iloc[:, 0].to_numpy()
            X_valid = pd.read_csv(f"tableshift_data/{ds}_validation_features.csv", index_col=0)
            y_valid = pd.read_csv(f"tableshift_data/{ds}_validation_labels.csv", index_col=0).iloc[:, 0].to_numpy()

            # Optimize the hyperparameters if they have not been optimized already
            if OPTIMIZE:
                params = optimize(X_train, y_train, X_valid, y_valid, ds)
                model = train(params, X_train, y_train, ds)
            else:
                param_path = PATH / "XGBoost_params" / f"{ds}.pkl"
                with open(param_path, 'rb') as f:
                    params = pickle.load(f)
                model = train(params, X_train, y_train, ds)

        # Otherwise, just load the model
        else:
            model_path = PATH / "XGBoost_models" / f"{ds}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # Train the calibration mapping if it hasn't been trained already
        if CALIBRATE is True:
            print("Training calibrator.")
            if TRAIN is False:
                X_valid = pd.read_csv(f"tableshift_data/{ds}_validation_features.csv", index_col=0)
                y_valid = pd.read_csv(f"tableshift_data/{ds}_validation_labels.csv", index_col=0).iloc[:, 0].to_numpy()
            
            u_probas = model.predict_proba(X_valid)[:, 1]
            u_ece = calculate_ese(y_valid, u_probas, scheme="dynamic")
            print("ACE (uncalibrated):", u_ece)

            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')   
            calibrator.fit(u_probas, y_valid)

            c_probas = calibrator.predict(u_probas)
            c_ece = calculate_ese(y_valid, c_probas, scheme="dynamic")
            print("ACE (calibrated):", c_ece)

            # Save the calibration mappings if wanted
            if SAVE is True:
                print("Saving trained calibrator.")
                model_path = PATH / "Tableshift_calibrators" / f"{ds}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(calibrator, f)
        # Otherwise, just load the calibration mapping
        else:
            model_path = PATH / "Tableshift_calibrators" / f"{ds}.pkl"
            with open(model_path, 'rb') as f:
                calibrator = pickle.load(f)

        if TRIALS > 0:
            # Perform trials for both ID and OoD splits
            for split in SPLITS:
                # Read the data
                print(f"Running {TRIALS} trials on {split} test set with monitoring window size = {WINDOW}.")
                X_test = pd.read_csv(f"tableshift_data/{ds}_{split}_test_features.csv", index_col=0)
                y_test = pd.read_csv(f"tableshift_data/{ds}_{split}_test_labels.csv", index_col=0).iloc[:, 0].to_numpy()    
                n_test = len(y_test)
                
                # Make predictions and get uncalibrated and calibrated confidence scores
                preds = model.predict(X_test)
                u_probas = model.predict_proba(X_test)[:, 1]
                c_probas = calibrator.predict(u_probas)

                # Calculate the true metric values
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                f_one = f1_score(y_test, preds)

                # Calculate calibration errors (as ACE)
                u_ece = calculate_ese(y_test, u_probas, scheme="dynamic")
                c_ece = calculate_ese(y_test, c_probas, scheme="dynamic")

                # Gather a random sample of size WINDOW and estimate the performance metrics using the shortcut estimators
                results = np.empty((TRIALS, 4, 2))
                for trial in range(TRIALS):
                    indexes = RNG.integers(0, n_test, WINDOW)
                    X = X_test.iloc[indexes, :]
                    y = y_test[indexes]

                    preds = model.predict(X)
                    u_probas = model.predict_proba(X)[:, 1]
                    c_probas = calibrator.predict(u_probas)
       
                    results[trial, :, 0] = s_all(u_probas)
                    results[trial, :, 1] = s_all(c_probas)

                # Aggregate the results over all trials
                u_est = np.nanmean(results[:, :, 0], axis = 0)
                u_dev = 2*np.nanstd(results[:, :, 0], axis = 0)

                c_est = np.nanmean(results[:, :, 1], axis = 0)
                c_dev = 2*np.nanstd(results[:, :, 1], axis = 0)

                print(f"Accuracy: True={acc:.3f}, uncalibrated estimate={u_est[0]:.3f} +/- {u_dev[0]:.3f}, calibrated estimate={c_est[0]:.3f} +/- {c_dev[0]:.3f}")
                print(f"Precision: True={prec:.3f}, uncalibrated estimate={u_est[1]:.3f} +/- {u_dev[1]:.3f}, calibrated estimate={c_est[1]:.3f} +/- {c_dev[1]:.3f}")
                print(f"Recall: True={rec:.3f}, uncalibrated estimate={u_est[2]:.3f} +/- {u_dev[2]:.3f}, calibrated estimate={c_est[2]:.3f} +/- {c_dev[2]:.3f}")
                print(f"F1: True={f_one:.3f}, uncalibrated estimate={u_est[3]:.3f} +/- {u_dev[3]:.3f}, calibrated estimate={c_est[3]:.3f} +/- {c_dev[3]:.3f}")
                print(f"Calibration errors for uncalibrated={u_ece:.4f}, calibrated={c_ece:.4f}")
                
                # Format and store the results 
                u_row = {'dataset': ds_map[ds], 
                        'split': splitmap[split],
                        'ACE': tableformat(u_ece),
                        'Acc': tableformat(acc),
                        'a_MAE': rf"${tableformat(np.abs(acc - u_est[0]))}\pm{tableformat(u_dev[0], total_digits=2)}$",
                        'Prec': tableformat(prec),
                        'p_MAE': rf"${tableformat(np.abs(prec - u_est[1]))}\pm{tableformat(u_dev[1], total_digits=2)}$",
                        'Rec': tableformat(rec),                     
                        'r_MAE': rf"${tableformat(np.abs(rec - u_est[2]))}\pm{tableformat(u_dev[2], total_digits=2)}$",
                        'F$_1$': tableformat(f_one),
                        'f_MAE': rf"${tableformat(np.abs(f_one - u_est[3]))}\pm{tableformat(u_dev[3], total_digits=2)}$"
                        }

                c_row = {'dataset': ds_map[ds],
                        'split': splitmap[split],
                        'ACE': tableformat(c_ece),
                        'Acc': tableformat(acc),
                        'a_MAE': rf"${tableformat(np.abs(acc - c_est[0]))}\pm{tableformat(c_dev[0], total_digits=2)}$",                     
                        'Prec': tableformat(prec),
                        'p_MAE': rf"${tableformat(np.abs(prec - c_est[1]))}\pm{tableformat(c_dev[1], total_digits=2)}$",                     
                        'Rec': tableformat(rec),                     
                        'r_MAE': rf"${tableformat(np.abs(rec - c_est[2]))}\pm{tableformat(c_dev[2], total_digits=2)}$",
                        'F$_1$': tableformat(f_one),
                        'f_MAE': rf"${tableformat(np.abs(f_one - c_est[3]))}\pm{tableformat(c_dev[3], total_digits=2)}$"
                        }

                u_row_list.append(u_row)
                c_row_list.append(c_row)

        # If TRIALS == 0, estimate performance (once) using the whole dataset
        else:
            for split in SPLITS:
                print(f"Testing on the whole {split} test set.")
                X_test = pd.read_csv(f"tableshift_data/{ds}_{split}_test_features.csv", index_col=0)
                y_test = pd.read_csv(f"tableshift_data/{ds}_{split}_test_labels.csv", index_col=0).iloc[:, 0].to_numpy()  

                preds = model.predict(X_test)
                u_probas = model.predict_proba(X_test)[:, 1]
                c_probas = calibrator.predict(u_probas)
                
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds)
                rec = recall_score(y_test, preds)
                f_one = f1_score(y_test, preds)
                ue_acc, ue_prec, ue_rec, ue_f_one = s_all(u_probas) 
                ce_acc, ce_prec, ce_rec, ce_f_one = s_all(c_probas) 

                u_ece = calculate_ese(y_test, u_probas, scheme="dynamic")
                c_ece = calculate_ese(y_test, c_probas, scheme="dynamic")

                print(f"Accuracy: True={acc:.3f}, uncalibrated estimate={ue_acc:.3f}, calibrated estimate={ce_acc:.3f}")
                print(f"Precision: True={prec:.3f}, uncalibrated estimate={ue_prec:.3f}, calibrated estimate={ce_prec:.3f}")
                print(f"Recall: True={rec:.3f}, uncalibrated estimate={ue_rec:.3f}, calibrated estimate={ce_rec:.3f}")
                print(f"F1: True={f_one:.3f}, uncalibrated estimate={ue_f_one:.3f}, calibrated estimate={ce_f_one:.3f}")
                print(f"\nCalibration errors for the whole ID test set: uncalibrated={u_ece:.4f}, calibrated={c_ece:.4f}")

                u_row = {'dataset': ds_map[ds], 
                        'split': splitmap[split],
                        'ACE': tableformat(u_ece),
                        'Acc': tableformat(acc),
                        'a_MAE': tableformat(np.abs(acc - ue_acc)),
                        'Prec': tableformat(prec),
                        'p_MAE': tableformat(np.abs(prec - ue_prec)),
                        'Rec': tableformat(rec),
                        'r_MAE': tableformat(np.abs(rec - ue_rec)),
                        'F$_1$': tableformat(f_one),
                        'f_MAE': tableformat(np.abs(f_one - ue_f_one)),
                        }

                c_row = {'dataset': ds_map[ds],
                        'split': splitmap[split],
                        'ACE': tableformat(c_ece),
                        'Acc': tableformat(acc),
                        'a_MAE': tableformat(np.abs(acc - ce_acc)),
                        'Prec': tableformat(prec),
                        'p_MAE': tableformat(np.abs(prec - ce_prec)),
                        'Rec': tableformat(rec),
                        'r_MAE': tableformat(np.abs(rec - ce_rec)),
                        'F$_1$': tableformat(f_one),
                        'f_MAE': tableformat(np.abs(f_one - ce_f_one)),
                        }

                u_row_list.append(u_row)
                c_row_list.append(c_row)

        # Store calibration mapping for plotting            
        setstyle()
        xs = np.linspace(0, 1, 1001)
        plt.subplot(2, 4, i+1)
        plt.plot(xs, calibrator.predict(xs))
        plt.plot(xs, xs, 'k--')
        plt.title(ds)

    # Convert result lists into pandas dataframes
    u_results = pd.DataFrame(u_row_list)
    c_results = pd.DataFrame(c_row_list)

    # Calculate correlations
    corr_columns = ['ACE', 'a_MAE', 'p_MAE', 'r_MAE', 'f_MAE']
    u_corr_data = u_results.loc[:, corr_columns].to_numpy()
    c_corr_data = c_results.loc[:, corr_columns].to_numpy()

    u_corrs = calculate_correlations_all(u_corr_data)
    c_corrs = calculate_correlations_all(c_corr_data)

    print(f"Uncalibrated correlations:\n\tID:\n\t{u_corrs}")
    print(f"Calibrated correlations:\n\tID:\n\t{c_corrs}")

    # Write results into a .csv file (if wanted)
    if SAVE is True:
        if TRIALS > 0:
            u_results.to_csv("u_results_sample.csv", index=False)
            c_results.to_csv("c_results_sample.csv", index=False)
        else:
            u_results.to_csv("u_results.csv", index=False)
            c_results.to_csv("c_results.csv", index=False)

    print(u_results.head())
    print("All done!")
    # Plot the calibration mappings for fun
    plt.suptitle("Calibrators")
    plt.show()    


if __name__ == "__main__":
    main()
