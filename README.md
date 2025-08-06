# CBPE-experiments
This code repository holds the scripts for the experiments performed in our paper "Performance Estimation in Binary Classification Using Calibrated Confidence".
To run just experiments 4.1 and 4.2, clone the repository and run the simulations.py script.

To run the TableShift experiments, you will have to download the data from https://tinyurl.com/tableshift. 
Then, unzip the tableshift data into a subfolder named "tableshift_data" in the working directory. 
Finally, run the tableshift.py script.

To run the CNN - Image Data experiment, run the binary_cnn_experiment.py.

There is an additional experiment with simulated data, where we control for covariate shift.
You can perform this experiment by running the binary_classification_metrics.py script.
