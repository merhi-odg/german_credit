# german_credit
A Logistic Regression Python model to predict likelihood of loan default.
Model was trained on the German Credit Data dataset.
logistic_regression_model.pickle is the trained model artifact.
A sample input to the scoring function is included (`input_data.csv`)

Model code includes a metrics function used to compute disparity metrics (Bias).
The metrics function expects a DataFrame with three columns: `score` (predicted), `label_value` (actual), and `gender` (protected attribute).
A sample input to the metrics function is included (`german_data_scored_labeled.csv`)
