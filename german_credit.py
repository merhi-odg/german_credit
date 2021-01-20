# modelop.schema.0: input_schema.avsc
# modelop.slot.1: in-use

import pandas as pd
import pickle
import numpy as np

# Bias libraries
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias 


# modelop.init
def begin():
    
    global logReg_model, train_encoded_columns
    
    # load pickled logistic regression model
    logReg_model = pickle.load(open("logistic_regression_model.pickle", "rb"))
    
    # Encoded columns generating in training
    train_encoded_columns = [
        'duration_months', 'credit_amount', 'installment_rate',
        'present_residence_since', 'age_years', 'number_existing_credits',
        'number_people_liable', 'checking_status_A11',
        'checking_status_A12', 'checking_status_A13', 'checking_status_A14',
        'credit_history_A30', 'credit_history_A31', 'credit_history_A32',
        'credit_history_A33', 'credit_history_A34', 'purpose_A40',
        'purpose_A41', 'purpose_A410', 'purpose_A42', 'purpose_A43',
        'purpose_A44', 'purpose_A45', 'purpose_A46', 'purpose_A48',
        'purpose_A49', 'savings_account_A61', 'savings_account_A62',
        'savings_account_A63', 'savings_account_A64', 'savings_account_A65',
        'present_employment_since_A71', 'present_employment_since_A72',
        'present_employment_since_A73', 'present_employment_since_A74',
        'present_employment_since_A75', 'debtors_guarantors_A101',
        'debtors_guarantors_A102', 'debtors_guarantors_A103', 'property_A121',
        'property_A122', 'property_A123', 'property_A124',
        'installment_plans_A141', 'installment_plans_A142',
        'installment_plans_A143', 'housing_A151', 'housing_A152',
        'housing_A153', 'job_A171', 'job_A172', 'job_A173', 'job_A174',
        'telephone_A191', 'telephone_A192', 'foreign_worker_A201',
        'foreign_worker_A202', 'gender_female', 'gender_male']


# modelop.score
def action(data):
    
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    
    # remove ground truth if present
    if "label" in data.columns:
        print("True")
        data = data.drop(["label"], axis=1)
    
    # engineer gender from status_sex
    data["gender"] = np.where(
        # females can be under A92 and A95 status_sex codes
        ((data['status_sex'] == "A92") | (data['status_sex'] == "A95")),
        'female', 
        'male')
    
    # drop status_sex from features
    data = data.drop(["status_sex"], axis=1)
    
    print(data.shape)

    # All categorical vars to be encoded
    categorical_columns = [
        'checking_status', 'credit_history', 'purpose', 
        'savings_account', 'present_employment_since', 
        'debtors_guarantors', 'property', 'installment_plans', 
        'housing', 'job', 'telephone', 'foreign_worker', 'gender']

    # encoding categprical vars in data
    encoded_features = pd.get_dummies(data, columns = categorical_columns)
    
    print((encoded_features.columns))
    
    # Missing encoded columns from data
    missing_columns = set(train_encoded_columns) - set(encoded_features.columns)
    
    print(missing_columns)
    
    # Encoding missing categorical feats with 0
    for c in missing_columns:
        encoded_features[c] = 0
        
    print(len(encoded_features.columns))
        
    # Matching order of variables to those used in training
    encoded_features = encoded_features[train_encoded_columns]
    
    # Add prediction column
    data["predicted_score"] = logReg_model.predict(encoded_features)
    
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(scored_labeled_data):
    
    scored_labeled_data = pd.DataFrame(scored_labeled_data)

    # To measure Bias towards gender, filter DataFrame
    # to "score", "label_value" (ground truth), and
    # "gender" (protected attribute)
    
    scored_labeled_data = scored_labeled_data[
        ["score", "label_value", "gender"]
    ]

    # Process DataFrame
    data_processed, _ = preprocess_input_df(scored_labeled_data)

    # Group Metrics
    g = Group()
    xtab, _ = g.get_crosstabs(data_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = g.list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

    # For example:
    """
        attribute_name  attribute_value     tpr     tnr  ... precision
    0   gender          female              0.60    0.88 ... 0.75
    1   gender          male                0.49    0.90 ... 0.64
    """

    # Bias Metrics
    b = Bias()

    # Disparities calculated in relation gender for "male" and "female"
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=data_processed,
        ref_groups_dict={'gender': 'male'},
        alpha=0.05, mask_significance=True
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = b.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ['attribute_name', 'attribute_value'] + calculated_disparities]

    # For example:
    """
        attribute_name	attribute_value    ppr_disparity   precision_disparity
    0   gender          female             0.714286        1.41791
    1   gender          male               1.000000        1.000000
    """

    output_metrics_df = disparity_metrics_df  # or absolute_metrics_df

    # Output a JSON object of calculated metrics
    yield output_metrics_df.to_dict(orient="records")
