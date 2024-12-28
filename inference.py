import os
import json
import xgboost as xgb
import numpy as np
from io import StringIO
import pickle as pkl
import boto3

bucket = os.getenv('BUCKET')
base_s3_path = os.getenv('OUTPUTPATH')

s3 = boto3.client('s3')

# Read column_names.json
obj = s3.get_object(Bucket=bucket, Key=os.path.join(base_s3_path, 'column_names.json'))
column_names = json.loads(obj['Body'].read().decode('utf-8'))

# def shap_values_helper(shap_dict):
#     ''' Gets a dictionary with features that include one hot encoded features and
#         then combines them into an original feature.
#     '''
    
#     orig_feature_dict = {}
#     for feature in shap_dict:
#         if feature in non_cat_columns:
#             orig_feature_dict[feature] = shap_dict[feature]
            
#         else:
#             last_index = feature.rfind('_')
#             original_feature_name = feature[:last_index]
#             if original_feature_name in orig_feature_dict:
#                 orig_feature_dict[original_feature_name] += shap_dict[feature]
#             else:
#                 orig_feature_dict[original_feature_name] = shap_dict[feature]
                
#     return orig_feature_dict

def model_fn(model_dir):
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    return booster

def input_fn(input_data, content_type):
    """Parse input data payload"""
    print(content_type, ' is the content_type')
    # Convert the input CSV string directly to a numpy array
    input_data = input_data.decode('utf-8')
    input_str = StringIO(input_data)
    array = np.loadtxt(input_str, delimiter=",").reshape(1, -1)

    # Convert numpy array to DMatrix
    dmatrix = xgb.DMatrix(array)
    return dmatrix

def predict_fn(input_data, model):
    """Make predictions using the loaded model"""
    # Make predictions with SHAP values
    prediction = model.predict(input_data).tolist()
    shap_values = model.predict(input_data, pred_contribs=True)[0][:-1].tolist()
    
    #Ensure that size of expected columns is the same as size of shap_values
    if len(common_columns) != len(shap_values):
        return "Size of expected columns is not compatible with the model. (Something is broken)"
              
    shap_dict = dict(zip(column_names, shap_values))

    return {
        'predictions': prediction,
        'shap_values': shap_dict
    }

def output_fn(prediction, accept):
    """Format the prediction output"""
    print(prediction)
    return json.dumps(prediction), 'application/json'
