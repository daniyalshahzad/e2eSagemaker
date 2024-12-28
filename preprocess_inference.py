import joblib
import os
import json
import pandas as pd
import numpy as np
import io
from io import BytesIO
import csv
import logging
from sklearn.preprocessing import StandardScaler
import datetime
import boto3

logging.basicConfig(level=logging.INFO)

# Define the S3 bucket and base path

bucket = os.getenv('BUCKET')
base_s3_path = os.getenv('OUTPUTPATH')

s3 = boto3.client('s3')

# Read common_columns.json
obj = s3.get_object(Bucket=bucket, Key=os.path.join(base_s3_path, 'column_names.json'))
column_names = json.loads(obj['Body'].read().decode('utf-8'))

# def convert_numeric_to_int(value):
#     ''' Helper function to deal with features that have int, float, strings, and NaNs as categories'''
#     try:
#         # Try to convert to float first to handle cases like '6.0'
#         numeric_value = float(value)
#         return str(numeric_value)
    
#     except (ValueError, TypeError):
#         # Return original value if it's not numeric
#         return value

def convert_to_csv(data):
    logging.info("Converting preprocessed data to CSV format...")
    try:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
        csv_data = output.getvalue()
        logging.info("CSV conversion completed successfully.")
        return csv_data
    except Exception as e:
        logging.error(f"Error during CSV conversion: {e}")
        raise e
        
def preprocess(data):
    # Implement your preprocessing logic here
    df = pd.Series(df).to_frame().T
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    #Add any other preprocess for inferencing code

    df = df[column_names] #Will fail if any column was missing. Also ensures columns are in the same order
    preprocessed_data = df.values.tolist()
    return preprocessed_data

#The 4 functions below are always required (AWS specific)

def model_fn(model_dir):
    # This is preprocessing container, no model needed
    return None

def input_fn(request_body, request_content_type):
    if 'application/json' in request_content_type:
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    preprocessed_data = preprocess(input_data)
    preprocessed_csv = convert_to_csv(preprocessed_data)
    return preprocessed_csv

def output_fn(prediction, content_type):
    logging.info("Predicting the preprocess output :)")
    return prediction, 'text/csv'
