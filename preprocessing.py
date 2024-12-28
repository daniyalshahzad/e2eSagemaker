import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import warnings
import numpy as np


# def convert_numeric_to_int(value):
#     ''' Helper function to deal with features that have int, float, strings, and NaNs as categories'''
#     try:
#         # Try to convert to float first to handle cases like '6.0'
#         numeric_value = float(value)
#         return str(numeric_value)
    
#     except (ValueError, TypeError):
#         # Return original value if it's not numeric
#         return value

def main():
    #define the input and output arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    #parser.add_argument('--input_mappings', type=str, required=True)
    parser.add_argument('--output_train', type=str, required=True)
    parser.add_argument('--output_test', type=str, required=True)
    parser.add_argument('--output_txt', type=str, required=True)
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    df = df.drop(columns=['Unnamed: 0', 'y_no'])
    
    train, test = train_test_split(df, test_size=0.2, stratify=df['y_yes'],random_state=42)
    
    column_names = train.columns.tolist()
    column_names.remove('y_yes')
    
    train = pd.concat([train['y_yes'], train.drop(['y_yes'], axis=1)], axis=1)
    test = pd.concat([test['y_yes'], test.drop(['y_yes'], axis=1)], axis=1)
    
    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    
    # Save train and test data without headers
    train.to_csv(args.output_train, index=False, header=False)
    test.to_csv(args.output_test, index=False, header=False)
    
    with open(os.path.join(args.output_txt, 'column_names.json'), 'w') as f:
        json.dump(column_names, f)

if __name__ == '__main__':
    main()