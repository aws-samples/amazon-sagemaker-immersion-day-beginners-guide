# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Feature engineers the dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

# DON'T FORGET TO INCLUDE ALL NECESSARY LIBRARIES!!!!!!!!!!!
from sklearn.model_selection import train_test_split
# DON'T FORGET TO INCLUDE ALL NECESSARY LIBRARIES!!!!!!!!!!!

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ACTION 1
# Copy your `feature_column_names` and your `label_column` here
feature_columns_names = []
label_column = ''
# END ACTION 1

# ---
# EXAMPLE
"""""
feature_columns_names = [
    'UDI',
    'Product ID',
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]']
label_column = 'Failure Type'
"""""
# END EXAMPLE
# ---

if __name__ == "__main__":
    # Log what's happening and parse the inputs arguments
    # given to this processing job (here: input-data)
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    # Helper section:
    # The base directory of the Docker container is set
    # A folder for data output is generated
    # And the input_data argument is broken down into `bucket` and `key`
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    # Second helper section:
    # The file is downloaded from S3 and loaded as a DataFrame into the container
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/maintenance_dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)
    
    # ACTION 2
    # Put your preprocessing code here. In our example it is a train_test_split only
    # Feel free to add more transformations to your data
    
    # END ACTION 2
    
    # ---
    # EXAMPLE
    """""
    X_train, X_val, y_train, y_val = train_test_split(
        df[feature_columns_names],
        df[label_column],
        random_state=42,
        train_size=0.8,
        shuffle=True,
        stratify=df[label_column])
    
    train = pd.concat(objs=[y_train, X_train], axis=1)
    validation = pd.concat(objs=[y_val, X_val], axis=1)
    """""
    # END EXAMPLE
    # ---
    
    # Here the 2 files `training` and `validation` will be saved in our Docker container
    # By defining the `output` in our `ScriptProcessor` these files will be taken and
    # written to S3.
    logger.info("Writing classification out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", index=False)