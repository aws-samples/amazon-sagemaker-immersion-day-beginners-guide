# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from __future__ import print_function

import argparse
import logging
import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd

# DON'T FORGET TO INCLUDE ALL NECESSARY LIBRARIES!!!!!!!!!!!
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score

import xgboost as xgb
# DON'T FORGET TO INCLUDE ALL NECESSARY LIBRARIES!!!!!!!!!!!

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ACTION 1
# Copy your `numeric_features` and your `categorical_features` here
numeric_features = []
categorical_features = []
# END ACTION 1

# ---
# EXAMPLE
"""""
numeric_features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]']

categorical_features = ['Type']
"""""
# END EXAMPLE
# ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters are described here
    # YOU CAN EXTEND THIS SECTION TO YOUR FAVOR, e.g.
    # ADD MORE HYPERPARAMETERS THAT ARE USED IN THIS CONTAINER
    # EXAMPLE: parser.add_argument("--my-example-hp", type=str, default="awesome")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--n_jobs", type=int, default=-1)

    # HELPER SECTION
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    train = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(train) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    # Read DataFrames into array and concatenate them into one DF
    train_data = [pd.read_csv(file) for file in train]
    train_data = pd.concat(train_data)

    # Take the set of files and read them all into a single pandas dataframe
    validation = [os.path.join(args.validation, file) for file in os.listdir(args.validation)]
    if len(validation) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.validation, "train")
        )

    # Read DataFrames into array and concatenate them into one DF
    validation_data = [pd.read_csv(file) for file in validation]
    validation_data = pd.concat(validation_data)
    # END HELPER SECTION

    # ACTION 2
    # The data that will be read in contains all columns, i.e. your
    # features and your target. Remember in preprocessing we set the
    # target column as the very first one in the DataFrame.
    # Task: Create a X_train, y_train, X_val and y_val object using the
    # `train_data` and `validation_data`
    
    # END ACTION 2
    
    # ---
    # EXAMPLE
    """""
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_val, y_val = validation_data.iloc[:, 1:], validation_data.iloc[:, 0]
    """""
    # END EXAMPLE
    # ---
    
    # ACTION 3
    # Copy and paste your model code in here:
    
    # END ACTION 3
    
    # ---
    # EXAMPLE
    """""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)]
    ).fit(X_train, y_train)

    print("model score: %.3f" % model.score(X_val, y_val))
    
    y_pred = model.predict(X_val)
    y_hat = model.predict(X_train)

    print("In Sample")
    print(classification_report(y_train, y_hat, zero_division=1))
    print(confusion_matrix(y_train, y_hat))
    print("Out of Sample")
    print(classification_report(y_val, y_pred, zero_division=1))
    print(confusion_matrix(y_val, y_pred), "\n")

    print(f"train-recall:{recall_score(y_train, y_hat, average='macro', zero_division=True)};")
    print(f"validation-recall:{recall_score(y_val, y_pred, average='macro', zero_division=True)};")
    print(f"train-precision:{precision_score(y_train, y_hat, average='macro', zero_division=True)};")
    print(f"validation-precision:{precision_score(y_val, y_pred, average='macro', zero_division=True)};")
    print(f"train-f1:{f1_score(y_train, y_hat, average='macro', zero_division=True)};")
    print(f"validation-f1:{f1_score(y_val, y_pred, average='macro', zero_division=True)};")

    X = pd.concat(objs=[X_train, X_val], axis=0)
    y = pd.concat(objs=[pd.DataFrame(y_train), pd.DataFrame(y_val)], axis=0)

    model = model.fit(X, y)
    """""
    # END EXAMPLE
    # ---

    # Helper section - no action required!
    # Save the model and config_data to the model_dir so that it can be loaded by model_fn
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Print Success
    logger.info("Saved model!")

def input_fn(input_data, content_type="text/csv"):
    """Parse input data payload.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        content_type (str): A string expected to be 'text/csv'.

    Returns:
        df: pandas.core.frame.DataFrame
    """
    try:
        if "text/csv" in content_type:
            df = pd.read_csv(StringIO(input_data))
            return df
        elif "application/json" in content_type:
            df = pd.read_json(StringIO(input_data.decode("utf-8")))
            return df
        else:
            df = pd.read_csv(StringIO(input_data.decode("utf-8")))
            return df
    except ValueError as e:
        raise logger.error(f"ValueError {e}")


def output_fn(prediction, accept="text/csv"):
    """Format prediction output.

    Args:
        prediction (pandas.core.frame.DataFrame): A DataFrame with predictions.
        accept (str): A string expected to be 'text/csv'.

    Returns:
        df: str (in CSV format)
    """
    return prediction.to_csv(index=False)


def predict_fn(input_data, model):
    """Preprocess input data.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        model: A model

    Returns:
        output: pandas.core.frame.DataFrame
    """
    # Read your model and config file
    output = pd.DataFrame(model.predict(input_data))
    return output


def model_fn(model_dir):
    """Deserialize fitted model.

    This simple function takes the path of the model, loads it,
    deserializes it and returns it for prediction.

    Args:
        model_dir (str): A string that indicates where the model is located.

    Returns:
        model:
    """
    # Load the model and deserialize
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
