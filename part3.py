import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def make_dataframe(path):
    return pd.read_csv(path)


def train_multivariate_linear_regressor(df):
    features = df.drop(["NOX"], axis=1).to_numpy()
    nox = df["NOX"].to_numpy()

    lr = LinearRegression()
    return lr.fit(features, nox)


def apply_multivariate_linear_regressor(mlr, test_df):
    test_features = test_df.drop(["NOX"], axis=1).to_numpy()
    test_nox = test_df["NOX"].to_numpy()

    # Predict nox value for each testcase
    prediction = mlr.predict(test_features)

    # TODO: Compare NOX prediction values with actual NOX test values

    return "TODO"


def phase_1():

    # Read data from csv
    d2011 = make_dataframe("pp_gas_emission/gt_2011.csv").drop(["CO"], axis=1)
    d2012 = make_dataframe("pp_gas_emission/gt_2012.csv").drop(["CO"], axis=1)
    d2013 = make_dataframe("pp_gas_emission/gt_2013.csv").drop(["CO"], axis=1)
    d2014 = make_dataframe("pp_gas_emission/gt_2014.csv").drop(["CO"], axis=1)
    d2015 = make_dataframe("pp_gas_emission/gt_2015.csv").drop(["CO"], axis=1)

    # Training set -> 2011 + 2012
    training_set = pd.concat([d2011, d2012])

    # Validation set -> 2013
    validation_set = d2013

    # Test set -> 2014 + 2015
    test_set = pd.concat([d2014, d2015])

    # phase 1a:

    # train mlr on training_set
    mlr = train_multivariate_linear_regressor(training_set)

    # apply mlr on validation set
    baseline = apply_multivariate_linear_regressor(mlr, validation_set)

    # Combine training and validation set
    train_valid_set = pd.concat([training_set, validation_set])

    # train mlr on combined training and validation set
    new_mlr = train_multivariate_linear_regressor(train_valid_set)

    # apply new mlr on test set
    #new_baseline = apply_multivariate_linear_regressor(new_mlr, test_set)

    return


if __name__ == "__main__":

    # Phase 1a:
    # - Using the training set and the original setof features, (Section 4.1)
    #       train a multivariate linear model
    # - apply it on the validation set

    # Phase 1b:
    # - Combining the training and validation sets, re-train a new linear model
    #       and apply the model on the test set
    # - Note down the validation and the test set performances 
    #       for reporting later (e.g.in comparison with other experiments)

    phase_1()