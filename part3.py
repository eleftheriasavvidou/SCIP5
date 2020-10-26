import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


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

    # z-score normalisation
    test_nox = (test_nox - np.mean(test_nox)) / np.std(test_nox)
    prediction = (prediction - np.mean(prediction)) / np.std(prediction)

    # Spearman Rank Correlation
    sc, _ = spearmanr(test_nox, prediction)

    # Mean Absolute Error
    mae = sum(abs(prediction - test_nox)) / len(prediction)

    # R^2
    r2 = r2_score(test_nox, prediction)

    return (sc, mae, r2)


def main():

    # Read data from csv
    d2011 = make_dataframe("pp_gas_emission/gt_2011.csv").drop(["CO"], axis=1)
    d2012 = make_dataframe("pp_gas_emission/gt_2012.csv").drop(["CO"], axis=1)
    d2013 = make_dataframe("pp_gas_emission/gt_2013.csv").drop(["CO"], axis=1)
    d2014 = make_dataframe("pp_gas_emission/gt_2014.csv").drop(["CO"], axis=1)
    d2015 = make_dataframe("pp_gas_emission/gt_2015.csv").drop(["CO"], axis=1)


    ###########################################################################

    # Phase 1a:
    # - Using the training set and the original setof features, (Section 4.1)
    #       train a multivariate linear model
    # - apply it on the validation set

    # Phase 1b:
    # - Combining the training and validation sets, re-train a new linear model
    #       and apply the model on the test set
    # - Note down the validation and the test set performances 
    #       for reporting later (e.g.in comparison with other experiments)

    ###########################################################################


    # Training set -> 2011 + 2012
    training_set = pd.concat([d2011, d2012])

    # Validation set -> 2013
    validation_set = d2013

    # Test set -> 2014 + 2015
    test_set = pd.concat([d2014, d2015])

    # train mlr on training_set
    mlr = train_multivariate_linear_regressor(training_set)

    # apply mlr on validation set
    validation_set_baseline = apply_multivariate_linear_regressor(mlr, validation_set)
    print(validation_set_baseline)

    # Combine training and validation set
    train_valid_set = pd.concat([training_set, validation_set])

    # train mlr on combined training and validation set
    new_mlr = train_multivariate_linear_regressor(train_valid_set)

    # apply new mlr on test set
    test_set_baseline = apply_multivariate_linear_regressor(new_mlr, test_set)

    ###########################################################################

    # Phase 2:

    ###########################################################################


if __name__ == "__main__":
    main()