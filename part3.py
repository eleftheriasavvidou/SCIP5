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

    # z-score normalisation
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    lr = LinearRegression()
    return lr.fit(features, nox)


def apply_multivariate_linear_regressor(mlr, df):
    features = df.drop(["NOX"], axis=1).to_numpy()
    nox = df["NOX"].to_numpy()

    # z-score normalisation
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Predict nox value for each testcase
    prediction = mlr.predict(features)

    # Spearman Rank Correlation
    sc, _ = spearmanr(nox, prediction)

    # Mean Absolute Error
    mae = sum(abs(prediction - nox)) / len(prediction)

    # R^2
    r2 = r2_score(nox, prediction)

    return (sc, mae, r2)


def plot_feature(df, feature_name):

    feature = df[feature_name].to_numpy()
    nox = df["NOX"].to_numpy()

    # Add n-th order polynomial
    poly = np.poly1d(np.polyfit(feature, nox, 2))
    new_x = np.linspace(np.min(feature), np.max(feature))
    new_y = poly(new_x)
    

    fig, ax = plt.subplots()

    ax.scatter(feature, nox, color="blue")
    plt.plot(new_x, new_y, color='red')

    ax.set_xlim(np.min(feature) - 1, np.max(feature) + 1)
    ax.set_ylim(np.min(nox) -1, np.max(nox) + 1)
    ax.set_aspect(1)

    ax.set_title("The relation between " + feature_name + " and result NOx")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("NOx")

    plt.show()
    return 


def add_squared_feature_column(old_df, feature_name):
    df = old_df.copy()
    df["squared_" + feature_name] = [e**2 for e in df[feature_name]]
    return df


def change_features(old_df):
    df = old_df.copy()
    df = add_squared_feature_column(df, "AT")
    df = add_squared_feature_column(df, "AH")

    # df = add_squared_feature_column(df, "CDP")
    # df = add_squared_feature_column(df, "TEY")
    return df

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

    # Combine training and validation set
    train_valid_set = pd.concat([training_set, validation_set])

    # train mlr on combined training and validation set
    mlr2 = train_multivariate_linear_regressor(train_valid_set)

    # apply new mlr on test set
    test_set_baseline = apply_multivariate_linear_regressor(mlr2, test_set)

    ###########################################################################

    # Phase 2:
    # - Create new features, e.g. PCA, multiplicate, log, devide, combine features
    #       - at any point, probe validation set performance of our set of features (5 times max)
    # - Provide model based explainations via visualising feature weights
    # - Pick two random instances from the test set and provide instance level explainations of them

    ###########################################################################

    new_training_set = change_features(training_set)
    new_validation_set = change_features(validation_set)


    # probe validation set performance of our set of features
    new_mlr = train_multivariate_linear_regressor(new_training_set)
    new_validation_set_baseline = apply_multivariate_linear_regressor(new_mlr, new_validation_set)


    feature_names = new_training_set.columns.values
    feature_names = np.append(feature_names[:9], feature_names[10:len(feature_names) + 1])

    print(feature_names)

    # summarize feature importance
    importance = new_mlr.coef_
    for i,v in enumerate(importance):
        print('Feature: ' + str(feature_names[i]) + ', Score: ' + str(v))

    # plot feature importance4
    fig, ax = plt.subplots()
    ax.bar([x for x in range(len(importance))], importance)

    ax.set_xticks([x for x in range(len(importance))])
    ax.set_xticklabels(feature_names)
    plt.show()

    print(validation_set_baseline)
    print(new_validation_set_baseline)

    # Method to plot feature columns
    #plot_feature(new_training_set, "CDP")
    #plot_feature(new_training_set, "squared_CDP")
    #plot_feature(new_training_set, "TEY")
    #plot_feature(new_training_set, "AT")

if __name__ == "__main__":
    main()