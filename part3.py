import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def make_dataframe(path):
    return pd.read_csv(path)


def split_dataframe(df, n):
    return np.array_split(df, n)


def add_dataframe(df, addition):
    return pd.concat([df, addition])


def train_multivariate_linear_regressor(df):
    features = df.drop(["NOX"], axis=1).to_numpy()
    nox = df["NOX"].to_numpy()

    # z-score normalisation
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    lr = LinearRegression()
    return lr.fit(features, nox)


def create_table(data, table_name):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(data, columns=list('SAR'))

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.title(table_name)

    fig.tight_layout()

    plt.savefig(table_name)



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


def plot_feature_weights(mlr, feature_names):

    # summarize feature importance
    importance = mlr.coef_
    for i,v in enumerate(importance):
        print('Feature: ' + str(feature_names[i]) + ', Score: ' + str(v))

    # plot feature importance4
    fig, ax = plt.subplots()
    ax.bar([x for x in range(len(importance))], importance)

    ax.set_xticks([x for x in range(len(importance))])
    ax.set_xticklabels(feature_names)

    plt.show()


def add_average(old_df, feature_name_1, feature_name_2):
    df = old_df.copy()
    feature1 = df[feature_name_1].array
    feature2 = df[feature_name_2].array

    i = 0
    x = []

    while i < len(feature1):
        value = feature1[i] + feature2[i] / 2
        x.append(value)
        i += 1

    df["average" + feature_name_1 + feature_name_2] = x

    return df


def add_log_feature_column(old_df, feature_name):
    df = old_df.copy()
    df["squared_" + feature_name] = [np.log(e) for e in df[feature_name]]
    return df


def add_squared_feature_column(old_df, feature_name):
    df = old_df.copy()
    df["squared_" + feature_name] = [e**2 for e in df[feature_name]]
    return df


def add_squareroot_feature_column(old_df, feature_name):
    df = old_df.copy()
    df["squared_" + feature_name] = [np.sqrt(e) for e in df[feature_name]]
    return df


def change_features(old_df):
    df = old_df.copy()
    df = add_log_feature_column(df, "CDP")
    df = add_log_feature_column(df, "TEY")
    df = add_log_feature_column(df, "TIT")

    return df


def simulate_online_learning(old_training_set, simulation_set, iterations, name):
    training_set = old_training_set.copy()
    simulation_blocks = split_dataframe(simulation_set, iterations)
    tableData = []

    baselines = []
    for i in range(iterations):

        block = simulation_blocks[i]

        # Train model for current training set
        mlr = train_multivariate_linear_regressor(training_set)

        # Compute performance of training set on current block
        linear_regressor = apply_multivariate_linear_regressor(mlr, block)
        tableData.append(linear_regressor)
        baselines.append(linear_regressor)

        # Add block to training set
        training_set = add_dataframe(training_set, block)

    create_table(tableData, name)

    return np.array(baselines)


def compare_block_wise_perfomance(original_baseline, new_baseline, original_block_wise_performance,
                                                                        new_block_wise_performance):

    original_baseline = np.array(original_baseline)
    new_baseline = np.array(new_baseline)

    # step 1 - 3

    print("\nBlockwise performance difference (original features):")
    print(original_block_wise_performance)
    create_table(original_block_wise_performance, "Blockwise performance difference (original features)")

    print("\nBlockwise performance difference (generated features):")
    print(new_block_wise_performance)
    create_table(new_block_wise_performance, "Blockwise performance difference (generated features)")

    block_wise_performance_difference = new_block_wise_performance - original_block_wise_performance

    print("\nBlockwise performance difference (generated features - original features):")
    print(block_wise_performance_difference)
    create_table(new_block_wise_performance, "Blockwise performance difference (generated features - original features):")

    print("\nAverage blockwise performance difference (generated features - original features):")
    print(np.mean(block_wise_performance_difference, axis=0))

    # Step 5

    print("\nOriginal baseline (original features):")
    print(original_baseline)

    print("\nOriginal baseline (generated features):")
    print(new_baseline)

    print("\nOnline Learning simulation average performance (original features):")
    print(np.mean(original_block_wise_performance, axis=0))

    print("\nOnline Learning simulation average performance (generated features):")
    print(np.mean(new_block_wise_performance, axis=0))


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
    training_set = add_dataframe(d2011, d2012)

    # Validation set -> 2013
    validation_set = d2013

    # Test set -> 2014 + 2015
    test_set = add_dataframe(d2014, d2015)

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
    new_train_valid_set = change_features(train_valid_set)
    new_test_set = change_features(test_set)


    # probe validation set performance of our set of features
    new_vaidation_mlr = train_multivariate_linear_regressor(new_training_set)
    new_validation_set_baseline = apply_multivariate_linear_regressor(new_vaidation_mlr, new_validation_set)

    # probe validation set performance of our set of features
    new_test_mlr = train_multivariate_linear_regressor(new_train_valid_set)
    new_test_set_baseline = apply_multivariate_linear_regressor(new_test_mlr, new_test_set)


    feature_names = new_training_set.columns.values
    feature_names = np.append(feature_names[:9], feature_names[10:len(feature_names) + 1])

    #print(validation_set_baseline)
    #print(new_validation_set_baseline)

    # Method to display feature weights
    #plot_feature_weights(new_mlr, feature_names)

    # Method to plot feature columns
    #plot_feature(new_training_set, "CDP")
    #plot_feature(new_training_set, "squared_CDP")
    #plot_feature(new_training_set, "TEY")
    #plot_feature(new_training_set, "AT")

    ###########################################################################

    # Phase 3:
    # - Online learning experiment 1:
    #        - Split validation set in 10 equal blocks
    #        - Split test set into 20 equal sized blocks
    #        - For validation set:
    #            - Iterativelly add blocks of data to training set
    #                  - Starting with training set (2011 + 2012)
    #                  - Keep track of scoring
    #            - Compare score to baseline score (the baseline without online learning)
    #        - For test set:
    #            - Same as validation set
    # - Online learning experiment 2:
    #        - Same experiment but then use generated features

    ###########################################################################


    print("\n###############################################################")
    print("################# validation set performances #################")
    print("###############################################################\n")


    # Step 1 - 5: Original features
    online_validation_baselines = simulate_online_learning(training_set, validation_set, 10, "Original features")

    # Step 1 - 5: Generated features
    new_online_validation_baselines = simulate_online_learning(new_training_set, new_validation_set, 10, "Generated features")

    # Display performance comparison between original and online learning simulation baselines on the validation set
    compare_block_wise_perfomance(validation_set_baseline, new_validation_set_baseline, 
                                    online_validation_baselines, new_online_validation_baselines)


    print("\n###############################################################")
    print("#################### test set performances ####################")
    print("###############################################################\n")


    # Step 6: Original features
    online_test_baselines = simulate_online_learning(train_valid_set, test_set, 20, "Original features test set")

    # Step 6 : Generated features
    new_online_test_baselines = simulate_online_learning(new_train_valid_set, new_test_set, 20, "Generated features test set")

    # Display performance comparison between original and online learning simulation baselines on the test set
    compare_block_wise_perfomance(test_set_baseline, new_test_set_baseline, 
                                    online_test_baselines, new_online_test_baselines)

if __name__ == "__main__":
    main()