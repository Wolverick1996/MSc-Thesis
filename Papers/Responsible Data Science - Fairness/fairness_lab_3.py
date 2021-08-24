import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import GermanDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import EqOddsPostprocessing
random.seed(6)

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)

# Load the data #
# Read in the aif360 dataset #
dataset_orig = GermanDataset(protected_attribute_names=['age'],
                             privileged_classes=[lambda x: x >= 25],
                             features_to_drop=['personal_status', 'sex'])  # age >=25 is considered privileged

# Store definitions of privileged and unprivileged groups
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

# Split into train/val/test sets #
# Split original data into train and test data
train_orig, test_orig = dataset_orig.split([0.8], shuffle=True)
train_orig, val_orig = dataset_orig.split([0.75], shuffle=True)

# Convert to dataframes
train_orig_df, _ = train_orig.convert_to_dataframe()
val_orig_df, _ = val_orig.convert_to_dataframe()
test_orig_df, _ = test_orig.convert_to_dataframe()

print("Train set: ", train_orig_df.shape)
print("Val set: ", val_orig_df.shape)
print("Test set: ", test_orig_df.shape)

print("\n", train_orig_df.columns)
print("\n", train_orig_df.head())

metric_orig_train = BinaryLabelDatasetMetric(
    train_orig,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("\nOriginal training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_orig_train.mean_difference())

print("\nOriginal training dataset")
print("Disparate Impact = %f" % metric_orig_train.disparate_impact())

# Train a classifier to predict credit using the original data #
# Training and evaluating a logistic regression model #
x_train = train_orig_df.drop("credit", axis=1)
y_train = train_orig_df.credit.replace({2: 0})
print("\nOutcomes: ")
print(y_train.value_counts())

# Set up the logistic regression model with the given hyperparameters
initial_lr = LogisticRegression(C=0.5, penalty="l1", solver='liblinear')

# Fit the model using the training data
initial_lr = initial_lr.fit(x_train, y_train, sample_weight=None)


def evaluate(model, X, y_true):
    """Calculates the AUC and accuracy for a trained logistic regression model"""

    # Calculate predicted values
    y_pred = model.predict_proba(X)
    # This returns a tuple for each observation containing the probability of being in each class.
    # Since we're doing binary classification, all we need to know is the probability that the outcome = 1 (good credit)
    y_pred = [row[1] for row in y_pred]  # This pulls the predicted probability that y = 1 for each observation

    # Calculate accuracy
    accuracy = accuracy_score(y_true, [pred_prob >= 0.5 for pred_prob in y_pred])

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)

    return accuracy, auc


# Before we call the function, we need to set up the validation data properly, the way we did for the training data.
x_val = val_orig_df.drop("credit", axis=1)
y_val = val_orig_df.credit.replace({2: 0})

accuracy, auc = evaluate(initial_lr, x_val, y_val)
print("\nAccuracy: ", accuracy)
print("AUC: ", auc)


# Hyperparameter tuning the logistic regression model #
def tune_logistic_regression(train_df, val_df, penalty_types, C_values, weights=None, verbose=True):
    """Tunes logistic regression models over the hyperparameters penalty type and C
       to maximize the AUC"""
    # Pre-process the training and validation data
    x_train = train_df.drop("credit", axis=1)
    y_train = train_df.credit.replace({2: 0})
    x_val = val_df.drop("credit", axis=1)
    y_val = val_df.credit.replace({2: 0})

    # Create empty lists where we will store the results of hyperparameter tuning
    parameters = []
    models = []
    val_aucs = []

    # Loop through the hyperparameters of interest
    for penalty in penalty_types:
        for C in C_values:
            # Train the logistic regression model with the given hyperparameters
            lr = LogisticRegression(C=C, penalty=penalty, solver='liblinear')

            # Fit the model using the training data
            lr = lr.fit(x_train, y_train, sample_weight=weights)

            # Get the evalution metrics on the validation set
            accuracy, auc = evaluate(lr, x_val, y_val)

            # Store the results
            parameters.append({'penalty': penalty, 'C': C})
            models.append(lr)
            val_aucs.append(auc)

            # Print the results
            if verbose:
                print("\nParameters: \tpenalty={} \tC={}".format(penalty, C))
                print("Validation AUC: {}".format(auc))

    # Determine the best model -- that is, the one with the AUC
    best_model_index = np.argmax(val_aucs)
    best_model = models[best_model_index]

    print("\nBest model parameters: ", parameters[best_model_index])
    print("Best model AUC: ", val_aucs[best_model_index])

    # Return best model
    return best_model, parameters, models, val_aucs


best_lr, parameters, models, val_aucs = tune_logistic_regression(train_orig_df, val_orig_df, penalty_types=["l1", "l2"],
                                                                 C_values=[0.001, 0.1, 1, 10, 100, 1000, 10000, 100000])

val_aucs_l1 = [val_aucs[i] for i in range(len(val_aucs)) if parameters[i]['penalty'] == "l1"]
val_aucs_l2 = [val_aucs[i] for i in range(len(val_aucs)) if parameters[i]['penalty'] == "l2"]
C_values = [parameters[i]['C'] for i in range(len(parameters)) if parameters[i]['penalty'] == "l2"]

fig, ax = plt.subplots()
ax.semilogx(C_values, val_aucs_l1, marker='.', markerfacecolor='blue', markersize=12, color='blue', linewidth=4,
            label='L1 Penalty')
ax.semilogx(C_values, val_aucs_l2, marker='.', markerfacecolor='red', markersize=12, color='red', linewidth=4,
            label='L2 Penalty')
ax.set_xlabel("C")
ax.set_ylabel("AUC")
plt.legend()
plt.show()

# Evaluating bias in our predictions #
# Copy the dataset
train_preds_df = train_orig_df.copy()
# Calculate predicted values
train_preds_df['credit'] = best_lr.predict(x_train)
# Recode the predictions so that they match the format that the dataset was originally provided in
# (1 = good credit, 2 = bad credit)
train_preds_df['credit'] = train_preds_df.credit.replace({0: 2})

orig_aif360 = StandardDataset(train_orig_df, label_name='credit', protected_attribute_names=['age'],
                              privileged_classes=[[1]], favorable_classes=[1])
preds_aif360 = StandardDataset(train_preds_df, label_name='credit', protected_attribute_names=['age'],
                               privileged_classes=[[1]], favorable_classes=[1])

metric_preds_train = BinaryLabelDatasetMetric(
    preds_aif360,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("\nPredicted dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_preds_train.mean_difference())

print("\nPredicted dataset")
print("Disparate Impact = %f" % metric_preds_train.disparate_impact())

orig_vs_preds_metrics = ClassificationMetric(orig_aif360, preds_aif360,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("\nError rate difference (unprivileged error rate - privileged error rate) = %f" %
      orig_vs_preds_metrics.error_rate_difference())

print("\nFalse negative rate for privileged groups = %f" % orig_vs_preds_metrics.false_negative_rate(privileged=True))
print("False negative rate for unprivileged groups = %f" % orig_vs_preds_metrics.false_negative_rate(privileged=False))
print("False negative rate ratio = %f" % orig_vs_preds_metrics.false_negative_rate_ratio())

print("\nFalse positive rate for privileged groups = %f" % orig_vs_preds_metrics.false_positive_rate(privileged=True))
print("False positive rate for unprivileged groups = %f" % orig_vs_preds_metrics.false_positive_rate(privileged=False))
print("False positive rate ratio = %f" % orig_vs_preds_metrics.false_positive_rate_ratio())

# Train a classifier to predict credit using the original data, excluding the sensitive feature #
best_lr_noage, _, _, _ = tune_logistic_regression(train_orig_df.drop('age', axis=1), val_orig_df.drop('age', axis=1),
                                                  penalty_types=["l1", "l2"],
                                                  C_values=[0.001, 0.1, 1, 10, 100, 1000, 10000, 100000], verbose=False)

preds_df_noage = train_orig_df.copy()
preds_df_noage['credit'] = best_lr_noage.predict(x_train.drop('age', axis=1))
preds_df_noage['credit'] = preds_df_noage.credit.replace({0: 2})

noage_preds_aif360 = StandardDataset(preds_df_noage, label_name='credit', protected_attribute_names=['age'],
                                     privileged_classes=[[1]], favorable_classes=[1])

noage_preds_metrics = BinaryLabelDatasetMetric(noage_preds_aif360,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print("Mean difference = %f" % noage_preds_metrics.mean_difference())
print("Disparate Impact = %f" % noage_preds_metrics.disparate_impact())

orig_vs_noage_preds_metrics = ClassificationMetric(orig_aif360, noage_preds_aif360,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

print("\nError rate difference (unprivileged error rate - privileged error rate) = %f" %
      orig_vs_noage_preds_metrics.error_rate_difference())

print("\nFalse negative rate for privileged groups = %f" %
      orig_vs_noage_preds_metrics.false_negative_rate(privileged=True))
print("False negative rate for unprivileged groups = %f" %
      orig_vs_noage_preds_metrics.false_negative_rate(privileged=False))
print("False negative rate ratio = %f" % orig_vs_noage_preds_metrics.false_negative_rate_ratio())

print("\nFalse positive rate for privileged groups = %f" %
      orig_vs_noage_preds_metrics.false_positive_rate(privileged=True))
print("False positive rate for unprivileged groups = %f" %
      orig_vs_noage_preds_metrics.false_positive_rate(privileged=False))
print("False positive rate ratio = %f" % orig_vs_noage_preds_metrics.false_positive_rate_ratio())

# Overall error rate slightly increased; false negative rate ratio decreased; false positive rate ratio increased

# Preprocess the data using the reweighting algorithm, #
# then train a classifier to predict credit using the re-weighted data #
# Fit the weights to our training data
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
RW_fit = RW.fit(train_orig)

# Pull the actual values of the weights for the training data
train_reweighed = RW_fit.transform(train_orig)
training_weights = train_reweighed.instance_weights

# Train a model using weights
best_lr_weights, _, _, _ = tune_logistic_regression(train_orig_df.drop('age', axis=1), val_orig_df.drop('age', axis=1),
                                                    penalty_types=["l1", "l2"],
                                                    C_values=[0.001, 0.1, 1, 10, 100, 1000, 10000, 100000],
                                                    weights=training_weights, verbose=False)

train_preds_df_weights = train_orig_df.copy()
train_preds_df_weights['credit'] = best_lr_weights.predict(x_train.drop('age', axis=1))
train_preds_df_weights['credit'] = train_preds_df_weights.credit.replace({0: 2})

preds_weights_aif360 = StandardDataset(train_preds_df_weights, label_name='credit', protected_attribute_names=['age'],
                                       privileged_classes=[[1]], favorable_classes=[1])
preds_weights_metrics = BinaryLabelDatasetMetric(preds_weights_aif360,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
print("Mean difference = %f" % preds_weights_metrics.mean_difference())
print("Disparate Impact = %f" % preds_weights_metrics.disparate_impact())

orig_vs_preds_weights_metrics = ClassificationMetric(orig_aif360, preds_weights_aif360,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

print("\nError rate difference (unprivileged error rate - privileged error rate) = %f" %
      orig_vs_preds_weights_metrics.error_rate_difference())

print("\nFalse negative rate for privileged groups = %f" %
      orig_vs_preds_weights_metrics.false_negative_rate(privileged=True))
print("False negative rate for unprivileged groups = %f" %
      orig_vs_preds_weights_metrics.false_negative_rate(privileged=False))
print("False negative rate ratio = %f" % orig_vs_preds_weights_metrics.false_negative_rate_ratio())

print("\nFalse positive rate for privileged groups = %f" %
      orig_vs_preds_weights_metrics.false_positive_rate(privileged=True))
print("False positive rate for unprivileged groups = %f" %
      orig_vs_preds_weights_metrics.false_positive_rate(privileged=False))
print("False positive rate ratio = %f" % orig_vs_preds_weights_metrics.false_positive_rate_ratio())

# Post-process the predictions from the model that we trained #
# using weights by using the calibrated equality of odds algorithm #
# Transform our predictions using the aif360 implementation of the equality of odds algorithm
eq_odds = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, seed=47)
preds_weights_eq_odds_aif360 = eq_odds.fit_predict(orig_aif360, preds_weights_aif360)

# write code to calculate fairness metrics here
preds_weights_eq_odds_metrics = BinaryLabelDatasetMetric(preds_weights_eq_odds_aif360,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)

orig_vs_preds_weights_eq_odds_metrics = ClassificationMetric(orig_aif360, preds_weights_eq_odds_aif360,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)

# print the metrics
print("\nMean difference = %f" % preds_weights_eq_odds_metrics.mean_difference())
print("Disparate Impact = %f" % preds_weights_eq_odds_metrics.disparate_impact())

print("\nError rate difference (unprivileged error rate - privileged error rate) = %f" %
      orig_vs_preds_weights_eq_odds_metrics.error_rate_difference())

print("\nFalse negative rate for privileged groups = %f" %
      orig_vs_preds_weights_eq_odds_metrics.false_negative_rate(privileged=True))
print("False negative rate for unprivileged groups = %f" %
      orig_vs_preds_weights_eq_odds_metrics.false_negative_rate(privileged=False))
print("False negative rate ratio = %f" % orig_vs_preds_weights_eq_odds_metrics.false_negative_rate_ratio())

print("\nFalse positive rate for privileged groups = %f" %
      orig_vs_preds_weights_eq_odds_metrics.false_positive_rate(privileged=True))
print("False positive rate for unprivileged groups = %f" %
      orig_vs_preds_weights_eq_odds_metrics.false_positive_rate(privileged=False))
print("False positive rate ratio = %f" % orig_vs_preds_weights_eq_odds_metrics.false_positive_rate_ratio())

# Test how accuracy has changed
print("\nAccuracy (on training data) before equality of odds algorithm = %f" % orig_vs_preds_weights_metrics.accuracy())
print("\nAccuracy (on training data) after equality of odds algorithm = %f" %
      orig_vs_preds_weights_eq_odds_metrics.accuracy())
