# Import Statements #
# import all necessary packages
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import GermanDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.explainers import MetricJSONExplainer
import tensorflow as tf
import json
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
print(tf.__version__)

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)

# Load Data, Specify Protected Attribute, and Split Data #
# note that we drop sex, which may also be a protected attribute
dataset_orig = GermanDataset(protected_attribute_names=['age'],
                             privileged_classes=[lambda x: x >= 25],
                             features_to_drop=['personal_status', 'sex'])

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

print("Original one hot encoded German dataset shape: ", dataset_orig.features.shape)
print("Train dataset shape: ", dataset_orig_train.features.shape)
print("Test dataset shape: ", dataset_orig_test.features.shape)

df, dict_df = dataset_orig.convert_to_dataframe()

print("\nShape: ", df.shape)
print(df.columns)
print("\n", df.head(5))

print("\nKey: ", dataset_orig.metadata['protected_attribute_maps'][1])
df['age'].value_counts().plot(kind='bar')
plt.xlabel("Age (0 = under 25, 1 = over 25)")
plt.ylabel("Frequency")
plt.show()

print("\nKey: ", dataset_orig.metadata['label_maps'])
df['credit'].value_counts().plot(kind='bar')
plt.xlabel("Credit (1 = Good Credit, 2 = Bad Credit)")
plt.ylabel("Frequency")
plt.show()

# Credit scores vary with age: people under 25 are more likely to be labelled as Bad Credit

# Compute Fairness Metrics on Original Training Data #
# Mean Outcomes #
metric_orig_train = BinaryLabelDatasetMetric(
    dataset_orig_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("\nOriginal training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_orig_train.mean_difference())

# Disparate Impact #
print("\nOriginal training dataset")
print("Disparate Impact = %f" % metric_orig_train.disparate_impact())

# Built-In Explainers #
json_expl = MetricJSONExplainer(metric_orig_train)


def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)


print("\n", format_json(json_expl.mean_difference()))

print(format_json(json_expl.disparate_impact()), "\n")

# Both the Mean Difference and the Disparate Impact suggest that the privileged group (people > 25) are more likely
# to get a favorable outcome (benefit)

# Bias Mitigation via In-Processing #
# Adversarial Debiasing #

# reset tensorflow graph
tf.compat.v1.reset_default_graph()

# start tensorflow session
sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()

# create AdversarialDebiasing model
debiased_model = AdversarialDebiasing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    scope_name='debiased_classifier',
    debias=True,
    sess=sess)

# fit the model to training data
debiased_model.fit(dataset_orig_train)

# make predictions on training and test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

# metrics
metric_dataset_debiasing_test = BinaryLabelDatasetMetric(
    dataset_debiasing_test,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
  )

# close session
sess.close()

# Fairness Metrics under Adversarial Debiasing #
print("\nDebiased dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_dataset_debiasing_test.mean_difference())

print("\nDebiased dataset")
print("Disparate Impact = %f" % metric_dataset_debiasing_test.disparate_impact())

# Both Mean Difference and Disparate Impact increased to reach the ideal values
# => there should not be a privileged group because the outcome should be fair for everyone

# Bias Mitigation via Pre-Processing #
# Reweighing #
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)

len(dataset_transf_train.instance_weights)
print("\n", dataset_transf_train.instance_weights[0:10])

# Compute Fairness Metrics in Transformed Data #
metric_transf_train = BinaryLabelDatasetMetric(
    dataset_transf_train,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)
print("\nTransformed dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_transf_train.mean_difference())

print("\nTransformed dataset")
print("Disparate Impact = %f" % metric_transf_train.disparate_impact())

# Both Mean Difference and Disparate Impact increased to reach the ideal values
# => there should not be a privileged group because the outcome should be fair for everyone
