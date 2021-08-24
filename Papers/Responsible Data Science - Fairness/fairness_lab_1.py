# Packages and Modules #
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)

# Load Data #
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df_compas = pd.read_csv(url)
print("\nShape: ", df_compas.shape)
display(df_compas.head(5))

# Inspect Data #
plt.hist(df_compas['age'])
plt.show()
plt.hist(df_compas['race'])
plt.show()
plt.hist(df_compas['sex'])
plt.show()

# Preprocess Data #
cols_to_keep = ["id", "age", "c_charge_degree", "race", "age_cat", "score_text",
                "sex", "priors_count", "days_b_screening_arrest",
                "decile_score", "is_recid", "two_year_recid"]

df_selected = df_compas[cols_to_keep].copy()

print("\nShape: ", df_selected.shape)
display(df_selected.head())

df_analysis = df_selected[
    (df_selected.score_text != "N/A") &
    (df_selected.days_b_screening_arrest <= 30) &
    (df_selected.days_b_screening_arrest >= -30) &
    (df_selected.is_recid != -1) &
    (df_selected.c_charge_degree != "O")
    ].copy()

print("\nShape df_compas: ", df_compas.shape)
print("Shape df_analysis: ", df_analysis.shape)

df_analysis["decile_score"] = pd.to_numeric(df_analysis["decile_score"])

# Inspect Data Again #
plt.hist(df_analysis['age'])
plt.show()
plt.hist(df_analysis['race'])
plt.show()
plt.hist(df_analysis['sex'])
plt.show()

print("\n", pd.crosstab(df_analysis['sex'], df_analysis['race']))

# Exploratory Analysis #
# plot score decile by sex
df_female = df_analysis[(df_analysis.sex == "Female")].copy()
df_male = df_analysis[(df_analysis.sex == "Male")].copy()

fig = plt.figure(figsize=(12, 6))

fig.add_subplot(121)
plt.hist(df_female["decile_score"], ec="white",
         weights=np.ones(len(df_female["decile_score"])) / len(df_female["decile_score"]))
plt.xlabel("Decile Score (0-10)")
plt.ylabel("Percent of Cases")
plt.title("Female Defendant's Decile Scores")
plt.ylim([0, 0.25])

fig.add_subplot(122)
plt.hist(df_male["decile_score"], ec="white",
         weights=np.ones(len(df_male["decile_score"])) / len(df_male["decile_score"]))
plt.xlabel("Decile Score (0-10)")
plt.ylabel("Percent of Cases")
plt.title("Male Defendant's Decile Scores")
plt.ylim([0, 0.25])

plt.show()

# plot score decile by race
df_black = df_analysis[(df_analysis.race == "African-American")].copy()
df_white = df_analysis[(df_analysis.race == "Caucasian")].copy()

fig = plt.figure(figsize=(12, 6))

fig.add_subplot(121)
plt.hist(df_black["decile_score"], ec="white",
         weights=np.ones(len(df_black["decile_score"])) / len(df_black["decile_score"]))
plt.xlabel("Decile Score (0-10)")
plt.ylabel("Percent of Cases")
plt.title("Black Defendant's Decile Scores")
plt.ylim([0, 0.3])

fig.add_subplot(122)
plt.hist(df_white["decile_score"], ec="white",
         weights=np.ones(len(df_white["decile_score"])) / len(df_white["decile_score"]))
plt.xlabel("Decile Score (0-10)")
plt.ylabel("Percent of Cases")
plt.title("White Defendant's Decile Scores")
plt.ylim([0, 0.3])

plt.show()

# High decile score more likely to be assigned to Black, low decile score more likely to be assigned to White

# plot risk labels by race
fig = plt.figure(figsize=(12, 6))

fig.add_subplot(121)
plt.hist(df_black["score_text"], ec="white",
         weights=np.ones(len(df_black["score_text"])) / len(df_black["score_text"]))
plt.xlabel("Risk Labels")
plt.ylabel("Percent of Cases")
plt.title("Black Defendant's Risk Labels")
plt.ylim([0, 0.7])

fig.add_subplot(122)
plt.hist(df_white["score_text"], ec="white",
         weights=np.ones(len(df_white["score_text"])) / len(df_white["score_text"]))
plt.xlabel("Risk Labels")
plt.ylabel("Percent of Cases")
plt.title("White Defendant's Risk Labels")
plt.ylim([0, 0.7])

plt.show()

# BIAS in COMPAS #
# Preprocess Data for Logistic Regression #
print("\n", df_analysis.dtypes)

for i, col_type in enumerate(df_analysis.dtypes):
    if col_type == "object":
        print("\nVariable {} takes the values: {}".format(
            df_analysis.columns[i],
            df_analysis[df_analysis.columns[i]].unique()))

df_logistic = df_analysis.copy()

# one-hot encoding
df_logistic = pd.get_dummies(df_logistic,
                             columns=["c_charge_degree", "race", "age_cat", "sex"])

# mutate score_text to binary variable where low = {low} and high = {medium, high}
df_logistic["score_binary"] = np.where(df_logistic["score_text"] != "Low", "High", "Low")
df_logistic["score_binary"] = df_logistic["score_binary"].astype('category')

# rename the columns to be more instructive and consistent with statsmodel requirements for variable names
df_logistic.columns = df_logistic.columns.str.replace(' ', '_')
df_logistic.columns = df_logistic.columns.str.replace('-', '_')

renamed_cols = {'age_cat_25___45': 'age_cat_25_to_45',
                'c_charge_degree_F': 'Felony',
                'c_charge_degree_M': 'Misdemeanor'}

df_logistic = df_logistic.rename(columns=renamed_cols)

print(df_logistic.head())

# Estimate Logistic Regression Model #
# Right-hand side
explanatory = "priors_count + two_year_recid + Misdemeanor + age_cat_Greater_than_45 + age_cat_Less_than_25 + \
race_African_American + race_Asian + race_Hispanic + race_Native_American + race_Other + sex_Female"

# Left-hand side
response = "score_binary"

# Formula
formula = response + " ~ " + explanatory
print("\n", formula)

# Note: using family = sm.families.Binomial() specifies a logistic regression
model = sm.formula.glm(formula=formula,
                       family=sm.families.Binomial(),
                       data=df_logistic).fit()

print("\n", model.summary())

# Interpret Estimates #
print("\nOdds for female defendants: ", "%.2f" % math.exp(0.2213))

print("Odds for Intercept: ", "%.2f" % math.exp(-1.5255))
print("Odds for priors_count: ", "%.2f" % math.exp(0.2689))
print("Odds for two_year_recid: ", "%.2f" % math.exp(0.6859))
print("Odds for Misdemeanor: ", "%.2f" % math.exp(-0.3112))
print("Odds for age_cat_Greater_than_45: ", "%.2f" % math.exp(-1.3556))
print("Odds for age_cat_Less_than_25: ", "%.2f" % math.exp(1.3084))
print("Odds for race_African_American: ", "%.2f" % math.exp(0.4772))
print("Odds for race_Asian: ", "%.2f" % math.exp(-0.2544))
print("Odds for race_Hispanic: ", "%.2f" % math.exp(-0.4284))
print("Odds for race_Native_American: ", "%.2f" % math.exp(1.3942))
print("Odds for race_Other: ", "%.2f" % math.exp(-0.8263))

# A person under 25 years old is 3.7 times more likely to be labelled as high risk compared to over 25 years old people
# A person over 45 years old is 0.26 times more likely to be labelled as high risk compared to under 25 years old people

# Predictive Accuracy #
# A: false positives, D: false negatives => focus on these because they're both errors in the prediction

print("\nAll defendants")
print(pd.crosstab(df_logistic["score_binary"], df_logistic["is_recid"]))

true_positive = 1817  # @param {type:"number"}
false_positive = 934  # @param {type:"number"}
true_negative = 2248  # @param {type:"number"}
false_negative = 1173  # @param {type:"number"}

print("\nFalse positive rate [FP / (FP + TN)]: ", "%.2f" % (false_positive / (false_positive + true_negative)))
print("False negative rate [FN / (FN + TP)]: ", "%.2f" % (false_negative / (false_negative + true_positive)))

mask = df_logistic["sex_Female"] == 1
print("\n", pd.crosstab(df_logistic.loc[mask, "score_binary"],
                        df_logistic.loc[mask, "is_recid"]))
print("Female defendants")

true_positive = 220  # @param {type:"number"}
false_positive = 256  # @param {type:"number"}
true_negative = 520  # @param {type:"number"}
false_negative = 179  # @param {type:"number"}

print("\nFalse positive rate (females): ", "%.2f" % (false_positive / (false_positive + true_negative)))
print("False negative rate (females): ", "%.2f" % (false_negative / (false_negative + true_positive)))

mask = df_logistic["sex_Male"] == 1
print("\n", pd.crosstab(df_logistic.loc[mask, "score_binary"],
                        df_logistic.loc[mask, "is_recid"]))
print("Male defendants")

true_positive = 714  # @param {type:"number"}
false_positive = 1561  # @param {type:"number"}
true_negative = 1728  # @param {type:"number"}
false_negative = 994  # @param {type:"number"}

print("\nFalse positive rate (males): ", "%.2f" % (false_positive / (false_positive + true_negative)))
print("False negative rate (males): ", "%.2f" % (false_negative / (false_negative + true_positive)))

mask = df_logistic["race_Caucasian"] == 1
print("\n", pd.crosstab(df_logistic.loc[mask, "score_binary"],
                        df_logistic.loc[mask, "is_recid"]))
print("White defendants")

true_positive = 266  # @param {type:"number"}
false_positive = 430  # @param {type:"number"}
true_negative = 963  # @param {type:"number"}
false_negative = 444  # @param {type:"number"}

print("\nFalse positive rate (White defendants): ", "%.2f" % (false_positive / (false_positive + true_negative)))
print("False negative rate (White defendants): ", "%.2f" % (false_negative / (false_negative + true_positive)))

mask = df_logistic["race_African_American"] == 1
print("\n", pd.crosstab(df_logistic.loc[mask, "score_binary"],
                        df_logistic.loc[mask, "is_recid"]))
print("Black defendants")

true_positive = 581  # @param {type:"number"}
false_positive = 1248  # @param {type:"number"}
true_negative = 821  # @param {type:"number"}
false_negative = 525  # @param {type:"number"}

print("\nFalse positive rate (Black defendants): ", "%.2f" % (false_positive / (false_positive + true_negative)))
print("False negative rate (Black defendants): ", "%.2f" % (false_negative / (false_negative + true_positive)))
