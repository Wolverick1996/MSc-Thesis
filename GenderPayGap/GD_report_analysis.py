import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.float_format', lambda f: '%.2f' % f)

rows = []
males = []
females = []
with open('data.csv') as File:
    reader = csv.DictReader(File)
    for row in reader:
        rows.append(row)
        if 'Male' in row.values():
            males.append(row)
        if 'Female' in row.values():
            females.append(row)
    print("%d tuples: %d males and %d females\n" % (len(rows), len(males), len(females)))

sumMales = 0
for i in range(len(males)):
    sumMales = sumMales + int(males[i].get('basePay'))
avgMalePay = sumMales / len(males)
print("Average basePay for males: %d" % avgMalePay)
sumFemales = 0
for i in range(len(females)):
    sumFemales = sumFemales + int(females[i].get('basePay'))
avgFemalePay = sumFemales / len(females)
print("Average basePay for females: %d" % avgFemalePay)

print("'Unadjusted' Gender Pay Gap: %f" % ((avgMalePay - avgFemalePay) / avgMalePay))


# Get the Data Ready #
df = pd.read_csv('data.csv')

# Create five employee age bins to simplify number of age groups.
labels = ["0-24", "25-34", "35-44", "45-54", "55+"]
bins = [0, 24, 34, 44, 54, 100]
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels)

# Take the natural logarithm of base pay (for percentage pay gap interpretation in regressions).
df['log_base'] = np.log(df['basePay'])
# Create dummy indicator for gender (male = 1, female = 0).
df['male'] = np.where(df['gender'] == 'Male', 1, 0)
print("\n\n", df)

# Look at the Data #
# Create an overall table of summary statistics for the data.
print(df.describe())

# Create a table showing overall male-female pay differences in base pay
print("\n", pd.pivot_table(df, index=["gender"], values=["basePay"], aggfunc=[np.average, np.median, len]))

# How are employees spread out among job titles?
print("\n", pd.pivot_table(df, index=["jobTitle", "gender"], values=["basePay"], aggfunc=[np.average, len]))

# Run Your Regressions #
t = PrettyTable()
t.add_column('', ['Male', 'Perf. Eval', 'Age', 'Seniority', 'Constant',
                  'Controls\n   - Job Title\n   - Department\n   - Education', 'Observations', 'R^2'])

# No controls. ("unadjusted" pay gap.)
x = df['male'].values.reshape(-1, 1)
y = df['log_base']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient = model.coef_[0]
t.add_column('(1)', ['%.3f' % coefficient, '', '', '', '%.3f' % constant, '\nNo\nNo\nNo', '1000', '%.3f' % r_sq])

# Add controls for age, education and performance evaluations.
x = df[['male', 'perfEval', 'age_bin', 'edu']]
x = pd.get_dummies(data=x, drop_first=True)
y = df['log_base']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient0 = model.coef_[0]
coefficient1 = model.coef_[1]
coefficient2 = model.coef_[2]
t.add_column('(2)', ['%.3f' % coefficient0, '%.3f' % coefficient1, '%.3f' % coefficient2, '', '%.3f' % constant,
                     '\nNo\nNo\nYes', '1000', '%.3f' % r_sq])

# Add all controls. ("adjusted" pay gap.)
x = df[['male', 'perfEval', 'age_bin', 'edu', 'dept', 'seniority', 'jobTitle']]
x = pd.get_dummies(data=x, drop_first=True)
y = df['log_base']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient0 = model.coef_[0]
coefficient1 = model.coef_[1]
coefficient2 = model.coef_[2]
coefficient3 = model.coef_[3]
t.add_column('(3)', ['%.3f' % coefficient0, '%.3f' % coefficient1, '%.3f' % coefficient2, '%.3f' % coefficient3,
                     '%.3f' % constant, '\nYes\nYes\nYes', '1000', '%.3f' % r_sq])

# Publish a clean table of regression results.
print('\nDependent Variable: Log of Base Pay\n', t)
