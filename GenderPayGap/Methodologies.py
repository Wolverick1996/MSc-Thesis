import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.float_format', lambda f: '%.2f' % f)

# SanFrancisco/SanFrancisco_Adjusted.csv
# Chicago/Chicago_Adjusted.csv
# ChicagoBiased/Chicago_Biased.csv
# ChicagoGrouped/Chicago_Grouped.csv
path = input("Enter CSV file path:\t")

# Get the Data Ready #
df = 0
try:
    df = pd.read_csv(path)
    #df = pd.read_csv('SanFrancisco/SanFrancisco_Adjusted.csv')
    #df = pd.read_csv('Chicago/Chicago_Adjusted.csv')
except (FileNotFoundError, IOError):
    print("Wrong file or file path")
    quit()

# --- STATISTICAL ANALYSIS --- #
print("--- STATISTICAL ANALYSIS ---")

print(df, "\n\nMales: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']),
      "\nNumber of different Job Titles (total): ", df['Job Title'].nunique(), "\n")

# Take the natural logarithm of Annual Salary (for percentage pay gap interpretation in regressions)
df['Log Annual Salary'] = np.log(df['Annual Salary'])
# Create dummy indicator for gender (male = 1, female = 0)
df['Male'] = np.where(df['Gender'] == 'male', 1, 0)

# Look at the Data #
# Create an overall table of summary statistics for the data
print(df.describe())

# Create a table showing overall male-female pay differences in Annual Salary
print("")
print(pd.pivot_table(df, index=["Gender"], values=["Annual Salary"], aggfunc=[np.average, np.median, len]))

# How are employees spread out among Job Titles?
pd.set_option('display.max_rows', None)
print("")
print(pd.pivot_table(df, index=["Job Title", "Gender", "Status"], values=["Annual Salary"], aggfunc=[np.average, len]))

# Run Your Regressions #
t = PrettyTable()
if 'Department' not in df.columns:
    t.add_column('', ['Male', 'Job Title', 'Status', 'Constant',
                      'Controls\n   - Job Title\n   - Status', 'Observations', 'R^2'])
else:
    t.add_column('', ['Male', 'Job Title', 'Department', 'Status', 'Constant',
                      'Controls\n   - Job Title\n   - Department\n   - Status', 'Observations', 'R^2'])

# No controls ("unadjusted" pay gap)
x = df['Male'].values.reshape(-1, 1)
y = df['Log Annual Salary']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient = model.coef_[0]
if 'Department' not in df.columns:
    t.add_column('(1)', ['%.3f' % coefficient, '', '', '%.3f' % constant, '\nNo\nNo', len(df), '%.3f' % r_sq])
else:
    t.add_column('(1)', ['%.3f' % coefficient, '', '', '', '%.3f' % constant, '\nNo\nNo\nNo', len(df), '%.3f' % r_sq])

if 'Department' not in df.columns:
    # Add controls for Job Title and Status ("adjusted" pay gap)
    x = df[['Male', 'Job Title', 'Status']]
else:
    # Add controls for Job Title, Department and Status ("adjusted" pay gap)
    x = df[['Male', 'Job Title', 'Department', 'Status']]
x = pd.get_dummies(data=x, drop_first=True)
y = df['Log Annual Salary']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient0 = model.coef_[0]
coefficient1 = model.coef_[1]
coefficient2 = model.coef_[2]
if 'Department' not in df.columns:
    t.add_column('(2)', ['%.3f' % coefficient0, '%.3f' % coefficient1, '%.3f' % coefficient2,
                         '%.3f' % constant, '\nYes\nYes', len(df), '%.3f' % r_sq])
else:
    coefficient3 = model.coef_[3]
    t.add_column('(2)', ['%.3f' % coefficient0, '%.3f' % coefficient1, '%.3f' % coefficient2, '%.3f' % coefficient3,
                         '%.3f' % constant, '\nYes\nYes\nYes', len(df), '%.3f' % r_sq])

# Publish a clean table of regression results
print('\nDependent Variable: Log Annual Salary')
print(t)
print("'Unadjusted' pay gap: men on average earn", '%.1f' % (coefficient * 100), "% more than women",
      "\n'Adjusted' pay gap: men on average earn", '%.1f' % (coefficient0 * 100),
      "% more than women")


# --- RankingFacts --- #
# Opening file #
dfRF = pd.read_csv(path)
#dfRF = pd.read_csv('SanFrancisco/SanFrancisco_Adjusted.csv')
#dfRF = pd.read_csv('Chicago/Chicago_Adjusted.csv')

# Convert categorical to numerical attributes (RankingFacts use numerical attributes)
dfRF['Status'] = np.where(dfRF['Status'] == 'F', 1, 0)  # (F = 1, P = 0)
if is_numeric_dtype(dfRF['Job Title']) is False:
    label_encoder = LabelEncoder()
    label_encoder.fit(dfRF['Job Title'])
    label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nEncoding Job Title:\n", label_encoder_name_mapping)
    dfRF['Job Title'] = label_encoder.fit_transform(dfRF['Job Title'])
if 'Department' in dfRF.columns:
    label_encoder = LabelEncoder()
    label_encoder.fit(dfRF['Department'])
    label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nEncoding Department:\n", label_encoder_name_mapping)
    dfRF['Department'] = label_encoder.fit_transform(dfRF['Department'])
if 'Salary or Hourly' in df.columns:
    dfRF['Salary or Hourly'] = np.where(dfRF['Salary or Hourly'] == 'Salary', 1, 0)  # (Salary = 1, Hourly = 0)

# Exporting CSV
#path = 'SanFrancisco/SanFrancisco_Adjusted.csv'
#path = 'Chicago/Chicago_Adjusted.csv'
pathRF = 'MethodologiesResults/' + path.split('/')[0] + '_RankingFacts.csv'
dfRF.to_csv(pathRF, index=False)


# --- FAIR-DB --- #
print("\n\n--- FAIR-DB ---")

pd.set_option('display.max_rows', 10)
print(df)

# Data Acquisition #
# Create employee income bins (numbers are not really useful in estimating correlations)
#labels = ["0-39", "40-59", "60-79", "80-99", "100-119", "120-139", "140+"]
#bins = [0, 40, 60, 80, 100, 120, 140, 500]
labels = ["<= 90K", "> 90K"]
bins = [0, 90, 500]
#df['Annual Salary'] = df['Annual Salary'] / 1000
df['Annual Salary Bin'] = pd.cut(df['Annual Salary'] / 1000, bins=bins, labels=labels)

# Plotting distribution of Annual Salary over Gender
df.hist(column='Annual Salary')
plt.savefig('MethodologiesResults/' + path.split('/')[0] + '_annual_salary_distribution.pdf', format='pdf')
d = pd.crosstab(df['Annual Salary Bin'], df['Gender'])
d.plot.bar(stacked=True, color=['hotpink', 'dodgerblue'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.title("Distribution of Annual Salary over Gender")
if len(labels) == 2:
    plt.savefig('MethodologiesResults/' + path.split('/')[0] + '_2bins_annual_salary_over_gender.pdf',
                format='pdf', bbox_inches='tight')
else:
    plt.savefig('MethodologiesResults/' + path.split('/')[0] + '_annual_salary_over_gender.pdf',
                format='pdf', bbox_inches='tight')

dfPlot = df[df['Gender'] == 'male']
# Get the frequency and CDF (Cumulative Distribution Function) for each value in the Annual Salary column (males)
stats_df = dfPlot.groupby('Annual Salary')['Annual Salary']\
    .agg('count').pipe(pd.DataFrame).rename(columns={'Annual Salary': 'frequency'})
stats_df['pmf'] = stats_df['frequency'] / sum(stats_df['frequency'])
stats_df['cdf'] = stats_df['pmf'].cumsum()
stats_df = stats_df.reset_index()
ax1 = stats_df.plot(x='Annual Salary', y=['cdf'], grid=True)
plt.title("CDF for Annual Salary (Males)")
plt.savefig('MethodologiesResults/' + path.split('/')[0] + '_cdf_annual_salary_males.pdf', format='pdf')

dfPlot = df[df['Gender'] == 'female']
# Get the frequency and CDF (Cumulative Distribution Function) for each value in the Annual Salary column (females)
stats_df = dfPlot.groupby('Annual Salary')['Annual Salary']\
    .agg('count').pipe(pd.DataFrame).rename(columns={'Annual Salary': 'frequency'})
stats_df['pmf'] = stats_df['frequency'] / sum(stats_df['frequency'])
stats_df['cdf'] = stats_df['pmf'].cumsum()
stats_df = stats_df.reset_index()
ax2 = stats_df.plot(x='Annual Salary', y=['cdf'], grid=True)
plt.title("CDF for Annual Salary (Females)")
plt.savefig('MethodologiesResults/' + path.split('/')[0] + '_cdf_annual_salary_females.pdf', format='pdf')

ax1.get_shared_x_axes().join(ax1, ax2)
ax2.autoscale()
plt.show()

# Removing useless columns
del df['Name'], df['Annual Salary'], df['Log Annual Salary'], df['Male']

all_tuples = len(df)
cols = df.columns

# Because there are numbers in the DataFrame, I transform them into strings to perform comparisons in completion phase
df = df.applymap(lambda x: str(x) if type(x) == int else x)

print("\nTotal number of tuples in df: ", len(df), "\n")
print(df.head())

# INPUTS
# Array of protected attributes
protected_attr = ['Gender']
# Target class
target = 'Annual Salary Bin'
binaryValues = df['Annual Salary Bin'].unique()
print("\nAnnual Salary Bin values: ", binaryValues)

# Input parameters
confidence = 0.8
supportCount = 100
support = supportCount / len(df)
maxSize = 2
grepValue = target + '='
minDiff = 0.02

# Removing ',' character to avoid problems with CFDDiscovery
df['Job Title'] = df['Job Title'].str.replace(', ', ' ')

# Exporting CSV
if len(labels) == 2:
    pathFDB = 'MethodologiesResults/' + path.split('/')[0] + '_FAIR-DB_2bins.csv'
else:
    pathFDB = 'MethodologiesResults/' + path.split('/')[0] + '_FAIR-DB.csv'
df.to_csv(pathFDB, index=False, line_terminator=',\n')

# ACFDs Discovery and Filtering #
# Apply CFDDiscovery algorithm ->
# https://codeocean.com/capsule/6146641/tree (original) | https://codeocean.com/capsule/5820643/tree (copied)

# SanFrancisco/cfds[s=100_FPT_2bins].txt
# Chicago/cfds[s=100_FPT_2bins].txt
# ChicagoBiased/cfds[s=100_FPT_2bins].txt
# ChicagoGrouped/cfds[s=100_FPT_2bins].txt
file = input("\nEnter path of the file containing the dependencies (obtained applying CFDDiscovery):\t")
with open(file) as f:
    output = f.read().splitlines()

# all rules obtained
print("\nTotal number of dependencies found: ", len(output))

# further filtering because the CFDDiscovery capsule doesn't allow to use the grepValue to filter rules with no target
o0 = list()
for i in range(0, len(output)):
    if grepValue in output[i]:
        o0.append(output[i])
print("Total number of dependencies found (grep): ", len(o0))

for i in range(0, 8):
    if len(o0) > i:
        print("Dependency n.", i, ":\t", o0[i])

# Transform the '<=' in '<' and viceversa to avoid problems with the following '=' detection
o1 = list(map(lambda x: x.replace("<=", "<"), o0))
# Delete the parenthesis
o1 = list(map(lambda x: x.replace("(", ""), o1))
o1 = list(map(lambda x: x.replace(")", ""), o1))
# Split the entire rule in a LHS and RHS
o2 = list(map(lambda x: x.split(' => '), o1))


# Function to select only CFDs from all rules (x is the single rule)
def parseCFD(x):
    # Flag indicates if the rule is a CFD (True) or AFD (False)
    isCFD = True
    rawLHS = x[0].split(', ')
    for i, y in enumerate(rawLHS):
        for attr in cols:
            if y in str(attr + '=!'):
                isCFD = False

    rawRHS = x[1].split(', ')
    for i, y in enumerate(rawRHS):
        for attr in cols:
            if y in str(attr + '=!'):
                isCFD = False

        # To keep only CFDs
        if isCFD:
            return [rawLHS, rawRHS]
        else:
            return None


# conditions is an array of conditions to delete some rules that are not interesting
def parseCFDWithCond(x, conditionsLHS, conditionsRHS):
    # Flag indicates if the rule is a CFD (True) or AFD (False)
    isCFD = True
    # Flag indicates if the rule contains unwanted condition(s) (RHS or LHS) - it doesn't contain the condition (True)
    takenRule = True
    rawLHS = x[0].split(', ')
    for i, y in enumerate(rawLHS):
        for attr in cols:
            if y in str(attr + '=!'):
                isCFD = False
            for condLHS in conditionsLHS:
                if y == condLHS:
                    takenRule = False

    rawRHS = x[1].split(', ')
    for i, y in enumerate(rawRHS):
        for attr in cols:
            if y in str(attr + '=!'):
                isCFD = False
            for condRHS in conditionsRHS:
                if y == condRHS:
                    takenRule = False

        # To keep only CFDs
        if isCFD and takenRule:
            return [rawLHS, rawRHS]
        else:
            return None


conditionsLHS = []
conditionsRHS = []
o3 = list()
if not conditionsLHS and not conditionsRHS:
    for i in o2:
        x = parseCFD(i)
        if x is not None:
            o3.append(x)
else:
    for i in o2:
        x = parseCFDWithCond(i, conditionsLHS, conditionsRHS)
        if x is not None:
            o3.append(x)
print("\nCFDs ([rawLHS, rawRHS]):")
for i in range(0, 8):
    if len(o3) > i:
        print(o3[i])


# To split every couple attribute-value
def splitElem(l1):
    return list(map(lambda x: x.split('='), l1))


# To create an array that contains all rules with the LHS and RHS separated
def createSplitting(elem):
    elemLHS = elem[0]
    elemRHS = elem[1]
    LHS = splitElem(elemLHS)
    RHS = splitElem(elemRHS)
    return [LHS, RHS]


# Now that we have deleted all the '=' we can replace the '<' with '<='
def createDictionaryElem(side):
    elem = {}
    for x in side:
        replaced = x[1].replace('<', '<=')
        elem[x[0]] = replaced
    return elem


o4 = list(map(createSplitting, o3))
# Create the dictionary with the LHS and RHS that contains all CFDs
parsedRules = list(map(lambda x: {'lhs': createDictionaryElem(x[0]), 'rhs': createDictionaryElem(x[1])}, o4))
print("\nTotal number of dependencies in the dictionary: ", len(parsedRules))

print("\nACFDs ({'lhs', 'rhs'}):")
for i in range(0, 8):
    if len(parsedRules) > i:
        print("ACFD n.", i, ":\t", parsedRules[i])


def countOccur(elem):
    # How many times appears the LHS of the rule
    countX = 0
    # How many times appears the RHS of the rule
    countY = 0
    # How many times appears the entire rule
    countXY = 0

    # For every row of the database, count the LHS, RHS and the total count
    for index, row in df.iterrows():
        # The flags help in dealing with missing values
        flagX = True
        flagY = True

        for key in list(elem['lhs'].keys()):
            value = elem['lhs'][key]

            # Add the constraint to manage missing values
            if str(row[key]) != value:
                flagX = False

        for key in list(elem['rhs'].keys()):
            value = elem['rhs'][key]

            # Add the constraint to manage missing values
            if str(row[key]) != value:
                flagY = False

        if flagX:
            # Increase the LHS support count
            countX += 1
        if flagY:
            # Increase the RHS support count
            countY += 1
        if flagX and flagY:
            # Increase the entire rule support count
            countXY += 1

    # Return the LHS support count, RHS support count and the entire rule support count
    return countX, countY, countXY


def computeConfidenceNoProtectedAttr(elem):
    filteredRule = {'lhs': {k: v for k, v in elem['lhs'].items()
                            if ((k not in protected_attr) and (k != target))}, 'rhs': elem['rhs']}

    fCount = countOccur(filteredRule)
    # If the rule is valid for at least one tuple
    if fCount[2] != 0 and fCount[0] != 0:
        ratio = fCount[2] / fCount[0]
    else:
        ratio = 0
    return ratio


def computeConfidenceForProtectedAttr(elem, protAttr):
    filteredRule = {'lhs': {k: v for k, v in elem['lhs'].items() if (k != protAttr)}, 'rhs': elem['rhs']}

    fCount = countOccur(filteredRule)
    # If the rule is valid for at least one tuple
    if fCount[2] != 0 and fCount[0] != 0:
        ratio = fCount[2] / fCount[0]
    else:
        ratio = 0
    return ratio


def computePDifference(rule, conf, attribute):
    if attribute in protected_attr:
        if attribute in rule['lhs'].keys():
            RHSConfidence = computeConfidenceForProtectedAttr(rule, attribute)
            diffp = conf - RHSConfidence
            return diffp
    return None


def createTable(parsedRules):
    df2 = pd.DataFrame(columns=['Rule', 'Support', 'Confidence', 'Diff'])
    for attribute in protected_attr:
        column = attribute + 'Diff'
        df2[column] = 0

    row_index = 0
    for i, parsedRule in enumerate(parsedRules):
        count = countOccur(parsedRule)
        support = tuple(map(lambda val: val / all_tuples, count))
        flagProt = False

        for keyL in parsedRule['lhs'].keys():
            if keyL in protected_attr:
                flagProt = True
        for keyR in parsedRule['rhs'].keys():
            if keyR in protected_attr:
                flagProt = True

        if support[0] != 0 and support[1] != 0 and flagProt:
            conf = count[2] / count[0]
            confNoProtectedAttr = computeConfidenceNoProtectedAttr(parsedRule)
            diff = conf - confNoProtectedAttr

            df2 = df2.append({'Rule': parsedRule, 'Confidence': conf, 'Support': support[2], 'Diff': diff},
                             ignore_index=True)

            # Compute the diff for each protected attribute
            for attribute in protected_attr:
                if attribute in parsedRule['lhs'].keys():
                    diffp = computePDifference(parsedRule, conf, attribute)
                    column = attribute + 'Diff'
                    df2.loc[row_index, column] = diffp
            row_index += 1
    return df2


df2 = createTable(parsedRules)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
print("\nTotal number of tuples in df2: ", len(df2))
print(df2.head())

# To select the not ethical rules
df3 = df2[df2.Diff > minDiff]
print("\nTotal number of tuples in df3: ", len(df3))
print(df3.head())


def cartesianProduct(set_a, set_b):
    result = []
    for i in range(0, len(set_a)):
        for j in range(0, len(set_b)):

            # For handling case having cartesian product first time of two sets
            if type(set_a[i]) != list:
                set_a[i] = [set_a[i]]

            # Coping all the members of set_a to temp
            temp = [num for num in set_a[i]]

            # Add member of set_b to temp to have cartesian product
            temp.append(set_b[j])
            result.append(temp)

    return result


def cartesian(list_a, n):
    # Result of cartesian product of all the sets taken two at a time
    temp = list_a[0]

    # Do product of N sets
    for i in range(1, n):
        temp = cartesianProduct(temp, list_a[i])

    return temp


def createSide(side):
    elem = {}
    for x in side:
        elem[x[0]] = x[1]

    return elem


def findCFDsCombinations(elem):
    CFDs = []
    perm = []
    attr_names = []
    flag = False
    # Select DB according to already set attributes
    for key in list(elem['lhs'].keys()):
        if (key in protected_attr) or (key == target):
            attr_names.append(key)
            perm.append(df[key].unique())
            flag = True

    for key in list(elem['rhs'].keys()):
        if (key in protected_attr) or (key == target):
            attr_names.append(key)
            perm.append(df[key].unique())
            flag = True

    if flag is True:
        assocRule = copy.deepcopy(elem)
        mat = cartesian(perm, len(perm))
        for m in mat:
            if len(attr_names) == 1:
                for key in list(assocRule['lhs'].keys()):
                    if key == attr_names[0]:
                        assocRule['lhs'][key] = m
                for key in list(assocRule['rhs'].keys()):
                    if key == attr_names[0]:
                        assocRule['rhs'][key] = m
            else:
                i = 0
                assocRule = copy.deepcopy(elem)
                while i < len(m):
                    for key in list(assocRule['lhs'].keys()):
                        if key == attr_names[i]:
                            assocRule['lhs'][key] = m[i]
                    for key in list(assocRule['rhs'].keys()):
                        if key == attr_names[i]:
                            assocRule['rhs'][key] = m[i]
                    i += 1

            CFDs.append(assocRule)
        return CFDs
    else:
        return elem


CFDCombinations = []
for elem in df3.Rule:
    # For every rule compute the combinations over the protected attribute
    rulesCount = findCFDsCombinations(elem)
    for ar in rulesCount:
        CFDCombinations.append(ar)
# Removing duplicate combinations
CFDCombinations = [i for n, i in enumerate(CFDCombinations) if i not in CFDCombinations[:n]]

print("\nTotal number of combinations found: ", len(CFDCombinations))

for i in range(0, 8):
    if len(CFDCombinations) > i:
        print("ACFD n.", i, ":\t", CFDCombinations[i])

df4 = createTable(CFDCombinations)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
print("\nTotal number of tuples in df4: ", len(df4))
print(df4.head(), "\n")

# ACFDs Selection and ACFDs Ranking #
# orderingCriterion = 0 (order using Support), 1 (order using Difference), 2 (order using Mean)
orderingCriterion = 2
# To select the not ethical rules
df4_1 = df4[df4.Diff > minDiff]

# Order the rules by diff or support or both
if orderingCriterion == 0:
    df5 = df4_1.iloc[df4_1['Support'].argsort()[::-1][:len(df4_1)]]
elif orderingCriterion == 1:
    df5 = df4_1.iloc[df4_1['Diff'].argsort()[::-1][:len(df4_1)]]
else:
    df4_1['Mean'] = 0
    for index, row in df4_1.iterrows():
        df4_1.loc[index, 'Mean'] = ((df4_1.loc[index, 'Support'] + df4_1.loc[index, 'Diff']) / 2)
    df5 = df4_1.iloc[df4_1['Mean'].argsort()[::-1][:len(df4_1)]]

print("\nNumber of original CFDs:", len(df3), "\nNumber of combinations rules: ", len(df4),
      "\nNumber of final rules found: ", len(df5))
print(df5)

# ACFDs User Selection and Scoring #
# INPUT PARAMETERS
# Indexes of the selected rules
indexArray = []
r = ''
print("\nEnter the indexes of the interesting rules (one by one). "
      "Write 'exit' to continue or write 'all' to select all the rules")
while (r != "exit") & (r != "all"):
    r = input()
    if r == "exit":
        if len(indexArray) == 0:
            print("indexArray cannot be empty")
            r = ''
        else:
            continue
    elif r == "all":
        indexArray = list(df5.index.values)
    elif int(r) in list(df5.index.values):
        indexArray.append(int(r))
    else:
        print("Index not found. Please retry")

# Minimum number of rules necessary to have a problematic tuple
nMarked = 0


# For every rule = elem, iter over all rows and add one if the tuple respect the rule
def validates(df, elem):
    for index, row in df.iterrows():
        flag = True
        for key in list(elem['lhs'].keys()):
            value = elem['lhs'][key]
            # Add the constraint to manage '?' that could be a missing values
            if str(row[key]) != value:
                flag = False

        for key in list(elem['rhs'].keys()):
            value = elem['rhs'][key]
            # Add the constraint to manage missing values
            if str(row[key]) != value:
                flag = False

        if flag is True:
            # Update the Marked field
            df.loc[index, 'Marked'] += 1


# Add column 'Marked'
df['Marked'] = 0

# Create the list of the selected dependencies
dependencies = []
for i in indexArray:
    dependencies.append(df5.Rule[i])

# Create a copy of the df to count the number of tuples involved by the dependencies
dfCopy = df
for dep in dependencies:
    # For every dependency add one to marked field if the tuple respect the rule
    validates(dfCopy, dep)

dfEthicalProblems = dfCopy[dfCopy['Marked'] > nMarked]
print("\nProblematic tuples: ", len(dfEthicalProblems))
print(dfEthicalProblems.head())

scores = 0
diffs = 0
marks = dfCopy['Marked'].sum()

for i in indexArray:
    scores = scores + df5.Mean[i]
    diffs = diffs + df5.Diff[i]

scoreMean = (scores / len(dependencies))
diffMean = (diffs / len(dependencies))
pMean = 0

dfM = dfCopy[dfCopy['Marked'] != 0]

print("\nNumber of tuples interested by the rules: ", len(dfM), "\nTotal number of tuples: ", len(df),
      "\n\nCumulative Support: ", '%.3f' % (len(dfM) / len(df)), "\nDifference Mean: ", '%.3f' % diffMean)

for attribute in protected_attr:
    deps = 0
    if attribute + 'Diff' in df5:
        for i in indexArray:
            if not (pd.isna(df5[attribute + 'Diff'][i])):
                pMean = pMean + df5[attribute + 'Diff'][i]
                deps += 1
        if pMean != 0:
            pMean = (pMean / deps)
            print(attribute, '- Difference Mean: ', '%.3f' % pMean, "\n")

finalRules = df5[df5.index.isin(indexArray)]
print("\nTotal number of ACFDs selected: ", len(finalRules), "\n", finalRules)
