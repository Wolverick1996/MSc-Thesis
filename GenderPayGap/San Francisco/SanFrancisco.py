import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import copy

'''
import gender_guesser.detector as gender
'''

'''
from sklearn.preprocessing import LabelEncoder
'''

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.float_format', lambda f: '%.2f' % f)

'''
df = pd.read_csv('SanFrancisco.csv')

# Keeping just the tuples related to 2019
df = df[df['Year'] == 2019]

# Getting First Name of employees
df['First Name'] = df['Employee Name'].str.split().str[0]

# Inferring Gender
df['Gender'] = df.index
d = gender.Detector(case_sensitive=False)
males = 0
females = 0
andy = 0
mostly_males = 0
mostly_females = 0
unknown = 0
for i in range(len(df)):
    df.loc[i, 'Gender'] = d.get_gender(df.loc[i, 'First Name'])
    if df.loc[i, 'Gender'] == 'male':
        males += 1
    if df.loc[i, 'Gender'] == 'female':
        females += 1
    if df.loc[i, 'Gender'] == 'andy':
        andy += 1
        #df.loc[i, 'Gender'] = random.choice(['male', 'female'])
    if df.loc[i, 'Gender'] == 'mostly_male':
        mostly_males += 1
        df.loc[i, 'Gender'] = 'male'
    if df.loc[i, 'Gender'] == 'mostly_female':
        mostly_females += 1
        df.loc[i, 'Gender'] = 'female'
    if df.loc[i, 'Gender'] == 'unknown':
        unknown += 1
        #df.loc[i, 'Gender'] = random.choice(['male', 'female'])

# Removing useless columns
del df['Base Pay'], df['Overtime Pay'], df['Other Pay'], df['Benefits'], df['Total Pay & Benefits'],\
    df['Year'], df['First Name']

# Removing tuples related to Andy or Unknown names
df = df[(df['Gender'] != 'andy') & (df['Gender'] != 'unknown')]

print(df, "\nMales: ", males, "\t\tFemales: ", females, "\t\tAndy: ", andy,
      "\t\tMostly Males: ", mostly_males, "\t\tMostly Females: ", mostly_females, "\t\tUnknown: ", unknown,
      "\nMostly Males are inferred to be Males, Mostly Females are inferred to be Females, "
      "Andy and Unknown are deleted from the DataFrame\n\n"
      "=> Males: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']))

print("Number of different Job Titles (total): ", df['Job Title'].nunique())

# Removing Job Titles with less than 100 occurrences
df = df.groupby('Job Title').filter(lambda o: len(o) > 100)

# Removing the tuples with a NaN Total Pay or Total Pay == 0
df = df.dropna(how='any', subset=['Total Pay'])
df = df[df['Total Pay'] > 0]

print("\n# Filtered DataFrame (more than 100 Job Title occurrences) #"
      "\nMales: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']),
      "\nNumber of different Job Titles: ", df['Job Title'].nunique(), "\n")
#print(df.groupby(['Job Title', 'Gender']).size(), "\n")

# Exporting CSV
df.to_csv(r'SanFrancisco_Adjusted.csv', index=False)
'''

# --- STATISTICAL ANALYSIS --- #
print("--- STATISTICAL ANALYSIS ---")

# Get the Data Ready #
df = pd.read_csv('SanFrancisco_Adjusted.csv')

print(df, "\n\nMales: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']),
      "\nNumber of different Job Titles (total): ", df['Job Title'].nunique(), "\n")

# Take the natural logarithm of Total Pay (for percentage pay gap interpretation in regressions)
df['Log Total Pay'] = np.log(df['Total Pay'])
# Create dummy indicator for gender (male = 1, female = 0).
df['Male'] = np.where(df['Gender'] == 'male', 1, 0)

# Look at the Data #
# Create an overall table of summary statistics for the data
print(df.describe())

# Create a table showing overall male-female pay differences in Annual Salary
print("\n", pd.pivot_table(df, index=["Gender"], values=["Total Pay"], aggfunc=[np.average, np.median, len]))

# How are employees spread out among Job Titles?
pd.set_option('display.max_rows', None)
print("\n", pd.pivot_table(df, index=["Job Title", "Gender"], values=["Total Pay"], aggfunc=[np.average, len]))

# Run Your Regressions #
t = PrettyTable()
t.add_column('', ['Male', 'Job Title', 'Status', 'Constant',
                  'Controls\n   - Job Title\n   - Status', 'Observations', 'R^2'])

# No controls ("unadjusted" pay gap)
x = df['Male'].values.reshape(-1, 1)
y = df['Log Total Pay']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient = model.coef_[0]
t.add_column('(1)', ['%.3f' % coefficient, '', '', '%.3f' % constant, '\nNo\nNo', len(df), '%.3f' % r_sq])

# Add controls for Job Title and Status ("adjusted" pay gap)
x = df[['Male', 'Job Title', 'Status']]
x = pd.get_dummies(data=x, drop_first=True)
y = df['Log Total Pay']
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
constant = model.intercept_
coefficient0 = model.coef_[0]
coefficient1 = model.coef_[1]
coefficient2 = model.coef_[2]
t.add_column('(2)', ['%.3f' % coefficient0, '%.3f' % coefficient1, '%.3f' % coefficient2,
                     '%.3f' % constant, '\nYes\nYes', len(df), '%.3f' % r_sq])

# Publish a clean table of regression results
print('\nDependent Variable: Log Total Pay\n', t,
      "\n'Unadjusted' pay gap: men on average earn", '%.1f' % (coefficient * 100), "% more than women",
      "\n'Adjusted' pay gap: men on average earn", '%.1f' % (coefficient0 * 100),
      "% more than women (not statistically significant)")

'''
# --- RankingFacts --- #
# Get the Data Ready #
dfRF = pd.read_csv('SanFrancisco_Adjusted.csv')

# Convert categorical to numerical attributes (RankingFacts use numerical attributes)
dfRF['Status'] = np.where(dfRF['Status'] == 'FT', 1, 0)  # (FT = 1, PT = 0)
label_encoder = LabelEncoder()
label_encoder.fit(dfRF['Job Title'])
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nEncoding Job Title:\n", label_encoder_name_mapping)
dfRF['Job Title'] = label_encoder.fit_transform(dfRF['Job Title'])

# Exporting CSV
dfRF.to_csv(r'SanFrancisco_RankingFacts.csv', index=False)
quit()
'''

# --- FAIR-DB --- #
print("\n\n--- FAIR-DB ---")

pd.set_option('display.max_rows', 10)
print(df)

# Data Acquisition #
# Create employee income bins (numbers are not really useful in estimating correlations)
#labels = ["0-39", "40-59", "60-79", "80-99", "100-119", "120-139", "140+"]
#bins = [0, 39.99, 59.99, 79.99, 99.99, 119.99, 139.99, 500]
labels = ["< 90K", "≥ 90K"]
bins = [0, 89.99, 500]
#df['Total Pay'] = df['Total Pay'] / 1000
df['Total Pay Bin'] = pd.cut(df['Total Pay'] / 1000, bins=bins, labels=labels)

# Plotting distribution of Annual Salary over Gender
df.hist(column='Total Pay')
d = pd.crosstab(df['Total Pay Bin'], df['Gender'])
d.plot.bar(stacked=True, color=['hotpink', 'dodgerblue'])
plt.xticks(rotation=45)
plt.tight_layout()

# Get the frequency, PMF (Probability Mass Function) and CDF (Cumulative Distribution Function)
# for each value in the Annual Salary column
stats_df = df.groupby('Total Pay')['Total Pay']\
    .agg('count').pipe(pd.DataFrame).rename(columns={'Total Pay': 'frequency'})
stats_df['pmf'] = stats_df['frequency'] / sum(stats_df['frequency'])
stats_df['cdf'] = stats_df['pmf'].cumsum()
stats_df = stats_df.reset_index()
stats_df.plot(x='Total Pay', y=['pmf', 'cdf'], grid=True)

plt.show()

# Removing useless columns
del df['Employee Name'], df['Total Pay'], df['Log Total Pay'], df['Male']

all_tuples = len(df)
cols = df.columns

# Because there are numbers in the DataFrame, I transform them into strings to perform comparisons in completion phase
df = df.applymap(lambda x: str(x) if type(x) == int else x)

print("\nTotal number of tuples in df: ", len(df), "\n")
print(df.head())

# INPUTS
# array of protected attributes
protected_attr = ['Gender']
# target class
target = 'Total Pay Bin'
binaryValues = df['Total Pay Bin'].unique()
print("\nTotal Pay Bin values: ", binaryValues)
print("\nStatus values: ", df['Status'].unique())

# input parameters
confidence = 0.8
supportCount = 100  # previously 2100
support = supportCount / len(df)
maxSize = 2
grepValue = target + '='
minDiff = 0.02

# Removing ',' character to avoid problems with CFDDiscovery
df['Job Title'] = df['Job Title'].str.replace(', ', ' ')

# Exporting CSV
#df.to_csv(r'SanFrancisco_FAIR-DB.csv', index=False, line_terminator=',\n')
#df.to_csv(r'SanFrancisco_FAIR-DB_2bins.csv', index=False, line_terminator=',\n')

# ACFDs Discovery and Filtering #
# Apply CFDDiscovery algorithm ->
# https://codeocean.com/capsule/6146641/tree (original) | https://codeocean.com/capsule/5820643/tree (copied)
with open('cfds[s=100_FPT_2bins].txt') as f:
    output = f.read().splitlines()

# all rules obtained
print("\nTotal number of dependencies found: ", len(output))

for i in range(0, 8):
    if len(output) > i:
        print("Dependency n.", i, ":\t", output[i])

# Transform the '<=' in '<' and viceversa to avoid problems with the following '=' detection
o1 = list(map(lambda x: x.replace("<=", "<"), output))
# Delete the parenthesis
o1 = list(map(lambda x: x.replace("(", ""), o1))
o1 = list(map(lambda x: x.replace(")", ""), o1))
# Split the entire rule in a lhs and rhs
o2 = list(map(lambda x: x.split(' => '), o1))


# Function to select only CFDs from all rules
# x is the single rule
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
    # Flag indicates if the rule contains unwanted condition(s) (rhs or lhs) - it doesn't contain the condition (True)
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


# To create an array that contains all rules with the lhs and rhs separated
def createSplitting(elem):
    elemlhs = elem[0]
    elemrhs = elem[1]
    LHS = splitElem(elemlhs)
    RHS = splitElem(elemrhs)
    return [LHS, RHS]


# Now that we have deleted all the '=' we can replace the "<" with '<='
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
    # How many times appears the lhs of the rule
    countX = 0
    # How many times appears the rhs of the rule
    countY = 0
    # How many times appears the entire rule
    countXY = 0

    # for every row of the database, count the LHS, RHS and the total count
    for index, row in df.iterrows():
        # The flags help in dealing with missing values
        flagX = True
        flagY = True

        for key in list(elem['lhs'].keys()):
            value = elem['lhs'][key]

            # add the constraint to manage missing values
            if str(row[key]) != value:
                flagX = False

        for key in list(elem['rhs'].keys()):
            value = elem['rhs'][key]

            # add the constraint to manage missing values
            if str(row[key]) != value:
                flagY = False

        if flagX:
            # increase the LHS support count
            countX += 1
        if flagY:
            # increase the RHS support count
            countY += 1
        if flagX and flagY:
            # increase the entire rule support count
            countXY += 1

    # return the LHS supp count, RHS supp count and the entire rule supp count
    return countX, countY, countXY


def computeConfidenceNoProtectedAttr(elem):
    filteredRule = {'lhs': {k: v for k, v in elem['lhs'].items()
                            if ((k not in protected_attr) and (k != target))}, 'rhs': elem['rhs']}

    fCount = countOccur(filteredRule)
    # if the rule is valid for at least one tuple
    if fCount[2] != 0 and fCount[0] != 0:
        ratio = fCount[2] / fCount[0]
    else:
        ratio = 0
    return ratio


def computeConfidenceForProtectedAttr(elem, protAttr):
    filteredRule = {'lhs': {k: v for k, v in elem['lhs'].items() if (k != protAttr)}, 'rhs': elem['rhs']}

    fCount = countOccur(filteredRule)
    # if the rule is valid for at least one tuple
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

            # compute the diff for each protected  attributes
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

            # for handling case having cartesian product first time of two sets
            if type(set_a[i]) != list:
                set_a[i] = [set_a[i]]

            # coping all the members of set_a to temp
            temp = [num for num in set_a[i]]

            # add member of set_b to temp to have cartesian product
            temp.append(set_b[j])
            result.append(temp)

    return result


def Cartesian(list_a, n):
    # result of cartesian product of all the sets taken two at a time
    temp = list_a[0]

    # do product of N sets
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
    # select DB according to already set attributes
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
        mat = Cartesian(perm, len(perm))
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
    # for every rule compute the combinations over the protected attribute
    rulesCount = findCFDsCombinations(elem)
    for ar in rulesCount:
        CFDCombinations.append(ar)
# removing duplicate combinations
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

# Order the rules by Diff or Support or both
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
# indexes of the selected rules
#indexArray = [51, 40, 32, 43, 53, 54] with 8 bins
indexArray = [17, 14, 21, 10, 18]
#indexArray = list(df5.index.values)

# minimum number of rules necessary to have a problematic tuple
nMarked = 0


# for every rule = elem, iter over all rows and add one if the tuple respect the rule
def validates(df, elem):
    for index, row in df.iterrows():
        flag = True
        for key in list(elem['lhs'].keys()):
            value = elem['lhs'][key]
            # add the constraint to manage '?' that could be a missing values
            if str(row[key]) != value:
                flag = False

        for key in list(elem['rhs'].keys()):
            value = elem['rhs'][key]
            # add the constraint to manage missing values
            if str(row[key]) != value:
                flag = False

        if flag is True:
            # update the Marked field
            df.loc[index, 'Marked'] += 1


# add column 'Marked'
# add one column to count the number of tuples involved by the dependencies
df['Marked'] = 0

# create the list of the selected dependencies
dependencies = []
for i in indexArray:
    dependencies.append(df5.Rule[i])

# create a copy of the df to count the number of tuples involved by the dependencies
dfCopy = df
for dep in dependencies:
    # for every dependency add one to marked field if the tuple respect the rule
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
