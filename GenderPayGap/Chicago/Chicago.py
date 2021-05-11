import pandas as pd
import numpy as np
import gender_guesser.detector as gender

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.float_format', lambda f: '%.2f' % f)

df = pd.read_csv('Chicago.csv')

# Renaming columns and attributes to get the same attributes names among all the CSV files
df.rename(columns={'Job Titles': 'Job Title', 'Full or Part-Time': 'Status'}, inplace=True)

# Estimating Annual Salary for hourly employees
df['Annual Salary'] = np.where(df['Annual Salary'].isnull(),
                               df['Typical Hours'] * df['Hourly Rate'] * 52, df['Annual Salary'])

# Getting First Name of employees
df['First Name'] = df['Name'].str.split(',').str[1]
df['First Name'] = df['First Name'].str.split().str[0]

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
del df['Typical Hours'], df['Hourly Rate'], df['First Name']

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

# Removing Part-Time employees (as suggested in the Glassdoor report)
#df = df[df['Status'] == 'F']

print("\n# Filtered DataFrame (more than 100 Job Title occurrences) #"
      "\nMales: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']),
      "\nNumber of different Job Titles: ", df['Job Title'].nunique(), "\n")
#print(df.groupby(['Job Title', 'Gender']).size(), "\n")

print(df)

# Exporting CSV
df.to_csv(r'Chicago_Adjusted.csv', index=False)

# Generating biased dataset (dividing females' Annual Salary by 2)
df.loc[df['Gender'] == 'female', 'Annual Salary'] = df.loc[df['Gender'] == 'female', 'Annual Salary'] / 2

print(df)

# Exporting CSV
df.to_csv(r'Chicago_Biased.csv', index=False)

df = pd.read_csv('Chicago_Adjusted.csv')

# Grouping Job Titles
g1 = ['CAPTAIN-EMT', 'LIEUTENANT', 'LIEUTENANT-EMT']
g2 = ['LIBRARIAN I', 'OPERATING ENGINEER-GROUP A', 'OPERATING ENGINEER-GROUP C']
g3 = ['FIRE ENGINEER-EMT', 'FIREFIGHTER', 'FIREFIGHTER-EMT', 'FIREFIGHTER-EMT (RECRUIT)', 'FIREFIGHTER/PARAMEDIC',
      'PARAMEDIC', 'PARAMEDIC I/C', 'POLICE OFFICER', 'POLICE OFFICER (ASSIGNED AS DETECTIVE)',
      'POLICE OFFICER (ASSIGNED AS EVIDENCE TECHNICIAN)', 'POLICE OFFICER / FLD TRNG OFFICER', 'SERGEANT']
g4 = ['ADMINISTRATIVE ASST II', 'LIBRARY PAGE', 'POLICE COMMUNICATIONS OPERATOR I', 'POLICE COMMUNICATIONS OPERATOR II']
g5 = ['ELECTRICAL MECHANIC', 'HOISTING ENGINEER', 'PLUMBER']
g6 = ['DETENTION AIDE', 'FOSTER GRANDPARENT', 'SANITATION LABORER']
g7 = ['TRAFFIC CONTROL AIDE-HOURLY']
g8 = ['MACHINIST (AUTOMOTIVE)', 'MOTOR TRUCK DRIVER', 'POOL MOTOR TRUCK DRIVER']
g9 = ['AVIATION SECURITY OFFICER', 'CONSTRUCTION LABORER', 'GENERAL LABORER - DSS']
for i in range(len(df)):
    if df.loc[i, 'Job Title'] in g1:
        df.loc[i, 'Job Title'] = 1
    if df.loc[i, 'Job Title'] in g2:
        df.loc[i, 'Job Title'] = 2
    if df.loc[i, 'Job Title'] in g3:
        df.loc[i, 'Job Title'] = 3
    if df.loc[i, 'Job Title'] in g4:
        df.loc[i, 'Job Title'] = 4
    if df.loc[i, 'Job Title'] in g5:
        df.loc[i, 'Job Title'] = 5
    if df.loc[i, 'Job Title'] in g6:
        df.loc[i, 'Job Title'] = 6
    if df.loc[i, 'Job Title'] in g7:
        df.loc[i, 'Job Title'] = 7
    if df.loc[i, 'Job Title'] in g8:
        df.loc[i, 'Job Title'] = 8
    if df.loc[i, 'Job Title'] in g9:
        df.loc[i, 'Job Title'] = 9

print(df)

# Exporting CSV
df.to_csv(r'Chicago_Grouped.csv', index=False)
