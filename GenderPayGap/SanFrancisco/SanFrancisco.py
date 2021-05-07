import pandas as pd
import gender_guesser.detector as gender

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.float_format', lambda f: '%.2f' % f)

df = pd.read_csv('SanFrancisco.csv')

# Renaming columns and attributes to get the same attributes names among all the CSV files
df.rename(columns={'Employee Name': 'Name', 'Total Pay': 'Annual Salary'}, inplace=True)
df["Status"].replace({"FT": "F", "PT": "P"}, inplace=True)

# Keeping just the tuples related to 2019
df = df[df['Year'] == 2019]

# Getting First Name of employees
df['First Name'] = df['Name'].str.split().str[0]

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
df = df.dropna(how='any', subset=['Annual Salary'])
df = df[df['Annual Salary'] > 0]

print("\n# Filtered DataFrame (more than 100 Job Title occurrences) #"
      "\nMales: ", len(df[df['Gender'] == 'male']), "\t\tFemales: ", len(df[df['Gender'] == 'female']),
      "\nNumber of different Job Titles: ", df['Job Title'].nunique(), "\n")
#print(df.groupby(['Job Title', 'Gender']).size(), "\n")

print(df)

# Exporting CSV
df.to_csv(r'SanFrancisco_Adjusted.csv', index=False)
