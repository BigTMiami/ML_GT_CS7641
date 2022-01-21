import pandas as pd

survey_csv = "data/survey/gss_modified.csv"

df = pd.read_csv(survey_csv)
df.shape

df.columns

for index, col in enumerate(df.columns):
    print(f"{index}:{col}")
    if index > 1000:
        break


pd.options.display.max_rows = 50
pd.options.display.max_columns = 50

f = open(survey_csv)
line = f.readline()
print(line)
l2 = f.readline()
l3 = f.readline()

print(l3)

df.groupby(["RACE_OF_RESPONDENT"])["RACE_OF_RESPONDENT"].count()

df["AGE_WHEN_FIRST_MARRIED"].unique()

f.close()

with open(survey_csv) as f:
    headers = []
