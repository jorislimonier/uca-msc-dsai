# %%
import pandas as pd
import numpy as np
import plotly.express as px

# %%
RAW_DATA = """
		Minitest 1	Minitest 2	Minitest 3
BERAS CARABALLO	Julio Enrique	19	18	14
BIRBIRI	Ufuk Cem	16	12	7
DE VILLIERS	Marlize			
KACHMAR	Hadi	16		
LIMONIER	Joris Shan André	20	6	15
NAHED	Melissa			
NJOMGUE FOTSO	Evariste	9	18	17.5
RAI	Sourav	16	19	13
SAOUD	Hamidou	10	12	5
SARTAJ	Hajam	14	4	7
YEGHIAZARYAN	Martinos	19	13	2
YOUSSOUFI	Ayoub	15	18	13
"""


def preproc_raw_data(raw_data=RAW_DATA):
    raw_data = raw_data.split("\n")
    raw_data = [row.split("\t") for row in raw_data]
    raw_data = raw_data[2:-1]
    df = pd.DataFrame(
        data=raw_data,
        columns=["last_name", "first_name", "test1", "test2", "test3"],
    )
    df["name"] = df["last_name"] + " " + df["first_name"]
    df = df.set_index("name")
    df = df.drop(
        columns=["last_name", "first_name"],
        index=["DE VILLIERS Marlize", "NAHED Melissa"],
        errors="ignore",
    )

    df = df.replace("", np.nan)
    df = df.astype(float)
    df = df.T
    return df


data = preproc_raw_data()

def remove_worst_grade(df=RAW_DATA):
    df = preproc_raw_data()
    for col_index in range(len(df.columns)):
        if df.iloc[:,col_index].isna().sum() == 0:
            min_grade_arg = np.argmin(data.iloc[:,col_index])
            df.iloc[min_grade_arg, col_index] = np.nan
        else:
            pass
    return df
remove_worst_grade()
# %%
df_remove_worst_grade = remove_worst_grade()
px.bar(
    data_frame=df_remove_worst_grade.T.loc[df_remove_worst_grade.mean().sort_values(ascending=False).index],
    barmode="group",
)

# %%
px.bar(df_remove_worst_grade.mean().sort_values(ascending=False))
px.bar(data.mean().sort_values(ascending=False))