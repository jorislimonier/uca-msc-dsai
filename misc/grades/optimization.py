# %%
import pandas as pd
import numpy as np
import plotly.express as px

# %%
RAW_DATA = """
		Minitest 1	Minitest 2	Minitest 3	Minitest 4	Minitest 5
BERAS CARABALLO	Julio Enrique					
BIRBIRI	Ufuk Cem	16	12	7	11	5
DE VILLIERS	Marlize					
KACHMAR	Hadi					
LIMONIER	Joris Shan André	20	6	15	18	14
NAHED	Melissa					
NJOMGUE FOTSO	Evariste	9	18	17.5	14	13
RAI	Sourav	16	19	13	4	13
SAOUD	Hamidou	10	12	5	7	17
SARTAJ	Hajam	14	4	7	2	12
YEGHIAZARYAN	Martinos	19	13	2	18	13
YOUSSOUFI	Ayoub					
"""


class Optimization:
    def __init__(self, RAW_DATA=RAW_DATA) -> None:
        self.RAW_DATA = RAW_DATA

    @property
    def data(self):
        raw_data = self.RAW_DATA
        raw_data = raw_data.split("\n")
        raw_data = [row.split("\t") for row in raw_data]
        raw_data = raw_data[2:-1]  # remove header and empty row
        columns = (
            ["last_name", "first_name"]
            # adapt for future new columns
            + [f"test{i}" for i in range(1, len(raw_data[0]) - 1)]
        )
        df = pd.DataFrame(data=raw_data, columns=columns)
        df["name"] = df["last_name"] + " " + df["first_name"]
        df = df.set_index("name")
        df = df.drop(
            columns=["last_name", "first_name"],
        )

        # drop empty cells
        df = df.replace({"": np.nan})
        df = df.dropna(axis=0)

        df = df.astype(float)
        df = df.T
        return df

    @property
    def data_no_worst_grade(self):
        """
        remove minimum grade per student
        """
        df = self.data.copy()
        for col_index in range(len(df.columns)):
            if df.iloc[:, col_index].isna().sum() == 0:
                min_grade_arg = np.argmin(df.iloc[:, col_index])
                df.iloc[min_grade_arg, col_index] = np.nan
            else:
                pass
        return df


opt = Optimization()
opt.data

# %%
px.bar(
    data_frame=opt.data.T.loc[opt.data.mean().sort_values(ascending=False).index],
    barmode="group",
)
# %%
opt.data_no_worst_grade
# %%
df_remove_worst_grade = opt.data_no_worst_grade
px.bar(
    data_frame=df_remove_worst_grade.T.loc[
        df_remove_worst_grade.mean().sort_values(ascending=False).index
    ],
    barmode="group",
)

# %%
px.bar(
    data_frame=opt.data.mean().sort_values(ascending=False),
    title="Simple average",
    text_auto=True,
)
# %%
px.bar(
    data_frame=df_remove_worst_grade.mean().sort_values(ascending=False),
    title="Average without worst grade per student",
    text_auto=True,
)
