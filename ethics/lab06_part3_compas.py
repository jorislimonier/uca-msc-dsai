import pandas as pd
import numpy as np
from random import seed
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


class Compas():
    def __init__(self):
        self.set_seed()
        self.load_data()
        self.split_data()

    @staticmethod
    def set_seed():
        SEED = 1122334455
        seed(SEED)
        np.random.seed(SEED)

    def load_data(self):
        self.initial_data = pd.read_csv("propublica_ext.csv")
        self.data = (
            self.initial_data
            .copy()
            # We first binarize the categorical feature c_charge_degree
            .assign(c_charge=lambda x: x["c_charge_degree"].replace({"F": 1, "M": 0}))
            # race
            .assign(african_american=lambda x: x["race"].replace({"Other": 0, "African-American": 1, "Caucasian": 0, "Hispanic": 0, "Asian": 0, "Native American": 0}))
            .assign(caucasian=lambda x: x["race"].replace({"Other": 0, "African-American": 0, "Caucasian": 1, "Hispanic": 0, "Asian": 0, "Native American": 0}))
            .assign(native_american=lambda x: x["race"].replace({"Other": 0, "African-American": 0, "Caucasian": 0, "Hispanic": 0, "Asian": 0, "Native American": 1}))
            .assign(hispanic=lambda x: x["race"].replace({"Other": 0, "African-American": 0, "Caucasian": 0, "Hispanic": 1, "Asian": 0, "Native American": 0}))
            .assign(asian=lambda x: x["race"].replace({"Other": 0, "African-American": 0, "Caucasian": 0, "Hispanic": 0, "Asian": 1, "Native American": 0}))
            .assign(other=lambda x: x["race"].replace({"Other": 1, "African-American": 0, "Caucasian": 0, "Hispanic": 0, "Asian": 0, "Native American": 0}))
            # age_cat
            .assign(less_than_25=lambda x: x["age_cat"].replace({"Greater than 45": 0, "25 - 45": 0, "Less than 25": 1}))
            .assign(between_25_45=lambda x: x["age_cat"].replace({"Greater than 45": 0, "25 - 45": 1, "Less than 25": 0}))
            .assign(greater_than_25=lambda x: x["age_cat"].replace({"Greater than 45": 1, "25 - 45": 0, "Less than 25": 0}))
            # score_text
            .assign(score_low=lambda x: x["score_text"].replace({"Low": 1, "Medium": 0, "High": 0}))
            .assign(score_medium=lambda x: x["score_text"].replace({"Low": 0, "Medium": 1, "High": 0}))
            .assign(score_high=lambda x: x["score_text"].replace({"Low": 0, "Medium": 0, "High": 1}))
            # sex
            .assign(Male=lambda x: x["sex"].replace({"Male": 1, "Female": 0}))
        )
        DeleteList = ["c_charge_degree", "race", "age_cat",
                      "score_text", "sex", "c_jail_in", "c_jail_out"]
        self.data = self.data.drop(DeleteList, axis=1)

    def split_data(self):
        feature_columns = ["age", "priors_count", "days_b_screening_arrest",
                           "juv_fel_count", "juv_misd_count", "juv_other_count",
                           "is_violent_recid", "c_charge", "less_than_25",
                           "between_25_45", "greater_than_25", "Male"]
        self.X = self.data.copy()
        self.X = self.X[feature_columns]
        self.y = self.data["two_year_recid"].values
        self.data = self.data.assign(
            COMPAS_Decision=lambda x: x["score_low"].replace({0: 1, 1: 0}))
        self.y_compas = self.data["COMPAS_Decision"].values

        self.X_train, self.X_test, self.y_train, self.y_test, self.y_compas_train, self.y_compas_test, self.data_train, self.data_test = train_test_split(
            self.X, self.y, self.y_compas, self.data, train_size=.8, shuffle=False)

    def predict(self):
        svm_clf = SVC(kernel="linear", C=1.0)
        svm_clf.fit(self.X_train, self.y_train)
        self.y_pred = svm_clf.predict(self.X_test)

        b_recid = self.data_test[self.data_test["african_american"] == 1]
        w_recid = self.data_test[self.data_test["caucasian"] == 1]
        print(
            f"Accuracy SVM (All):  \t {metrics.accuracy_score(self.y_test, self.y_pred)*100} \n"
            f"""Accuracy SVM (Black):\t {metrics.accuracy_score(
                self.y_pred[self.data_test["african_american"] == 1], b_recid["two_year_recid"])*100}\n""",
            f"""Accuracy SVM (White):\t {metrics.accuracy_score(
                self.y_pred[self.data_test["caucasian"] == 1], w_recid["two_year_recid"])*100}""",
        )

        display(pd.crosstab(self.y_pred, self.data_test['two_year_recid'], rownames=[
            'Predicted recividism'], colnames=['Actual recividism'], normalize='columns'))
