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
        self.y = self.data["two_year_recid"]
        self.data = self.data.assign(
            COMPAS_Decision=lambda x: x["score_low"].replace({0: 1, 1: 0}))
        self.y_compas = self.data["COMPAS_Decision"]

        random_state = np.random.randint(10**9) # get same random state for all splits
        self.X_train, self.X_test = train_test_split(self.X,
                                                     train_size=.7,
                                                     shuffle=True,
                                                     random_state=random_state)
        self.y_train, self.y_test = train_test_split(self.y,
                                                     train_size=.7,
                                                     shuffle=True,
                                                     random_state=random_state)
        self.y_compas_train, self.y_compas_test = train_test_split(self.y_compas,
                                                                   train_size=.7,
                                                                   shuffle=True,
                                                                   random_state=random_state)
        self.data_train, self.data_test = train_test_split(self.data,
                                                           train_size=.7,
                                                           shuffle=True,
                                                           random_state=random_state)

    def predict_svm(self):
        """Classify `y_train` and assign to `y_pred`
        Subsequently, compute confusion matrix and accuracy by race.
        """
        self.svm_clf = SVC(kernel="linear", C=1.0)
        self.svm_clf.fit(self.X_train, self.y_train)
        self.y_pred = self.svm_clf.predict(self.X_test)
        self.confusion_matrix_svm()
        self.race_accuracy_svm()

    def confusion_matrix_svm(self):
        "Display prediction metrics from the svm classifier"

        conf_mat = pd.crosstab(self.y_pred, self.data_test['two_year_recid'],
                               rownames=['Predicted recividism'],
                               colnames=['Actual recividism'],
                               normalize='columns')
        FPR_s = conf_mat[0][1]
        FNR_s = conf_mat[1][0]

        print(f"FPR SVM = {FPR_s}")
        print(f"FNR SVM = {FNR_s}")
        self.conf_mat_svm = conf_mat

    def race_accuracy_svm(self):
        "Return accuracy for white, black and altogether"
        b_recid = self.data_test[self.data_test["african_american"] == 1]
        w_recid = self.data_test[self.data_test["caucasian"] == 1]

        accuracies = {
            "All": metrics.accuracy_score(self.y_test, self.y_pred),
            "Black": metrics.accuracy_score(self.y_pred[self.data_test["african_american"] == 1],
                                            b_recid["two_year_recid"]),
            "White": metrics.accuracy_score(self.y_pred[self.data_test["caucasian"] == 1],
                                            w_recid["two_year_recid"])
        }
        self.race_acc_svm = pd.DataFrame(accuracies, index=["SVM accuracy"])

    def confusion_matrix_compas(self, normalize=False):
        return pd.crosstab(self.data['COMPAS_Decision'],
                           self.data['two_year_recid'],
                           rownames=['Predicted recividism'],
                           colnames=['Actual recividism'],
                           normalize=normalize)

    def confusion_matrix_compas_black(self, normalize=False):
        "Confusion  matrix for black defendents"
        b_recid = self.data_test[self.data_test["african_american"] == 1]
        return pd.crosstab(b_recid['COMPAS_Decision'],
                           b_recid['two_year_recid'],
                           rownames=['Predicted recividism'],
                           colnames=['Actual recividism'],
                           normalize=normalize)

    def confusion_matrix_compas_white(self, normalize=False):
        "Confusion  matrix for white defendents"
        w_recid = self.data_test[self.data_test["caucasian"] == 1]
        return pd.crosstab(w_recid['COMPAS_Decision'],
                           w_recid['two_year_recid'],
                           rownames=['Predicted recividism'],
                           colnames=['Actual recividism'],
                           normalize=normalize)
