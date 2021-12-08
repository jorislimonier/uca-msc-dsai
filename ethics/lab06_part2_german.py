import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class German():
    def __init__(self, i_prot):
        self.i_prot = i_prot
        self.load_data()
        self.split_data()
        self.predict_train()

    def load_data(self):
        self.initial_data = pd.read_csv("German.txt", sep="\t", header=None)
        self.data = self.initial_data.copy()
        self.data = self.data.rename({0: "label", self.i_prot: "prot"}, axis=1)

    def split_data(self):
        self.X = self.data.drop(columns=["label", "prot"])
        self.y = self.data["label"]
        self.prot = self.data["prot"]
        self.X_train, self.X_test, self.y_train, self.y_test, self.prot_train, self.prot_test = train_test_split(
            self.X, self.y, self.prot, test_size=.5, shuffle=False)

    def predict_train(self):
        svm_clf = SVC(kernel="linear", C=5)
        svm_clf.fit(self.X_train, self.y_train)
        self.y_pred_train = svm_clf.predict(self.X_train)

    @property
    def eq_opp(self):
        TP_prot = self.y_pred_train[np.all(
            [self.prot_train == 1, self.y_train == 1], axis=0)].sum()
        pos_prot = np.all(
            [self.prot_train == 1, self.y_train == 1], axis=0).sum()

        TPR_prot = TP_prot / pos_prot

        TP_non_prot = self.y_pred_train[np.all(
            [self.prot_train == 0, self.y_train == 1], axis=0)].sum()
        pos_non_prot = np.all(
            [self.prot_train == 0, self.y_train == 1], axis=0).sum()

        TPR_non_prot = TP_non_prot / pos_non_prot
        
        return np.abs(TPR_non_prot - TPR_prot)

    @property
    def pred_eq(self):
        FP_non_prot = self.y_pred_train[np.all(
            [self.prot_train == 0, self.y_train == 0], axis=0)].sum()
        neg_non_prot = np.all(
            [self.prot_train == 0, self.y_train == 0], axis=0).sum()

        FPR_non_prot = FP_non_prot / neg_non_prot

        FP_prot = self.y_pred_train[np.all(
            [self.prot_train == 1, self.y_train == 0], axis=0)].sum()
        neg_prot = np.all(
            [self.prot_train == 1, self.y_train == 0], axis=0).sum()

        FPR_prot = FP_prot / neg_prot

        return np.abs(FPR_non_prot - FPR_prot)

    @property
    def eq_odds(self):
        return self.eq_opp + self.pred_eq

    @property
    def pred_par(self):
        TP_non_prot = self.y_train[np.all(
            [self.prot_train == 0, self.y_pred_train == 1], axis=0)].sum()
        PP_non_prot = np.all(
            [self.prot_train == 0, self.y_pred_train == 1], axis=0).sum()

        PPV_non_prot = TP_non_prot / PP_non_prot

        TP_prot = self.y_train[np.all(
            [self.prot_train == 1, self.y_pred_train == 1], axis=0)].sum()
        PP_prot = np.all(
            [self.prot_train == 1, self.y_pred_train == 1], axis=0).sum()

        PPV_prot = TP_prot / PP_prot

        return np.abs(PPV_prot - PPV_non_prot)

    @property
    def stat_par(self):
        PP_prot = self.y_pred_train[np.all(
            [self.prot_train == 1], axis=0)].sum()
        prot = np.all([self.prot_train == 1], axis=0).sum()

        PPR_prot = PP_prot / prot

        PP_non_prot = self.y_pred_train[np.all(
            [self.prot_train == 0], axis=0)].sum()
        non_prot = np.all([self.prot_train == 0], axis=0).sum()

        PPR_non_prot = PP_non_prot / non_prot

        return np.abs(PPR_prot - PPR_non_prot)

    @property
    def disp_imp(self):
        PP_prot = self.y_pred_train[np.all(
            [self.prot_train == 1], axis=0)].sum()
        prot = np.all([self.prot_train == 1], axis=0).sum()

        PPR_prot = PP_prot / prot

        PP_non_prot = self.y_pred_train[np.all(
            [self.prot_train == 0], axis=0)].sum()
        non_prot = np.all([self.prot_train == 0], axis=0).sum()

        PPR_non_prot = PP_non_prot / non_prot

        return np.minimum(PPR_prot / PPR_non_prot, PP_non_prot / PPR_prot)

    @property
    def fairness_table(self):
        data = {
            "Equal Opportunity": self.eq_opp,
            "Predictive Equality": self.pred_eq,
            "Equalized Odds": self.eq_odds,
            "Predictive Parity": self.pred_par,
            "Statistical Parity": self.stat_par,
            "Disparate Impact": self.disp_imp
        }
        df = pd.DataFrame(data, index=["German Dataset"])
        df.columns = [col + " (%)" for col in df.columns]
        return round(df, 4) * 100
