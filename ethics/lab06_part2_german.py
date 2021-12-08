import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class German():
    def __init__(self, i_prot):
        self.i_prot = i_prot
        self.load_data()
        self.split_data()

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

    def predict(self, clf, X_to_pred):
        clf.fit(self.X_train, self.y_train)
        return clf.predict(X_to_pred)
