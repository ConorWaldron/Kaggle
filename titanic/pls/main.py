import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../..")


class PLSclassifier:

    def __init__(self, train_path, test_path, survived_map_path):

        # Load datasets and perform precleaning
        self.df_train = self.preprocessing(pd.read_csv(train_path))
        self.df_test = self.preprocessing(pd.read_csv(test_path), survived_map_path=survived_map_path)

        # Train a feature normaliser to mean-center and scale features
        # This is required to apply effectively the PLS
        self.features_X = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
        self.features_y = ["Survived"]
        self.all_features = self.features_X + self.features_y

        self.normaliser = StandardScaler()
        self.normaliser_train()
        self.X_train, self.y_train, self.survived_train = self.normaliser_transform(self.df_train)
        self.X_test, self.y_test, self.survived_test = self.normaliser_transform(self.df_test)

        # Initialise PLS and logistic model
        self.PLS = None
        self.LR = None

    def preprocessing(self, df, survived_map_path=None):
        df = df.dropna().reset_index(drop=True)
        sex_mapping = {"male": -1, "female": 1}
        df["Sex"] = df["Sex"].apply(lambda x: sex_mapping[x])
        if "Survived" not in df.columns and survived_map_path:
            df_map = pd.read_csv(survived_map_path)
            id2survived = {row["PassengerId"]: row["Survived"] for _, row in df_map.iterrows()}
            df["Survived"] = df["PassengerId"].apply(lambda x: id2survived[x])
        return df

    def normaliser_train(self):
        data = self.df_train[self.all_features].values
        self.normaliser.fit(data)

    def normaliser_transform(self, df):
        df_transformed_data = pd.DataFrame(self.normaliser.transform(df[self.all_features]),
                                           columns=self.all_features)
        X = df_transformed_data[self.features_X].values
        y = df_transformed_data[self.features_y].values
        survived = df[self.features_y].values
        return X, y, survived

    def train(self, n_components):
        self.PLS = PLSRegression(n_components=n_components)
        self.PLS.fit(self.X_train, self.y_train)
        y_scores = self.PLS.predict(self.X_train)
        self.LR = LogisticRegression().fit(y_scores, self.survived_train)

    def predict(self):
        y_proba = self.LR.predict_proba(self.PLS.predict(self.X_test))
        y_true = self.survived_test.ravel()
        y_pred = [1 if score > 0.5 else 0 for score in y_proba[:, 1]]
        print(classification_report(y_true, y_pred, target_names=["Died", "Survived"]))
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        self.plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":

    train_path = "titanic/data/train.csv"
    test_path = "titanic/data/test.csv"
    survived_map_path = "titanic/data/gender_submission.csv"
    pls = PLSclassifier(train_path=train_path,
                        test_path=test_path,
                        survived_map_path=survived_map_path)

    pls.train(n_components=2)
    pls.predict()
