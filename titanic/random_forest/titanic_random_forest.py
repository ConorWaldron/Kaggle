from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, plot_roc_curve, plot_precision_recall_curve, f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def nan_checker(df, column_name):
    """returns proportion of NAN values in any given column"""
    num_nan = df[column_name].isna().sum()
    return num_nan/len(df)


def data_processing(df):
    '''
    creates X and Y matrix from raw data
    drops columns with little information (PassengerId', 'Name', 'Cabin', 'Ticket')
    drops rows with missing values
    applies one hot encoding to categorical data
    '''
    nan_dict={}  # dictionary key = column name, value = proportion of NaN
    for (column_name, column_data) in df.iteritems():
        nan_dict[column_name]=nan_checker(df, column_name)
    print(f'The missing data fractions are {nan_dict}')

    # Drop columns which are unlikely to be useful
    useful_columns = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

    # Drop Columns which contain missing data (have NaN values), this is primarily due to Age
    complete_data = useful_columns.dropna(axis=0, how='any')
    num_droped_rows = len(df) - len(complete_data)
    print(f'Dropped {num_droped_rows} rows out of a total of {len(df)} rows as they contained NaN')

    # Split data into categorical, binary and numerical
    categorical_data = complete_data[['Pclass', 'Embarked']]
    binary_data = complete_data[['Survived', 'Sex']]
    numerical_data = complete_data[['Age', 'SibSp', 'Parch', 'Fare']]

    # Preprocess binary data
    # binary_data['Sex'].replace({'female': 0, 'male': 1}, inplace=True)  # convert strings to int
    Sex_Int = binary_data['Sex'].map({'male': 0, 'female': 1}) # convert strings to int

    # Preprocess numeric data
    # later on I should try mean centring the data...

    # Preprocess the categorical data
    categorical_data_one_hot_encoded = pd.get_dummies(categorical_data, prefix=['class', 'embark'],
                                                      columns=['Pclass', 'Embarked'], drop_first=True)
    # Recombine to get final input data
    x = categorical_data_one_hot_encoded.join(Sex_Int).join(numerical_data)
    y = binary_data['Survived']
    return x, y


def test_merge(feature_file, answer_file):
    """ merges feature vector and y values from 2 seperate files """
    features = pd.read_csv(feature_file)
    answer = pd.read_csv(answer_file)
    df = features.merge(answer, how='inner', on='PassengerId')
    return df


def train_random_forest(x_train, y_train):
    """ Trains a Random Forest, returns train predictions and probabilities """
    model = RandomForestClassifier()  # using default values for max depth...
    model.fit(x_train, y_train)
    training_predictions = model.predict(x_train)  # returns vectors of 1s and 0s
    training_probabilities = model.predict_proba(x_train)  # returns floats of probablilities
    return model, training_predictions, training_probabilities


def test_model(model, x_test, y_test):
    """ computes test probabilities using a random forest """
    predictions = model.predict_proba(x_test)[:, 1]  # gives prediction for each class, so for binary just select [1]
    predicted_labels = model.predict(x_test)

    plot_roc_curve(model, x_test, y_test)
    plt.show()

    plot_precision_recall_curve(model, x_test, y_test)
    plt.show()

    f1 = f1_score(y_test.values, predicted_labels)
    accuracy = accuracy_score(y_test.values, predicted_labels)
    print('On the test set samples that did not have missing values')
    print(f'the models accuracy was {accuracy} and the F1 score was {f1}')
    fpr, tpr, thresholds = roc_curve(y_test, predictions)  # useful if you want to store values from graph
    return predictions, predicted_labels


def main():
    train_df = pd.read_csv('../data/train.csv')
    test_df = test_merge('../data/test.csv', '../data/gender_submission.csv')

    x_train_data, y_train_data = data_processing(train_df)
    x_test_data, y_test_data = data_processing(test_df)

    random_forest, training_predictions, training_probabilities = train_random_forest(x_train_data, y_train_data)
    test_predictions, test_predicted_labels = test_model(random_forest, x_test_data, y_test_data)

    return y_test_data


if __name__ == '__main__':
    main()
