import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from configuration import EXCLUDE_FROM_SCALING
from data_objects import DesiredModel


class Model:
    def __init__(self, desired_model: str = 'MLP'):
        self.desired_model = DesiredModel(desired_model)
        self.models = {'MLP': [MLPClassifier, {'hidden_layer_sizes': (100), 'max_iter': 300, 'random_state': 1}],
                       'Forest': [RandomForestClassifier, {'n_estimators': 200, 'random_state': 666, 'criterion': 'entropy'}],
                       'SVM': [SVC, {'random_state': 666}]}

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, oversampling: bool):
        """
        This function trains the chosen model.

        :param x_train: regrssors dataframe
        :param y_train: target dataframe
        :param oversampling: boolean
        :return: a fitted classifer
        """
        balanced_x, y_train = self._handle_imbalanced_data(x_train, y_train, over=oversampling)
        self.data_scaler_fit(balanced_x)
        x_train_scaled = self._scale(balanced_x)
        balanced_x = np.concatenate([x_train_scaled, balanced_x[EXCLUDE_FROM_SCALING]], axis=1)
        # balanced_x = pd.concat([pd.DataFrame(x_train_scaled), balanced_x[EXCLUDE_FROM_SCALING]], axis=1)
        print(f"Training the {self.desired_model.value} classifier")
        classifier = self.models[self.desired_model.value][0]()
        classifier.set_params(**self.models[self.desired_model.value][1])
        classifier.fit(balanced_x, y_train)
        print(f"Model {self.desired_model.value} is trained correctly")
        return classifier

    @staticmethod
    def _handle_imbalanced_data(x: pd.DataFrame, y: pd.DataFrame, over: bool = True):
        """
        Handles the imbalanced data by applying over/under sampling

        :param x: regressors dataframe
        :param y: target dataframe
        :param over: bool
        :return: balanced x and y
        """
        if not over:
            print('Performing undersampling')
            undersample = RandomUnderSampler(sampling_strategy='majority', random_state=9634)
            x_train, y_train = undersample.fit_resample(x, y)
        else:
            print('Performing oversampling')
            oversample = RandomOverSampler(sampling_strategy='minority', random_state=9634)
            x_train, y_train = oversample.fit_resample(x, y)
        return x_train, y_train

    def predict(self, input_vector: np.ndarray):
        """
        Predict function loads the trained model and predicts the input data
        :param input_vector: input numpy array for prediction
        :return: predicted outcome
        """
        loaded_model = self.load_model(self.desired_model.value)
        return loaded_model.predict(input_vector)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Evaluation function calls the predict fucntion on the chosen trained model
        evaluates the data on the testset for inference

        :param x_test: regressors testset dataframe
        :param y_test: targets testset dataframe
        :return: Dictionary with evaluation metrics
        """
        x_test_scaled = self._scale(x_test)
        data_for_testing = np.concatenate([x_test_scaled, x_test[EXCLUDE_FROM_SCALING]], axis=1)
        test_prediction = self.predict(input_vector=data_for_testing)
        np.set_printoptions(threshold=np.inf)
        pr, rec, fscore, support = precision_recall_fscore_support(y_test, test_prediction, average='binary')
        return {'Model': self.desired_model.value,
                'Accuracy score': accuracy_score(y_test, test_prediction),
                'Precision score': pr,
                'Recall score': rec,
                'F1 score': fscore}

    @staticmethod
    def data_scaler_fit(x: pd.DataFrame, scale_range: tuple = (-2, 2)):
        """
        Data scaler performs scaling on input data based on MixMaxScaler from sklearn
        :param x: input dataframe
        :param scale_range: tuple with the clipping range for the MinMax Scaler
        :return: None
        """
        scaler = MinMaxScaler(feature_range=scale_range)
        x_excluded_columns = x.loc[:, ~x.columns.isin(EXCLUDE_FROM_SCALING)]
        print('Shape for scaling fit: ', x_excluded_columns.shape)
        scaler.fit(x_excluded_columns)
        pickle.dump(scaler, open('../trained_models/scaler.pickle', 'wb'))
        print('Scaler is fitted and saved')

    @staticmethod
    def _scale(data_to_be_scaled: pd.DataFrame):
        print('Loading scaler and transform data')
        loaded_scaler = pickle.load(open('../trained_models/scaler.pickle', 'rb'))
        return loaded_scaler.transform(data_to_be_scaled.loc[:, ~data_to_be_scaled.columns.isin(EXCLUDE_FROM_SCALING)])

    def save_model(self,
                   model_object,
                   model_name: str,
                   path: str = '../trained_models'):
        path_name = "/".join([path, model_name]) + ".pickle"
        pickle.dump(model_object, open(path_name, 'wb'))
        print(f'The fitted model: {self.desired_model} is saved at {path_name}')
        return True

    @staticmethod
    def load_model(model_name: str,
                   path: str = '../trained_models'):
        path_name = "/".join([path, model_name]) + ".pickle"
        loaded_model = pickle.load(open(path_name, 'rb'))
        return loaded_model
