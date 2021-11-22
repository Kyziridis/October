import re

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split

from data_objects import RequestInputData
from configuration import COLUMNS_TO_EXCLUDE, KEEP_ID


class DataManager:
    """
    Class manipulates the data

    """
    def __init__(self, path_to_data: str, response_column_name: str = None):
        self.path_to_data = path_to_data
        self.response_column_name = response_column_name
        self.data = self._load_data_to_memory()
        self._check_duplicate_customers()

    def _load_data_to_memory(self):
        return pd.read_csv(self.path_to_data)

    def _check_duplicate_customers(self, customer_col: str = 'customer_id'):
        is_unique_customers = len(self.data[customer_col].unique()) == self.data.shape[0]
        print(f'All clients unique in dataset : {is_unique_customers}')

    def get_preprocessed_data(self):
        """
        Calls _preprocess function and returns the preprocessed data
        :return: preprocessed data
        """
        preprocessed_data = self._preprocess()
        return preprocessed_data

    @staticmethod
    def _remove_inf_nan_from_data(data: pd.DataFrame):
        indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
        return data[indices_to_keep]

    @staticmethod
    def _remove_all_id_columns(data: pd.DataFrame):
        catch_id = re.compile(r'.*id$')
        columns_without_id = [colname for colname in data.columns if not catch_id.match(colname)]
        return data[columns_without_id]

    def _preprocess(self):
        """
        Preprocess applying all essential steps on data
        Constructs features and log transformations
        Apply the binning on categorical variables
        Cleans the data remove nans and excluded columns
        :return: preprocessed data
        """
        processed_data = self._transform_data(self.data)
        processed_data = self._make_age_bins(processed_data)
        processed_data = self._handle_employment_type(processed_data)
        processed_data = processed_data.loc[:, ~processed_data.columns.isin(COLUMNS_TO_EXCLUDE)]
        processed_data = self._remove_inf_nan_from_data(processed_data)
        final = self._remove_all_id_columns(processed_data)
        # final = processed_data.drop(COLUMNS_TO_EXCLUDE, axis=1)
        print('Final data shape:', final.shape)
        print('Final data columns to be kept:\n', final.columns)
        return final

    @staticmethod
    def _make_age_bins(data: pd.DataFrame):
        data_copy = data.copy()
        data_copy.loc[data_copy['age'] >= 30, 'age'] = 1
        data_copy.loc[data_copy['age'] != 1, 'age'] = 0
        return data_copy

    @staticmethod
    def _handle_employment_type(data: pd.DataFrame):
        data_copy = data.copy()
        data_copy = pd.get_dummies(data_copy, prefix=['employment_type'], columns=['employment_type'])
        return data_copy

    @staticmethod
    def _transform_data(data: pd.DataFrame):
        data_copy = data.copy()
        data_copy['overdue_to_active_ratio'] = (data_copy['total_overdue_no'] + 1) / (data_copy['main_account_active_loan_no'] + data_copy['sub_account_active_loan_no'] + 1)
        data_copy['overdue_to_total_ratio'] = (data_copy['total_overdue_no'] + 1) / (data_copy['total_account_loan_no'] + 1)
        data_copy['monthly_payment_to_outstanding_ratio'] = (data_copy['total_monthly_payment'] + 1) / (
                    data_copy['total_outstanding_loan'] + 1)
        data_copy['outstanding_to_disburse_ratio'] = (data_copy['total_outstanding_loan'] + 1) / (
                    data_copy['total_disbursed_loan'] + 1)
        data_copy['total_outstanding_loan'] = np.log(data_copy['total_outstanding_loan'] + 1)
        data_copy['total_disbursed_loan'] = np.log(data_copy['total_disbursed_loan'] + 1)
        data_copy['total_monthly_payment'] = np.log(data_copy['total_monthly_payment'] + 1)
        return data_copy

    def split_train_test(self, data: pd.DataFrame, testset_size: float = 0.2):
        """
        Train test splitting
        :param data: dataframe to be splitted
        :param testset_size: ratio float
        :return: regressprs train and test, target train and test
        """
        x, y = self._split_x_y(data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_size, random_state=36232)
        return x_train, x_test, y_train, y_test

    def _split_x_y(self, data: pd.DataFrame):
        y_column = 'loan_default'
        if self.response_column_name:
            y_column = self.response_column_name
        regressors_x = data.loc[:, data.columns != y_column]
        target_y = data[y_column]
        return regressors_x, target_y

    def return_head(self):
        return self.data.head().to_html()

    @staticmethod
    def explore(data: pd.DataFrame):
        return data.describe().to_html()

    def handle_request_data(self, input_request: RequestInputData):
        """
        Handles the input requested data for the predict endpoint.
        Preprocessed the data in order to feed them in the model.
        :param input_request: data to be predicted (x data)
        :return: dataframe ready to be fitted
        """
        df = pd.DataFrame([input_request.dict()])
        processed_data = self._transform_data(df)
        processed_data = self._make_age_bins(processed_data)
        # processed_data = self._handle_employment_type(processed_data)
        employment_type_hot = {'employment_type_0': 0, 'employment_type_1':0, 'employment_type_2':0}
        key = f'employment_type_{df.employment_type.item()}'
        employment_type_hot[key] = 1
        processed_data.drop(columns=['employment_type', 'total_account_loan_no', 'total_overdue_no', 'main_account_active_loan_no', 'sub_account_active_loan_no'], inplace=True)
        final = pd.concat([processed_data, pd.DataFrame([employment_type_hot])], axis=1)
        print(final.columns)
        return final
