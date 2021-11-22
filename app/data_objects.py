from pydantic import BaseModel, validator, ValidationError
from enum import Enum

from typing_extensions import Literal


class DesiredModel(Enum):
    mlp = 'MLP'
    knn = 'KNN'
    svm = 'SVM'
    logreg = 'LogReg'
    forest = 'Forest'


class RequestInputData(BaseModel):
    Driving_flag: int
    last_six_months_new_loan_no: int
    last_six_month_defaulted_no: int
    average_age: int
    credit_history: int
    loan_to_asset_ratio: float
    total_outstanding_loan: int
    total_disbursed_loan: int
    total_monthly_payment: float
    active_to_inactive_act_ratio: float
    Credit_level: int
    age: int
    employment_type: int
    total_overdue_no: int
    main_account_active_loan_no: int
    total_account_loan_no: int
    total_monthly_payment: float
    total_outstanding_loan: float
    total_disbursed_loan: float
    sub_account_active_loan_no: int





