from http.client import HTTPException

import requests

import pytest
import base64
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_auth_main():
    token = base64.b64encode(b'October:123').decode('utf-8')
    response = client.post('/response_variable_name', headers={"Authorization": f"Basic {token}"})
    assert response.status_code == 200
    assert response.ok == True


def test_bad_auth():
    token = base64.b64encode(b'October:1234').decode('utf-8')
    try:
        response = client.post('/response_value_variable_name', headers={"Authorization": f"Basic {token}"})
    except HTTPException:
        assert True


def test_request_predict_endpoint():
    token = base64.b64encode(b'October:123').decode('utf-8')
    body = {
        "Driving_flag": 1,
        "last_six_months_new_loan_no": 0,
        "last_six_month_defaulted_no": 12,
        "average_age": 40,
        "credit_history": 0,
        "loan_to_asset_ratio": 0.65,
        "total_outstanding_loan": 0,
        "total_disbursed_loan": 0,
        "total_monthly_payment": 123,
        "active_to_inactive_act_ratio": 0.43,
        "Credit_level": 2,
        "age": 28,
        "loan_default": 1,
        "employment_type": 2,
        "total_overdue_no": 1000,
        "main_account_active_loan_no": 0,
        "total_account_loan_no": 0,
        "sub_account_active_loan_no": 0
    }
    response = requests.post('http://127.0.0.1:8000/predict/MLP', headers={"Authorization": f"Basic {token}"},
                             json=body)
    assert response.ok == True
    assert response.status_code == 200


def test_bad_request_predict_endpoint():
    token = base64.b64encode(b'October:123').decode('utf-8')
    body = {
        "Driving_flag": "lala",
        "last_six_months_new_loan_no": 0,
        "last_six_month_defaulted_no": 12,
        "average_age": 40,
        "credit_history": 0,
        "loan_to_asset_ratio": 0.65,
        "total_outstanding_loan": 0,
        "total_disbursed_loan": 0,
        "total_monthly_payment": 123,
        "active_to_inactive_act_ratio": 0.43,
        "Credit_level": 2,
        "age": 28,
        "loan_default": 1,
        "employment_type": 2,
        "total_overdue_no": 1000,
        "main_account_active_loan_no": 0,
        "total_account_loan_no": 0,
        "sub_account_active_loan_no": 0
    }
    response = requests.post('http://127.0.0.1:8000/predict/MLP', headers={"Authorization": f"Basic {token}"},
                             json=body)
    assert response.ok == False
    assert response.status_code != 200
