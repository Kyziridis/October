import secrets
from http.client import HTTPException

import pandas as pd
from fastapi import Depends, FastAPI, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import ValidationError

from data_manager import DataManager
from data_objects import RequestInputData
from model import Model

app = FastAPI(title="October",
              description="Predict car loan eligibility",
              docs_url='/')

security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "October")
    correct_password = secrets.compare_digest(credentials.password, "123")
    if not (correct_username and correct_password):
        raise HTTPException()
    return credentials.username


data_manager = DataManager(path_to_data='../data_folder/car_loan_trainset.csv')
data = data_manager.get_preprocessed_data()
x_train, x_test, y_train, y_test = data_manager.split_train_test(data)


@app.post("/response_variable_name")
async def response_column_name(username: str = Depends(get_current_username)):
    return {"Response variable": 'loan_default', 'username': username}


@app.get("/descriptive")
async def descriptive_statistics():
    return HTMLResponse(data_manager.explore(data))


@app.post("/train/{model_to_fit}/oversampling/{oversampling}")
async def fit(model_to_fit: str, oversampling: bool):
    model_for_fit = Model(model_to_fit)  # Init model
    fitted_model = model_for_fit.train(x_train, y_train,
                                       oversampling=oversampling)  # inside train function loads scaler and transforms
    saved = model_for_fit.save_model(fitted_model, model_name=model_to_fit)
    evaluation = model_for_fit.evaluate(x_test, y_test)
    print(evaluation)
    return evaluation


@app.post("/predict/{desired_model}/")
async def predict(desired_model: str, feature_vector: dict, username: str = Depends(get_current_username)):
    try:
        feature_request = RequestInputData(**feature_vector)
    except ValidationError as e:
        raise e
    data_for_prediction = data_manager.handle_request_data(feature_request)
    model = Model(desired_model)
    prediction = model.predict(data_for_prediction)
    pred = prediction.item()
    return {'prediction_outcome_class': pred, 'username': username}
