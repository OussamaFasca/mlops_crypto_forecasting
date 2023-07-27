"""
This module goes through the complete pipeline
to train the forecasting model
"""
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from pandas import DataFrame, Series

#load data
def load_data_from_url(url:str) -> DataFrame:
    """
    Loads data from yahoo finance url
    """
    dataset = pd.read_csv(url)
    return dataset

def initial_features_selection(dataset:DataFrame) -> DataFrame:
    """
    Choosing the initial columns to use for feature engineering
    """
    dataset.sort_values(by="Date")
    dataset = dataset[["Close"]]
    return dataset

def create_time_series_features(dataset:DataFrame,n_inputs:int=7) -> DataFrame:
    """
    Applying feature engineering to create the necessary features
    for the time series.
    """
    lista = []
    for index in range(len(dataset.index) - n_inputs) :
        obj = {}
        for j in range(n_inputs):
            obj[f"day_{j+1}"] = dataset.iloc[j+index]["Close"]
        obj["target_day"] = dataset.iloc[n_inputs+index]["Close"]
        lista.append(obj)
    return pd.DataFrame(lista)

def data_formating(dataset:DataFrame) -> DataFrame:
    """
    Formatting data
    """
    dataset = dataset.round()
    dataset = dataset.astype("int")
    return dataset

def data_transformation(dataset:DataFrame) -> DataFrame:
    """
    Transforming target variable
    """
    dataset = dataset.apply(np.log)
    return dataset


def split_data(dataset:DataFrame,test_size:float = 0.2,random_state:int=42) -> tuple[DataFrame,DataFrame,Series,Series]:
    """
    Splitting Data into train and test set
    """
    x = dataset.drop(["target_day"],axis=1)
    y = dataset["target_day"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

def train_model(x_train:DataFrame, x_test:DataFrame, y_train:Series, y_test:Series) -> list:
    """
    Model training
    """
    models = {
        "Random Forrest" : RandomForestRegressor(),
        "Linear Regression" : LinearRegression()
    }
    scores = []
    for model,reg in models.items():
        reg.fit(x_train,y_train)
        y_pred = reg.predict(x_test)
        obj = {}
        obj["Model"] = model
        obj["R²"] = r2_score(y_test,y_pred)
        #y_test = np.exp(y_test)
        #y_pred = np.exp(y_pred)
        obj["RMSE"] = sqrt(mean_squared_error(y_test,y_pred))
        obj["MAE"] = mean_absolute_error(y_test,y_pred)
        #obj["params"] = reg.get_params()
        scores.append(obj)
    return scores



if __name__ == "__main__":
    dataset = load_data_from_url("https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1519603200&period2=1677369600&interval=1d&events=history&includeAdjustedClose=true")

    # Feature Selection
    dataset = initial_features_selection(dataset)

    # Create Time Series Features
    dataset = create_time_series_features(dataset)

    # Data Formatting
    dataset = data_formating(dataset)

    # Split Data into Train and Test Sets
    x_train, x_test, y_train, y_test = split_data(dataset)

    # Train Models and Evaluate Performance
    scores = train_model(x_train, x_test, y_train, y_test)

    # Display Results
    print("Model Evaluation:")
    print("-------------")
    for score in scores:
        print(f"Model: {score['Model']}")
        print(f"R²: {score['R²']}")
        print(f"RMSE: {score['RMSE']}")
        print(f"MAE: {score['MAE']}")
        print("-------------")
