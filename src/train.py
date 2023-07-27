import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from pandas import DataFrame, Series

#load data
def load_data_from_url(url:str) -> DataFrame:
    df = pd.read_csv(url)
    return df

def initial_features_selection(df:DataFrame) -> DataFrame:
    df.sort_values(by="Date")
    df = df[["Close"]]
    return df

def create_time_series_features(df:DataFrame,n_inputs:int=7) -> DataFrame:
    lista = []
    for index in range(len(df.index) - n_inputs) :
        obj = dict()
        for j in range(n_inputs):
            obj[f"day_{j+1}"] = df.iloc[j+index]["Close"]
        obj["target_day"] = df.iloc[n_inputs+index]["Close"]
        lista.append(obj)
    return pd.DataFrame(lista)

def data_formating(df:DataFrame) -> DataFrame:
    df = df.round()
    df = df.astype("int")
    return df

def data_transformation(df:DataFrame) -> DataFrame:
    df = df.apply(np.log)
    return df


def split_data(df:DataFrame,test_size:float = 0.2,random_state:int=42) -> tuple[DataFrame,DataFrame,Series,Series]:
    x = df.drop(["target_day"],axis=1)
    y = df["target_day"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test 

def train_model(x_train:DataFrame, x_test:DataFrame, y_train:Series, y_test:Series) -> list:
    models = {
        "Random Forrest" : RandomForestRegressor(),
        "Linear Regression" : LinearRegression()
    }
    scores = []
    for model,reg in models.items():
        reg.fit(x_train,y_train)
        y_pred = reg.predict(x_test)
        obj = dict()
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
    df = load_data_from_url("https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1519603200&period2=1677369600&interval=1d&events=history&includeAdjustedClose=true")

    # Feature Selection
    df = initial_features_selection(df)

    # Create Time Series Features
    n_inputs = 7  # Number of input days to use for prediction
    df = create_time_series_features(df, n_inputs)

    # Data Formatting
    df = data_formating(df)

    # Split Data into Train and Test Sets
    x_train, x_test, y_train, y_test = split_data(df)

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