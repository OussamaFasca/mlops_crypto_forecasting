import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

#load data
df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1519603200&period2=1677369600&interval=1d&events=history&includeAdjustedClose=true')

# data selection - initial feature selection
df.sort_values(by="Date")
df = df[["Close"]]

# data preparation
n_inputs = 9
lista = []
for index in range(len(df.index) - n_inputs) :
    obj = dict()
    for j in range(n_inputs):
        obj[f"day_{j+1}"] = df.iloc[j+index]["Close"]
    obj["day_8_target"] = df.iloc[n_inputs+index]["Close"]
    lista.append(obj)

new_df = pd.DataFrame(lista)

#data formatting
new_df = new_df.round()
new_df = new_df.astype("int")

#data transformation
new_df = new_df.apply(np.log)

#splitting
x = new_df.drop(["day_8_target"],axis=1)
y = new_df["day_8_target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Random Forrest" : RandomForestRegressor()
}

scores = []

for model,reg in models.items():
    reg.fit(x_train,y_train)
    y_pred = reg.predict(x_test)
    obj = dict()
    obj["Model"] = model
    obj["R²"] = r2_score(y_test,y_pred)
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    obj["RMSE"] = sqrt(mean_squared_error(y_test,y_pred))
    obj["MAE"] = mean_absolute_error(y_test,y_pred)
    #obj["params"] = reg.get_params()
    scores.append(obj)

results = pd.DataFrame(scores)
#results.to_csv('./classical/results.csv',index=False)
print(scores)