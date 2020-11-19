import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression

TRAIN_DATA = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area):
    df = pd.read_csv('TRAIN_DATA')
    df = df.T
    
    b = df.index
    a = list(b[1:]) #area
    
    a = list(df[0])
    p = a[1:] #price
    
    x_train = np.array(a).reshape(-1, 1)
    y_train = np.array(p).reshape(-1, 1)
    reg = LinearRegression.fit(x_train, y_train)
    res = reg.predict(a)
    return res
    


if _name_ == "_main_":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys())).reshape(-1, 1)
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
