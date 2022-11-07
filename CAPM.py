# imports
import pandas as pd
import holidays
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
pd.plotting.register_matplotlib_converters()

# import sklearn.metrics
# from scipy.stats import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pulp import *
# import numpy as np
# from sklearn.metrics import mean_squared_error
# import statsmodels.api as sm

# date range: 2002-2015

yf.pdr_override()

us_holidays = pd.DataFrame(holidays.US(years=range(2002,2016)).items(),
                           columns=['date', 'holiday'])
holidays_list = us_holidays['date'].to_list()


def prepare_data_frame(name):
    # Download needed data
    asset = pdr.get_data_yahoo(name, start='2002-01-01', end='2016-01-01')

    # Data cleaning
    # Delete holidays
    asset.index = list(map(datetime.date, asset.index))
    for dat in asset.index:
        if dat in holidays_list: asset = asset.drop(dat)

    # Remove row names
    asset.index.name = 'Date'
    asset.reset_index(inplace=True)

    # Count daily returns
    if name != '^IRX': asset['Value'] = ((1 + asset['Adj Close'])**(1/250) - 1) * asset['Volume']
    else: asset['Value'] = asset['Adj Close']
    temp = asset['Value'].iloc[:-1].to_list()
    asset = asset.iloc[1:]
    name = 'Daily returns' + ' ' + name
    asset[name] = (asset['Value'] - temp)/asset['Value']
    asset = asset[['Date', name]]
    return asset


# S&P500 - value of the market
sp500 = prepare_data_frame('^GSPC')

# 13 Week Treasury Bill (^IRX) - free-risk investment
treasury = prepare_data_frame('^IRX')
treasury_a = pdr.get_data_yahoo('^IRX', start='2002-01-01', end='2016-01-01')
print(treasury_a.to_markdown())

# Create main dataFrame with market risk premium
main_dataFrame = pd.merge(sp500, treasury, how='inner', on='Date')
main_dataFrame['Market Risk Premium'] = main_dataFrame['Daily returns ^GSPC'] - main_dataFrame['Daily returns ^IRX']
main_dataFrame = main_dataFrame[['Date', 'Daily returns ^IRX', 'Market Risk Premium']]
main_dataFrame.rename(columns={'Daily returns ^IRX':'Risk free investment'}, inplace=True)
main_dataFrame = main_dataFrame[main_dataFrame['Risk free investment'] > -10]


def create_capm(asset):
    asset = pd.merge(main_dataFrame, asset, how='inner', on='Date')
    return asset


# price of asset of 5 american companies - research subject
# Microsoft Corporation (MSFT) - IT business
msft = prepare_data_frame('MSFT')
msft = create_capm(msft)

# The Procter & Gamble Company (PG) - cosmetic business
pg = prepare_data_frame('PG')

# Pfizer Inc. (PFE) - pharmaceutical business
pfe = prepare_data_frame('PFE')

# Walmart Inc. (WMT) - supermarket business
wmt = prepare_data_frame('WMT')

# Harley-Davidson, Inc. (HOG) - automotive business
hog = prepare_data_frame('HOG')

# print(msft.to_markdown())

# msft_2009 = msft[msft['Date'] < datetime.strptime('2009-01-01', '%Y-%m-%d').date()]

# Plots
# sns.lineplot(data=msft['Risk free investment'])
# plt.show()
# plt.scatter(msft['Date'], msft['Daily returns MSFT'], color='g')
# plt.show()

# Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
# plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# msft['Close'].plot(label = 'MSFT', figsize=(10,8))
# sp500['Close'].plot(label = 'SPY')
# plt.legend()

# Conclusion and questions
# 2014-09-22 daily return:-4
# 2009-11-19 risk free:-5


# Creating model

# 1st attitude
# from scipy.optimize import minimize
#
# def model(params, X):
#     alpha = params[0]
#     y_pred = msft['Risk free investment'] + X*alpha
#     return y_pred
#
# def sum_of_squares(params, X, Y):
#     y_pred = model(params, X)
#     obj = np.sqrt(((y_pred - Y) ** 2).sum()/len(Y))
#     return obj
#
# alpha_0 = 0.1
#
# res = minimize(sum_of_squares, [alpha_0, ], args=(msft['Market Risk Premium'], msft['Daily returns MSFT']),
#                tol=1e-3, method="Powell")
# print(res)
#
# y_pred = model(res['x'], msft['Market Risk Premium'])
#
# print(sklearn.metrics.r2_score(msft['Daily returns MSFT'], y_pred))
#
# plt.scatter(msft['Market Risk Premium'], msft['Daily returns MSFT'], color="black")
# plt.plot(msft['Market Risk Premium'], y_pred, color="blue", linewidth=3)
# plt.show()

# 2nd attitude
# x = msft[['Risk free investment', 'Market Risk Premium']]
# y = msft['Daily returns MSFT']
# x = sm.add_constant(x)
# model = sm.OLS(y, x).fit()
# predictions = model.predict(x)
# print_model = model.summary()
# print(print_model)
# plt.scatter(msft['Market Risk Premium'], msft['Daily returns MSFT'], color="black")
# plt.plot(msft['Market Risk Premium'], predictions, color="blue", linewidth=3)
# plt.show()

# 3rd attitude
# x = msft['Market Risk Premium']
# y = msft['Daily returns MSFT']
# x = sm.add_constant(x)
# model = sm.OLS(y, x).fit()
# predictions = model.predict(x)
# print_model = model.summary()
# print(print_model)
# plt.scatter(msft['Date'], msft['Daily returns MSFT'], color="black")
# plt.plot(msft['Date'], predictions, color="blue", linewidth=3)
# plt.show()

# 4th attitude
# from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=True)
# x = msft['Market Risk Premium'].array.reshape(-1, 1)
# y = msft['Daily returns MSFT'] - msft['Risk free investment']
# reg = model.fit(x, y)
# print(reg.intercept_, reg.coef_, reg.score(x, y))
# y_pred = model.predict(x)
# plt.scatter(x, y, color='black')
# plt.plot(x, y_pred, color="blue", linewidth=3)
# plt.show()
