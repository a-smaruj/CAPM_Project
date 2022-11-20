# imports
import pandas as pd
import holidays
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
pd.plotting.register_matplotlib_converters()

# import seaborn as sns
# import numpy as np

# date range: 2002-2015

yf.pdr_override()

# List of chosen assets to analyse
# price of asset of 5 american companies - research subject
# Microsoft Corporation (MSFT) - IT business
# The Procter & Gamble Company (PG) - cosmetic business
# Pfizer Inc. (PFE) - pharmaceutical business
# Walmart Inc. (WMT) - supermarket business
# Harley-Davidson, Inc. (HOG) - automotive business
assets_dict = {'MSFT': 'Microsoft Corporation',
               'PG': 'The Procter & Gamble Company',
               'PFE': 'Pfizer Inc.',
               'WMT': 'Walmart Inc',
               'HOG': 'Harley-Davidson, Inc.'}

# List of holidays occurring between 2002 and 2016
us_holidays = pd.DataFrame(holidays.US(years=range(2002, 2016)).items(),
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
    if name != '^IRX':
        asset['Value'] = ((1 + asset['Adj Close'])**(1/250) - 1) * asset['Volume']
    else:
        asset['Value'] = asset['Adj Close']
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

# Create main dataFrame with market risk premium
main_dataFrame = pd.merge(sp500, treasury, how='inner', on='Date')
main_dataFrame['Market Risk Premium'] = main_dataFrame['Daily returns ^GSPC'] - main_dataFrame['Daily returns ^IRX']
main_dataFrame = main_dataFrame[['Date', 'Daily returns ^IRX', 'Market Risk Premium']]
main_dataFrame.rename(columns={'Daily returns ^IRX': 'Risk free investment'}, inplace=True)
main_dataFrame = main_dataFrame[main_dataFrame['Risk free investment'] > -10]


# Creat model
def create_capm(asset, stats_asset, name):
    model = LinearRegression(fit_intercept=True)
    x = asset['Market Risk Premium'].array.reshape(-1, 1)
    y = asset.iloc[:, 3] - asset['Risk free investment']
    results = model.fit(x, y)
    y_pred = model.predict(x)

    # Collect stats
    stats_asset.loc[name] = [results.coef_, results.intercept_, results.score(x, y) * 100]

    return stats_asset, y_pred


# Table with stats
stats_dataFrame = pd.DataFrame(columns=['Beta', 'Intercept', 'R^2'])


# Display results of capm
def result_capm(asset, name):
    asset = pd.merge(main_dataFrame, asset, how='inner', on='Date')

    # Split into two sets, before and after 2009
    asset_2009 = asset[asset['Date'] < datetime.strptime('2009-01-01', '%Y-%m-%d').date()]
    asset_2015 = asset[asset['Date'] >= datetime.strptime('2009-01-01', '%Y-%m-%d').date()]

    stats_asset = pd.DataFrame(columns=['Beta', 'Intercept', 'R^2'])
    stats_asset, y_pred_2009 = create_capm(asset_2009, stats_asset, name + ' 2009')
    stats_asset, y_pred_2015 = create_capm(asset_2015, stats_asset, name + ' 2015')

    global stats_dataFrame
    stats_dataFrame = pd.concat([stats_dataFrame, stats_asset])

    # Visualise dependence on a plot
    plt.plot(asset_2009['Market Risk Premium'], y_pred_2009, color='black')
    plt.plot(asset_2015['Market Risk Premium'], y_pred_2015, color="blue")
    plt.legend(['2009', '2015'])
    plt.title(name)
    plt.show()

    # plt.plot(asset_2009['Market Risk Premium'], y_pred_2009 + asset_2009['Risk free investment'], color='black')
    # plt.plot(asset_2015['Market Risk Premium'], y_pred_2015 + asset_2015['Risk free investment'], color="blue")
    # plt.legend(['2009', '2015'])
    # plt.title(name)
    # plt.show()


def analyse_asset(abr, name):
    result_capm(prepare_data_frame(abr), name)


for key in assets_dict.keys():
    analyse_asset(key, assets_dict[key])

print(stats_dataFrame.to_markdown())

