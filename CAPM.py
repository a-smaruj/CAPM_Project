# imports
import numpy as np
import pandas as pd
import holidays
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, date
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from chow_test import chow_test
from fpdf import FPDF
pd.plotting.register_matplotlib_converters()
yf.pdr_override()

# date range: 2002-2015


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


# Creat model
def create_capm(asset, stats_asset, name):
    model = LinearRegression(fit_intercept=True)
    x = asset['Market Risk Premium'].array.reshape(-1, 1)
    y = asset['y']
    results = model.fit(x, y)
    y_pred = model.predict(x)

    # Collect stats
    stats_asset.loc[name] = [results.coef_, results.intercept_, results.score(x, y) * 100]

    return stats_asset, y_pred


def conduct_chowtest(asset, name, n):
    print(name)
    res_chow = chow_test(asset['y'], asset['Market Risk Premium'], n - 1, n, 0.01)
    p = round(res_chow[1], 4)

    # Save to pdf
    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10, txt="Results of chow test:", ln=1, align='L')
    if p <= 0.01:
        pdf.cell(w=0, h=10, txt="Reject the null hypothesis of equality of regression coefficients in the two periods.",
                 ln=1, align='L')
        pdf.cell(w=0, h=10, txt="Asset \"" + name + "\" is not cyclical.", ln=1, align='L')
    else:
        pdf.cell(w=0, h=10, txt="Accept the null hypothesis of equality of regression coefficients in the two periods.",
                 ln=1, align='L')
        pdf.cell(w=0, h=10, txt="Asset \"" + name + "\" is cyclical.", ln=1, align='L')


# Display results of capm
def result_analyse(asset, name):
    asset = pd.merge(main_dataFrame, asset, how='inner', on='Date')

    # Create y by subtracting 'Risk free investment' to gain linear function
    asset['y'] = asset.iloc[:, 3] - asset['Risk free investment']

    # Split into two sets, before and after 2009
    asset_2009 = asset[asset['Date'] < datetime.strptime('2009-01-01', '%Y-%m-%d').date()]
    asset_2015 = asset[asset['Date'] >= datetime.strptime('2009-01-01', '%Y-%m-%d').date()]

    # Create stats table
    stats_asset = pd.DataFrame(columns=['Beta', 'Intercept', 'R^2'])
    stats_asset, y_pred_2009 = create_capm(asset_2009, stats_asset, name + ' 2009')
    stats_asset, y_pred_2015 = create_capm(asset_2015, stats_asset, name + ' 2015')
    print(stats_asset.to_markdown())

    global stats_dataFrame
    stats_dataFrame = pd.concat([stats_dataFrame, stats_asset])

    # Function formula
    function_2009 = '2009: y = ' + str(round(stats_asset.iloc[0, 0][0], 2)) + 'x ' + str(
        round(stats_asset.iloc[0, 1], 2))
    function_2015 = '2015: y = ' + str(round(stats_asset.iloc[1, 0][0], 2)) + 'x ' + str(
        round(stats_asset.iloc[1, 1], 2))
    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10, txt=function_2009, ln=1, align='L')
    pdf.cell(w=0, h=10, txt=function_2015, ln=1, align='L')

    # Visualise dependence on a plot
    fig_1 = plt.figure()
    plt.plot(asset_2009['Market Risk Premium'], y_pred_2009, color='black')
    plt.plot(asset_2015['Market Risk Premium'], y_pred_2015, color="darkviolet")
    plt.legend([function_2009, function_2015])
    plt.title(name)
    plt.show()
    fig_1.savefig('./results/chart_mrp_' + name + '.png',
                  transparent=False,
                  facecolor='white',
                  bbox_inches="tight")
    pdf.image('./results/chart_mrp_' + name + '.png',
              x=50, y=None, w=100, h=0, type='PNG')

    fig_2 = plt.figure()
    plt.plot(asset['Date'], np.append(y_pred_2009, y_pred_2015), color='darkviolet')
    plt.scatter(asset['Date'], asset['y'], color='black', linewidths=0.1)
    plt.legend(['estimated', 'observed'])
    plt.title(name)
    plt.show()
    fig_2.savefig('./results/chart_date_' + name + '.png',
                  transparent=False,
                  facecolor='white',
                  bbox_inches="tight")
    pdf.image('./results/chart_date_' + name + '.png',
              x=50, y=None, w=100, h=0, type='PNG')

    fig_3, axs = plt.subplots(2)
    fig_3.suptitle(name)
    axs[0].plot(asset_2009['Market Risk Premium'], y_pred_2009, color='darkviolet', label='estimated')
    axs[0].scatter(asset_2009['Market Risk Premium'], asset_2009['y'], color='black', linewidths=0.1, label='observed')
    axs[0].set_title('2009')
    axs[1].plot(asset_2015['Market Risk Premium'], y_pred_2015, color='darkviolet', label='estimated')
    axs[1].scatter(asset_2015['Market Risk Premium'], asset_2015['y'], color='black', linewidths=0.1, label='observed')
    axs[1].set_title('2015')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig_3.legend(handles, labels, loc='upper right')
    plt.show()
    fig_3.savefig('./results/charts_two_' + name + '.png',
                  transparent=False,
                  facecolor='white',
                  bbox_inches="tight")
    pdf.image('./results/charts_two_' + name + '.png',
              x=50, y=None, w=100, h=0, type='PNG')

    # Chow test
    conduct_chowtest(asset, name, len(asset_2009['y']))


def analyse_asset(abr, name):
    result_analyse(prepare_data_frame(abr), name)


# List of holidays occurring between 2002 and 2016
us_holidays = pd.DataFrame(holidays.US(years=range(2002, 2016)).items(),
                           columns=['date', 'holiday'])
holidays_list = us_holidays['date'].to_list()

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

# Table with stats
stats_dataFrame = pd.DataFrame(columns=['Beta', 'Intercept', 'R^2'])

# List of chosen assets to analyse
# price of asset of 5 american companies - research subject
assets_dict = {'MSFT': 'Microsoft Corporation',
               'PG': 'The Procter & Gamble Company',
               'PFE': 'Pfizer Inc.',
               'WMT': 'Walmart Inc',
               'HOG': 'Harley-Davidson, Inc.'}

# Create pdf
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 28)
pdf.cell(w=0, h=10, txt="Report of CAPM analysis", ln=1, align='C')
pdf.set_font('Arial', 'I', 14)
pdf.cell(w=0, h=10, txt="author: Alicja Smaruj", ln=1, align='C')
pdf.set_font('Arial', 'I', 14)
today = date.today()
pdf.cell(w=0, h=10, txt="date: " + today.strftime("%d %B %Y"), ln=1, align='C')
pdf.ln(8)
pdf.set_font('Arial', '', 12)
pdf.cell(w=0, h=10, txt="Chosen assets: ", ln=1, align='L')

for key in assets_dict.keys():
    pdf.cell(w=0, h=10, txt='- ' + assets_dict[key], ln=1, align='L')

for key in assets_dict.keys():
    # Save to pdf
    pdf.cell(w=0, h=10, txt='', ln=1, align='L')
    pdf.set_font('Arial', 'U', 14)
    pdf.cell(w=0, h=10, txt=assets_dict[key], ln=1, align='L')

    # Analyse asset
    analyse_asset(key, assets_dict[key])

pdf.cell(w=0, h=10, txt='', ln=1, align='L')
pdf.set_font('Arial', 'U', 14)
pdf.cell(w=0, h=10, txt='Summary', ln=1, align='L')
# TODO table to pdf
print(stats_dataFrame.to_markdown())

pdf.output(f'./results/report.pdf', 'F')

