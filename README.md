# CAPM_Project

## General

The main idea is to create Capital Asset Pricing Model based on data from [https://finance.yahoo.com/](https://finance.yahoo.com/) and compare the values of beta of investment before and after 2008-2009 Global Financial Crisis. All results are collected in a generated pdf report in results folder.

## Used data

13 Week Treasury Bill was taken as free-risk investment and S&P500 as value of the market. In order to make a comparison, five exemplary assets were chosen, including:
- Microsoft Corporation (MSFT)  - IT business
- The Procter & Gamble Company (PG) - cosmetic business
- Pfizer Inc. (PFE) - pharmaceutical business
- Walmart Inc. (WMT) - supermarket business
- Harley-Davidson, Inc. (HOG) - automotive business

To test other assets, all you have to do is to change the <code>assets_dict</code>.

## Technology

The analysis was made using Python and packages like:
- yfinance
- LinearRegression
- pandas
- matplotlib
