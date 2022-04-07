import plotly.tools as tls
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas_ta as ta
import pandas as pd
import plotly.express as px
from datetime import date
from datetime import timedelta

def getWeightVector(n):
    weights = np.random.random(n)
    return weights / sum(weights)


def getReturnsData(ticker_list):
    returnSeriesList = []
    for s in ticker_list:
        returnSeriesList.append(pdr.DataReader(s, data_source='yahoo', start=str(date.today() - timedelta(days=730)),
                                               end=str(date.today() - timedelta(days=1))).ta.percent_return(length=1))
    returns_df = pd.concat(returnSeriesList, axis=1)
    returns_df.columns = ticker_list
    returns_df.dropna(inplace=True)
    return returns_df


def portfolioPerformance(mean_returns, cov_returns, weight_vector, num_trading_days):
    ret = np.sum(mean_returns * weight_vector) * num_trading_days
    weight_vec_trans = weight_vector.T
    std = np.sqrt(np.dot(weight_vec_trans, np.dot(cov_returns, weight_vector))) * np.sqrt(252)
    return std, ret


def generatePortfolios(mean_returns, cov_returns, num_assets, num_portfolios, risk_free_rate):
    portfolio_weights = []
    portfolio_results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weight_vector = getWeightVector(num_assets)
        portfolio_weights.append(weight_vector)
        portfolio_results[0, i], portfolio_results[1, i] = portfolioPerformance(mean_returns, cov_returns,
                                                                                weight_vector, 252)
        portfolio_results[2, i] = (portfolio_results[1, i] - risk_free_rate) / portfolio_results[0, i]
    return portfolio_results, portfolio_weights


def getOptimalPortfolio(ticker_list, data):
    returns_df = getReturnsData(ticker_list)
    risk_free_rate = 0.0151
    num_assets = len(ticker_list)
    num_portfolios = 10000

    mean_returns = returns_df.mean()
    cov_returns = returns_df.cov()
    portfolio_results, portfolio_weights = generatePortfolios(mean_returns, cov_returns, num_assets, num_portfolios,
                                                              risk_free_rate)
    best_sharpe_ratio_index = np.argmax(portfolio_results[2])
    best_portfolio_volatility, best_portfolio_annl_return = portfolio_results[0, best_sharpe_ratio_index], \
                                                            portfolio_results[1, best_sharpe_ratio_index]
    portfolios_graph = plt.figure()
    plt.plot(portfolio_results[0, :], portfolio_results[1, :], 'o', markersize=5)
    plt.plot(best_portfolio_volatility, best_portfolio_annl_return, 'r-o', markersize=10)
    plt.xlabel('Risk')
    plt.ylabel('Returns')
    portfolios_graph = tls.mpl_to_plotly(portfolios_graph)
    portfolios_graph.layout.height = 400
    portfolios_graph.layout.width = 800
    portfolios_graph.write_image('generatedPortfolios.png')

    best_portfolio_alloc = pd.DataFrame(portfolio_weights[best_sharpe_ratio_index], index=ticker_list,
                                        columns=['Allocation'])
    best_portfolio_alloc['Allocation'] = [round(i * 100, 2) for i in best_portfolio_alloc['Allocation']]
    best_portfolio_alloc = best_portfolio_alloc.T
    print("Optimal portfolio details")
    print("Annualised Returns {:.2f}".format(best_portfolio_annl_return))
    print("Annualised Volatility {:.2f}".format(best_portfolio_volatility))
    print("Securities distribution: ")
    print(best_portfolio_alloc)

    data["optimalPortfolioReturn"] = round(best_portfolio_annl_return * 100, 2)
    data["optimalPortfolioVolatility"] = round(best_portfolio_volatility * 100, 2)
    fig = px.pie(best_portfolio_alloc.T, values='Allocation', names=best_portfolio_alloc.T.index)
    data['optimalPortfolio'] = fig.to_html(full_html=False)
