from portfoliobuilder import StockPrediction, SentimentAnalysis
from portfoliobuilder import PortfolioOptimization

data = dict()


def processStocks(ticker_list):
    data.clear()
    print("Stocks received for processing: {}".format(ticker_list))
    finalized_list = []
    data["tickerList"] = ticker_list
    for ticker in ticker_list:
        pricePredictionResponse = StockPrediction.getClosePricePrediction(ticker, data)
        sentimentPredictionResponse = SentimentAnalysis.getSentimentAnalysis(ticker, data)
        if pricePredictionResponse or sentimentPredictionResponse:
            finalized_list.append(ticker)

    if (len(finalized_list) != 0):
        data["finalizedList"] = finalized_list
        PortfolioOptimization.getOptimalPortfolio(finalized_list, data)
    else:
        data['optimalPortfolio'] = "No bullish stocks identified for portfolio building"
        data['optimalPortfolioReturn'] = "Not Applicable"
        data['optimalPortfolioVolatility'] = "Not Applicable"
    return data


def getData():
    return data
