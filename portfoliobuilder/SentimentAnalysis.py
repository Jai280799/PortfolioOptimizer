import nltk
from urllib.request import urlopen, Request
import os
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px


def getSentimentAnalysis(ticker, data):
    finwizUrl = "https://finviz.com/quote.ashx?t="
    requestHeader_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}
    news_dict = {}
    # get news headlines table from finvix.com by appending ticker
    response = urlopen(Request(url=finwizUrl + ticker, headers=requestHeader_dict))
    htmlTable = BeautifulSoup(response).find(id="news-table")
    news_dict[ticker] = htmlTable

    # format the news headlines and prepare the dataframe accordingly
    parsed_news = []
    news_date, news_time = "", ""
    for ticker_key, ticker_table in news_dict.items():
        for row in ticker_table.findAll("tr"):
            news_headline = row.a.get_text()
            date_info = row.td.get_text().split()
            if len(date_info) == 2:
                news_date = date_info[0]
                news_time = date_info[1]
            else:
                news_time = date_info[0]
            parsed_news.append([ticker_key, news_date, news_time, news_headline])

    # Initialize the vader sentiment analyser
    vader_analyser = SentimentIntensityAnalyzer()
    df_news = pd.DataFrame(parsed_news, columns=['Ticker', 'Date', 'Time', 'News Headline'])
    df_news['Date'] = pd.to_datetime(df_news.Date).dt.date
    # generate sentiment score
    sentiment_scores = pd.DataFrame(df_news['News Headline'].apply(vader_analyser.polarity_scores).tolist())
    df_news = df_news.join(sentiment_scores, rsuffix='_right')
    # consolidate results to get final sentiment score for a day
    average_daily_scores = df_news.groupby(['Ticker', 'Date']).mean()
    average_daily_compound_scores = average_daily_scores.unstack().xs('compound', axis="columns").transpose()

    # plot the bar graph displaying the average daily compound score
    fig = px.bar(average_daily_compound_scores)
    fig.update_layout(barmode='group')
    data["{}_sent".format(ticker)] = fig.to_html(full_html=False)

    # Check for positive sentiment to indicate it is bullish
    if average_daily_compound_scores.iloc[-1].values > 0.1:
        print("Bullish Response = {} for {}".format(average_daily_compound_scores.iloc[-1], ticker))
        return True
    else:
        print("Bearish response for {}".format(ticker))
        return False
