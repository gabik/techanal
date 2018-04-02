#!/usr/bin/env python
""" Machine learn buy or sell """
import bs4 as bs
from statistics import mean
import datetime as dt
#import matplotlib.pyplot as plt
#from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import warnings
warnings.simplefilter('ignore')

#style.use('ggplot')

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        ticker = str(ticker)
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            print "Downloading " + ticker
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()
    tickers_to_remove = []
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        if not df.empty:
            print ticker
            df.rename(columns={'Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Volume'], 1, inplace=True)
            main_df = main_df.join(df, how='outer')
        else:
            print "Removing " + ticker
            tickers_to_remove.append(ticker)

        if count % 10 == 0: print(count)
    for ttr in tickers_to_remove:
        tickers.remove(ttr)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


#def visualize_data():
#    df = pd.read_csv('sp500_joined_closes.csv')
#    df_corr = df.corr()
#    print(df_corr.head())
#    df_corr.to_csv('sp500corr.csv')
#    data1 = df_corr.values
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#
#    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
#    fig1.colorbar(heatmap1)
#
#    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
#    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
#    ax1.invert_yaxis()
#    ax1.xaxis.tick_top()
#    column_labels = df_corr.columns
#    row_labels = df_corr.index
#    ax1.set_xticklabels(column_labels)
#    ax1.set_yticklabels(row_labels)
#    plt.xticks(rotation=90)
#    heatmap1.set_clim(-1, 1)
#    plt.tight_layout()
#    plt.show()


def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.05
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    #print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[t for t in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.iloc[:-6].values
    today_X = df_vals.iloc[-6:].values
    y = df['{}_target'.format(ticker)].iloc[:-6].values
    return X, y, df, today_X


def do_ml(ticker, back_check=30, predict_days=1):
    X, y, df, today_X = extract_featuresets(ticker)

    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)
    df_vals = df[[t for t in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    last_week = predict_days
    test_size = 300 + last_week + back_check
    X_train = df_vals.iloc[:-test_size]
    y_train = df['{}_target'.format(ticker)].iloc[:-test_size]
    X_test = df_vals.iloc[-test_size:-last_week-back_check]
    y_test = df['{}_target'.format(ticker)].iloc[-test_size:-last_week-back_check]
    X_week = df_vals.iloc[-last_week-back_check:-back_check]

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    if not any(y_train):
        return -1, [0]

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_week)
    #print('predicted class counts:', Counter(predictions))
    return confidence, predictions

def run_all(back_check=30, predict_days=1):
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
    tickers = ['XEC']
    
    accuracies = []
    actions = {}
    for ticker in tickers:
        accuracy, prediction = do_ml(ticker, back_check, predict_days)
        if any(prediction):
            actions[ticker] = [accuracy] + prediction.tolist()
        if accuracy > -1:
            accuracies.append(accuracy)
        print '{} -- accuracy: {} , prediction: {} , mean accuracy: {}'.format(ticker, round(accuracy, 3), prediction, round(mean(accuracies), 3))
    return actions
    

def play_money_back_check(back_check=30):
    back_check += 1
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    cash = 1000000
    portion = 0.1
    predict_days = 1
    stop_loss = 0.95
    take_profit = 1.02
    portfolio = {}
    while back_check > 1:
        back_check -= 1
        today = df.index[-back_check-predict_days]
        print "Modelling {}".format(today)
        actions = run_all(back_check, predict_days)
        scores = {k:v for k,v in {k: v[0] * sum(v[1:]) for k,v in actions.items()}.items() if abs(v) > 0.5}
        if scores:
            for k,v in scores.items():
                price = df.loc[today][k]
                if v > 0 and k not in portfolio:
                    if cash < 0:
                        print "Want to Buy {} at {} for {} , but cash is {}".format(k, today, price, cash)
                    stocks = np.floor((cash * portion) / price)
                    total = price * stocks
                    if cash < total:
                        stocks = np.ceil(cash / price)
                        total = price * stocks
                    portfolio[k] = (price, stocks)
                    cash -= total
                    print "Buy {} - {} : {} x {} = {} ==> {}".format(k, today, price, stocks, total, cash)
                if v < 0 and k in portfolio:
                    buy_price = portfolio[k][0]
                    buy_stocks = portfolio[k][1]
                    profit = (price - buy_price) * buy_stocks
                    cash += price * buy_stocks
                    del(portfolio[k])
                    print "Sell on predict {} - {} : {} -> {} x {}  = {} ==> {}".format(k, today, buy_price, price, buy_stocks, profit, cash)
        stocks_to_del = []
        for k in portfolio:
            price = df.loc[today][k]
            buy_price = portfolio[k][0]
            buy_stocks = portfolio[k][1]
            profit = (price - buy_price) * buy_stocks
            if portfolio[k][0] > price * take_profit or portfolio[k][0] < price * stop_loss:
                cash += price * buy_stocks
                stocks_to_del.append(k)
                print "Sell {} - {} : {} -> {} x {}  = {} ==> {}".format(k, today, buy_price, price, buy_stocks, profit, cash)
            else:
                print "Hold {} - {} : {} -> {} x {}  = {} ==> {}".format(k, today, buy_price, price, buy_stocks, profit, cash)
            for k in stocks_to_del:
                del(portfolio[k])
    return portfolio

#get_data_from_yahoo(True)
#compile_data()
p = play_money_back_check(30)
