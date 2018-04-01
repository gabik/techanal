""" Lib to read sp500 tickers from web """
import os
import pickle
import datetime
import requests
import bs4 as bs
import pandas as pd
from pandas_datareader import data as web
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

OUTPUT_DIR = 'stocks_dfs'
SPLIST_FILE = OUTPUT_DIR + '/sp500_comps'
MAIN_DF_FILE = OUTPUT_DIR + '/main_df'
WEEK_FORECAST_FILE = OUTPUT_DIR + '/week_forecast'
TRAINING_TABLE_FILE = OUTPUT_DIR + '/training_table'

def get_sp500_list(force=False):
    """ Get is from wiki """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if os.path.exists(SPLIST_FILE) and not force:
        with open(SPLIST_FILE, "r") as spf:
            tickers = pickle.load(spf)
    else:
        tickers = []
        req = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(req.text, 'lxml')
        tbl = soup.find('table', {'class': 'wikitable sortable'}).findAll('tr')[1:]
        for row in tbl:
            ticker = row.find('td').text
            tickers.append(ticker)

        with open(SPLIST_FILE, "wb") as spf:
            pickle.dump(tickers, spf)

    return tickers


def get_tickers(force=False, start=2000):
    """ Get the tickers from web """
    if not force and os.path.exists(MAIN_DF_FILE):
        main_df = pd.read_csv(MAIN_DF_FILE, index_col=[0,1])
        return main_df

    spl = [str(x) for x in get_sp500_list(reload_list)]
    main_df = pd.DataFrame()
    for count, tick in enumerate(spl):
        csv_file = "{}/df_{}".format(OUTPUT_DIR, tick)
        print count, csv_file
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        if os.path.exists(csv_file) and not force:
            print "Reading from disk"
            tkr = pd.read_csv(csv_file)
        else:
            print "Downloading"
            tkr = web.DataReader(tick, "morningstar", datetime.datetime(start, 1, 1), datetime.date.today())
            tkr.to_csv(csv_file)
        main_df = main_df.append(tkr)
    main_df.set_index(['Symbol', 'Date'], inplace=True)
    main_df = main_df.loc[:, ~main_df.columns.str.contains('^Unnamed')]
    main_df.to_csv(MAIN_DF_FILE)
    return main_df

def create_closes_table():
    main_df = get_tickers()
    main_df = main_df.loc[:, 'Close':'Close']
    tickers = pd.DataFrame()
    for stock in main_df.index.levels[0].tolist():
        print "Calculating {}".format(stock)
        tickers = tickers.join(main_df.loc[stock].rename(columns={'Close': stock}), how='outer')
    return tickers

def create_week_forecast(force=False, start=2000):
    if not force and os.path.exists(WEEK_FORECAST_FILE):
        main_df = pd.read_csv(WEEK_FORECAST_FILE, index_col=0)
        return main_df
    tickers = create_closes_table()
    for ticker in tickers.columns.values:
        for day in range(7):
            tickers['{}_d{}'.format(ticker, day + 1)] = (tickers[ticker].shift(-day-1) - tickers[ticker]) / tickers[ticker]
    tickers.fillna(0)
    tickers.to_csv(WEEK_FORECAST_FILE)
    return tickers

def signal_action_helper(x):
    for day in sorted(x.index.tolist()):
        if x[day] > 0.05:
            return 1
        if x[day] < -0.05:
            return -1
        return 0

def create_training_table(force_train=False, force_qoute=False, reload_list=False, start=2000):
    if not force_train and os.path.exists(TRAINING_TABLE_FILE):
        train = pd.read_csv(TRAINING_TABLE_FILE, index_col=0)
        return train
    train = pd.DataFrame({'Date': []})
    train.set_index('Date', inplace=True)
    spl = get_sp500_list(reload_list)
    tickers = create_week_forecast(force_qoute, start)
    for tick in spl:
        print tick
        tmp = tickers[[tick] + [tick + '_d' + str(x+1) for x in range(7)]]
        tmp.drop('0', inplace=True)
        #tmp['max'] = tmp.loc[:, tick + "_d1":tick + "_d7"].max(axis=1)
        #tmp['min'] = tmp.loc[:, tick + "_d1":tick + "_d7"].min(axis=1)
        tmp_actions = tmp.loc[:, tick + "_d1":tick + "_d7"].apply(signal_action_helper, axis=1)
        train[tick] = tmp_actions
    train.to_csv(TRAINING_TABLE_FILE)
    return train

def train_model():
    train = create_training_table()
    
