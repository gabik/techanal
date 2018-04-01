#!/usr/bin/env python
"""
Test
"""

import sys
import datetime
import pandas as pd
#import plotly.plotly as py
#import plotly.tools as ptools
#import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc
from pandas_datareader import data as web

#pylint: disable=invalid-name
#ptools.set_credentials_file(username='gabik', api_key='ZwcpVV98tVNpxDApBE1O')

sml = web.DataReader(sys.argv[1],
                     "morningstar",
                     datetime.datetime(2017, 1, 1),
                     datetime.date.today())
#SPY = web.DataReader("spy",
#                     "morningstar",
#                     datetime.datetime(2017, 1, 1),
#                     datetime.date.today())

#trace = go.Candlestick(x=jnug.index,
#                       open=jnug.open,
#                       high=jnug.high,
#                       low=jnug.low,
#                       close=jnug.close)
#data = [trace]
#py.iplot(data, filename='simple_candlestick')
#
#stocks = pd.DataFrame({"SPX": SPX["Close"].reset_index(drop=True),
#                       "SPY": SPY["Close"].reset_index(drop=True)})
#stocks_i = stocks.apply(lambda x: x/x[0])
#stocks_i.plot(grid=True)

fig, ax = plt.subplots()
candlestick2_ohlc(ax,
                  sml["Open"],
                  sml["High"],
                  sml["Low"],
                  sml["Close"],
                  width=1,
                  colorup='g',
                  colordown='r',
                  alpha=0.5)
base_ix = sml["Close"][0]
#SPX['profit'] = SPX["Close"].apply(lambda x: x/base_ix*100)
#SPX['profit'].plot(secondary_y=["profit"])
sml['m20'] = np.round(sml['Close'].rolling(window=20).mean(), 2)
sml['m20'].plot()
sml['m200'] = np.round(sml['Close'].rolling(window=200).mean(), 2)
sml['m200'].plot()
sml['m50'] = np.round(sml['Close'].rolling(window=50).mean(), 2)
sml['m50'].plot()
#plt.show()
