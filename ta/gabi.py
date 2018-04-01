import datetime
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as ticker
from pandas_datareader import data as web

style.use('seaborn')

SYMBOL = 'spx'
cash = 1000000
portion = 0.1
stop_loss = 0.2
days = 200

sml = web.DataReader(SYMBOL,
                     "morningstar",
                     datetime.datetime(2015, 1, 1),
                     datetime.date.today())
#sml = sml.reset_index()
ax1 = plt.subplot2grid((10,1), (0,0), rowspan=6, colspan=1)
ax1.xaxis_date()
ax2 = plt.subplot2grid((10,1), (6,0), rowspan=2, colspan=1, sharex=ax1)
ax2.xaxis_date()
ax3 = plt.subplot2grid((10,1), (8,0), rowspan=2, colspan=1, sharex=ax1)
ax2.set_ylim([-100, 100])
for m in [26, 12, 200, 20, 50]:
  sml['m' + str(m)] = np.round(sml['Close'].rolling(window=m).mean(), 2)
  sml['e' + str(m)] = np.round(sml['Close'].ewm(span=m, adjust=False).mean(), 2)
smlf = sml.loc[SYMBOL].iloc[-days:]
#smlf.index = np.arange(1, len(smlf) + 1)
#candlestick2_ohlc(ax1,
#                  smlf["Open"],
#                  smlf["High"],
#                  smlf["Low"],
#                  smlf["Close"],
#                  width=1,
#                  colorup='g',
#                  colordown='r',
#                  alpha=0.5)
cndl = pd.DataFrame({"Date": smlf.index.map(date2num),
                     "open": smlf["Open"],
                     "high": smlf["High"],
                     "low": smlf["Low"],
                     "close": smlf["Close"]}, index=smlf.index)
cndl = cndl[['Date', 'open', 'high', 'low', 'close']]
#cndl.index = smlf.index
candlestick_ohlc(ax1, cndl.values, width=1, colorup='g', colordown='r', alpha=0.5)
maxx = smlf["High"].max()
minx = smlf["Low"].min()
smlf["MACD_line"] = smlf["e12"] - smlf["e26"]
smlf["MACD_signal"] = smlf["MACD_line"].ewm(span=9, adjust=False).mean()
smlf["MACD_hist"] = smlf["MACD_line"] - smlf["MACD_signal"]
smlf['MACD_cross'] = np.where((smlf["MACD_signal"] - smlf['MACD_line']) < 0 , maxx, 0)
smlf['MACD_cross'][0] = 0
smlf["MACD_action"] = np.sign(smlf["MACD_cross"] - smlf["MACD_cross"].shift(1))

smlf['e20cross'] = np.where((smlf["e20"] - smlf['e50']) > 0 , maxx, 0)
smlf['e20cross'][0] = 0
smlf['m20cross'] = np.where((smlf["m20"] - smlf['m50']) > 0 , maxx, 0)
smlf['m20cross'][0] = 0

buy = pd.DataFrame(smlf.loc[smlf["MACD_action"] > 0, "Close"]).reset_index().rename(columns={"Close": "Buy"})
sell = pd.DataFrame(smlf.loc[smlf["MACD_action"] < 0, "Close"]).reset_index().rename(columns={"Close": "Sell"})
price = pd.DataFrame({"Buy":buy["Buy"],
                      "Sell":sell["Sell"],
                      "Days": sell["Date"] - buy["Date"],
                      "Start": buy["Date"],
                      "Stop": sell["Date"]})

price["profit"] = price["Sell"] - price["Buy"]
price["pct"] = ((price["Sell"] / price["Buy"]) - 1) * 100

smlf["Date"] = smlf.index
smlf.index = smlf.index.map(date2num)
smlf["MACD_line"].plot(ax=ax2)
smlf["MACD_signal"].plot(color='red', ax=ax2)
smlf["MACD_hist"].plot(alpha=0.3, ax=ax2)
smlf['MACD_cross'].plot(ax=ax1, ylim=(minx, maxx), color='green', alpha=0.3, kind='area')
smlf['e20cross'].plot(ax=ax1, ylim=(minx, maxx), color='yellow', alpha=0.1, kind='area')
smlf['m20cross'].plot(ax=ax1, ylim=(minx, maxx), color='blue', alpha=0.3, kind='area')

profit_data = pd.DataFrame({"Cash Before": [], "Cash After": []})
for index, row in price.iterrows():
    low = smlf["Low"].loc[date2num(row["Start"]):date2num(row["Stop"])].min()
    sell_price = row["Sell"] if (low > stop_loss * row["Buy"]) else np.floor(stop_loss * row["Buy"])
    buy_batch = np.round(np.floor(portion * cash) / row["Buy"])
    profit = buy_batch * (sell_price - row["Buy"])
    profit_data = profit_data.append(pd.DataFrame({"Cash Before": cash, "Cash After": cash + profit}, index=[index]))
    cash += profit
#profit_data.plot(ax=ax3)
plt.show()
