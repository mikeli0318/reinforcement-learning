"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
from pandas.tseries.offsets import BDay


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        x=self.getTrainIndicator(symbol,sd,ed)
        #x stores the indicators
        price=ut.get_data([symbol], pd.date_range(sd, ed))[symbol]
        #trade=price.copy();
        #trade.columns=["order"]
        #trade will contain the info of our trade
        state=x.ind1.astype(str).str.cat(x.ind2.astype(str)).str.cat(x.ind3.astype(str)).astype(int)
        #each day with the dtate
        self.learner = ql.QLearner(num_states=1000, num_actions = 3, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False)
        #position: 0:short, 1:no, 2:long
        #action: 0:become short 1:do nothing 2:become long
        daynum=price.shape[0]

        iter=0;
        while iter<10:
            cash=sv;
            position=1;
            act=self.learner.querysetstate(state.iloc[0])
            if act!=1:
                #do something
                cash=cash-500*price.iloc[0]*(act-position)
                position-act;
            valueprev=sv;#value yeaterday

            for i in range(1,daynum):
                value=cash+500*price.iloc[i]*(position-1);#value today
                act=self.learner.query(state.iloc[i],1000*(value/valueprev-1));
                if act!=1:
                    cash = cash - 500 * price.iloc[i] * (act - position)
                    position=act
                valueprev=value;
            iter+=1;

        iter=0
        prevrecord=np.zeros(daynum)
        record=np.zeros(daynum)
        while iter<90:
            cash=sv;
            position=1;
            act=self.learner.querysetstate(state.iloc[0])
            record[0]=act;
            if act!=1:
                #do something
                cash=cash-500*price.iloc[0]*(act-position)
                position-act;
            valueprev=sv;#value yeaterday

            for i in range(1,daynum):
                value=cash+500*price.iloc[i]*(position-1);#value today
                act=self.learner.query(state.iloc[i],1000*(value/valueprev-1));
                record[i]=act;
                if act!=1:
                    cash = cash - 500 * price.iloc[i] * (act - position)
                    position=act
                valueprev=value;
            iter+=1;
            if np.all([record,prevrecord]):
                break;
            else:
                prevrecord=record;






    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        x = self.getTestIndicator(symbol, sd, ed)
        # x stores the indicators
        price = ut.get_data([symbol], pd.date_range(sd, ed)).drop('SPY', 1)
        trade = price.copy();
        trade.columns = ["order"]
        # trade will contain the info of our trade
        state = x.ind1.astype(str).str.cat(x.ind2.astype(str)).str.cat(x.ind3.astype(str)).astype(int)
        # each day with the dtate
        daynum = price.shape[0]
        position =1;
        for i in range(daynum):
            act = self.learner.querysetstate(state.iloc[i])
            if act!=1:
                trade.iloc[i]=500*(act-position)
                position = act;
            else:
                trade.iloc[i] = 0;
        return trade

    def getTrainIndicator(self,symbol = "IBM",sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1)):
        #get the indicator matrix
        price=ut.get_data([symbol], pd.date_range(sd-BDay(36), ed)).drop('SPY', 1)
        ind1=price / price.shift(1) - 1;
        sma = pd.rolling_mean(price, 10);
        std = pd.rolling_std(price, 10);
        ind2 = (price - sma) / (2 * std);
        dif = price - price.shift(1);
        dif.iloc[0] = 0;
        up = dif * (dif > 0);
        down = dif * (dif < 0);
        smaup = pd.rolling_mean(up, 26);
        smadown = pd.rolling_mean(down, 26);
        ind3 = smaup / (smaup - smadown);
        length=ind1.shape[0]
        #pd.Series(ind1[symbol])
        ind1, self.bin1 = pd.qcut(ind1[symbol], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], retbins=True);
        ind2, self.bin2 = pd.qcut(ind2[symbol], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], retbins=True);
        ind3, self.bin3 = pd.qcut(ind3[symbol], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], retbins=True);
        indicators = pd.concat([ind1, ind2, ind3], axis=1);
        indicators.columns = ['ind1', 'ind2', 'ind3'];
        indicators=indicators[sd:ed]
        #print self.bin1
        self.bin1[0]=-100;
        self.bin1[-1]=100;
        #pd.cut(ind1,bins=self.bin1)
        self.bin2[0] = -100;
        self.bin2[-1] = 100;
        self.bin3[0] = -100;
        self.bin3[-1] = 100;
        return indicators

    def getTestIndicator(self,symbol = "IBM",sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1)):
        #get the indicator matrix
        price=ut.get_data([symbol], pd.date_range(sd-BDay(36), ed)).drop('SPY', 1)
        ind1=price / price.shift(1) - 1;
        sma = pd.rolling_mean(price, 10);
        std = pd.rolling_std(price, 10);
        ind2 = (price - sma) / (2 * std);
        dif = price - price.shift(1);
        dif.iloc[0] = 0;
        up = dif * (dif > 0);
        down = dif * (dif < 0);
        smaup = pd.rolling_mean(up, 26);
        smadown = pd.rolling_mean(down, 26);
        ind3 = smaup / (smaup - smadown);
        length=ind1.shape[0]
        #pd.Series(ind1[symbol])
        ind1 = pd.cut(ind1[symbol], bins=self.bin1, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        ind2 = pd.cut(ind2[symbol], bins=self.bin2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        ind3 = pd.cut(ind3[symbol], bins=self.bin3, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        indicators = pd.concat([ind1, ind2, ind3], axis=1);
        indicators.columns = ['ind1', 'ind2', 'ind3'];
        indicators=indicators[sd:ed]
        #pd.cut(ind1,bins=self.bin1)
        return indicators

if __name__=="__main__":
    print "One does not simply think up a strategy"
