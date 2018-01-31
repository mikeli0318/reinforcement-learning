"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl

def test_code(sym1="GOOG",st1=dt.datetime(2008,1,1),ed1=dt.datetime(2008,1,1),sym2="GOOG",st2=dt.datetime(2008,1,1),ed2=dt.datetime(2008,1,1)):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = False)

    # set parameters for training the learner
    sym = sym1
    stdate =st1
    enddate =ed1 # just a few days for "shake out"

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000) 

    # set parameters for testing
    sym = sym2
    stdate =st2
    enddate =ed2

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # test the learner
    df_trades = learner.testPolicy(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=500] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violoates holding restrictions (more than 500 shares)"

    res=prices.copy()
    res["do"]=df_trades
    print res




if __name__=="__main__":
    test_code(sym1="IBM",st1=dt.datetime(2006,1,1),ed1=dt.datetime(2009,12,31),sym2="IBM",st2=dt.datetime(2010,1,1),ed2=dt.datetime(2010,12,31))
