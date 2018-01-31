import datetime as dt
from pandas.tseries.offsets import BDay



if __name__=="__main__":
    stdate = dt.datetime(2008, 1, 1)
    print stdate
    print stdate-BDay(200)


