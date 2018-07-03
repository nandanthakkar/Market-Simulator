

import pandas as pd
import numpy as np
import datetime as dt
import math
import os
import sys
sys.path.append(os.path.abspath("/Users/nandan/Documents/Data Science/ML-Trading/"))
from util import get_data, plot_data

def update_portvals(start_date, end_date, accountvals, stocksvals, portvals, stockdailychange, start_val):
    
    
    if start_date == -1:
        dates = pd.date_range(stocksvals.index[0], end_date)
        dates = np.intersect1d(dates, stocksvals.index)
        for date in dates:
            portvals.Cash.loc[date] = start_val
            portvals.Value.loc[date] = 0
    else:
        dates = pd.date_range(start_date, end_date)
        dates = np.intersect1d(dates, stocksvals.index)
        if dates.shape[0] == 0:
            print("startdate", start_date, "end_date", end_date)
        start = dates[0]
        for i in dates[1:]:
            
            portvals.loc[i] = portvals.loc[start]
            for index, shares in accountvals.iterrows():
                dailychange = stockdailychange[index].loc[i]*shares
                portvals['Value'].loc[i] +=  dailychange.values
            start = i
                
            
            

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):

    sys.path.append(os.path.abspath("/Users/nandan/Documents/Data Science/ML-Trading/marketsim/"))
    df = pd.read_csv(orders_file, index_col='Date',parse_dates=True, na_values=['nan'])
    df = df.sort_index()
    stockvals = get_data(df.Symbol.unique(), pd.date_range(df.index[0], df.index[-1]), addSPY=False)

    SPY = get_data(['SPY'], pd.date_range(df.index[0], df.index[-1]), addSPY=False)

    stockvals = stockvals.join(SPY, how = 'inner')
    stockdailychange = stockvals[1:] - stockvals[:-1].values
    accountvals = pd.DataFrame(index = df.Symbol.unique(), columns = ['Shares'])
    accountvals.Shares = np.zeros(df.Symbol.unique().shape[0])
  
    portvals = pd.DataFrame(index = SPY.index, columns=['Value', 'Cash'])
    portvals.Value = np.zeros(SPY.index.shape[0])
    portvals.Cash = np.zeros(SPY.index.shape[0])
    portvals.iloc[0] = [0,start_val]
    
    start_date = -1
    for index, row in df.iterrows():

        update_portvals(start_date, index, accountvals, stockvals, portvals, stockdailychange, start_val)
        if row[1] == 'BUY':
            
            val = stockvals[row[0]].loc[index]
            cost = val * row[2] 
            accountvals.loc[row[0]] += row[2]
            portvals['Cash'].loc[index] -= cost*(1.0 + impact) + commission
            portvals['Value'].loc[index] += cost

        else:
            val = stockvals[row[0]].loc[index]
            cost = val * row[2] 
            accountvals.loc[row[0]] -= row[2]
            portvals['Cash'].loc[index] += cost* (1.0 - impact) - commission
            portvals['Value'].loc[index] -= cost
 
        start_date = index
    #print(portvals)
    SPYvals = pd.DataFrame(index = SPY.index, columns=['Value', 'Cash'])
    SPYvals.Value = np.zeros(SPY.index.shape[0])
    SPYvals.Cash = np.zeros(SPY.index.shape[0])
    SPYvals.iloc[0] = [0,start_val]
    numSPY = math.floor(start_val/stockvals['SPY'].loc[SPY.index[0]])
    Value = numSPY * stockvals['SPY'].loc[SPY.index[0]]
    Cash = start_val - Value
    SPYvals['Value'].loc[SPY.index[0]] = Value
    SPYvals['Cash'].loc[SPY.index[0]] = Cash
    SPYshares = pd.DataFrame(index = ['SPY'], columns = ['Shares'])
    SPYshares['Shares'].loc['SPY'] = numSPY
    #print(SPYvals)
    update_portvals(SPY.index[0], SPY.index[-1], SPYshares, stockvals, SPYvals, stockdailychange, start_val)
    result = pd.DataFrame(index = SPY.index, columns=['portvals', 'SPY'])
    result['portvals'] =portvals.sum(axis=1)
    result['SPY'] = SPYvals.sum(axis=1)
    #print(SPYvals)
    #print(result)
    return result #portvals.sum(axis=1)


            
            

def test_code():

    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv, commission= 0.0, impact=0.0)
    if isinstance(portvals, pd.DataFrame):
        print("passed")#portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    
    
    port_val_pct_change = portvals['portvals'].pct_change()
    avg_daily_ret = port_val_pct_change.mean()
    std_daily_ret = port_val_pct_change.std()
    sharpe_ratio = ((avg_daily_ret)* np.sqrt(252))/std_daily_ret
    #print(portvals)
    start_date = portvals.index[0] #dt.datetime(2008,1,1)
    end_date = portvals.index[-1] #dt.datetime(2008,6,1)
    cum_ret = portvals['portvals'].loc[end_date] # avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY = portvals['SPY'].loc[end_date]
    SPY_val_pct_change = portvals['SPY'].pct_change()
    avg_daily_ret_SPY = SPY_val_pct_change.mean()
    std_daily_ret_SPY = SPY_val_pct_change.std()
    sharpe_ratio_SPY =  ((avg_daily_ret_SPY)* np.sqrt(252))/std_daily_ret_SPY#[0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print ("Date Range: {} to {}".format(start_date, end_date))
    print()
    print ("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print()
    print ("Cumulative Return of Fund: {}".format(cum_ret))
    print ("Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print()
    print ("Standard Deviation of Fund: {}".format(std_daily_ret))
    print ("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print()
    print ("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print ("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print()
    print ("Final Portfolio Value: {}".format(portvals['portvals'].loc[end_date]))
    print ("Final SPY Value: {}".format(portvals['SPY'].loc[end_date]))

if __name__ == "__main__":
    test_code()
