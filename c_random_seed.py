import pandas as pd

from b_add_EVdata import csv2g_final3

#this is a one-time random seed of all the charger numbers that are included in the DfT EV dataset
thldf = pd.read_csv('../Raw data/random_charger_seed1.csv')
trialhardlist = list(thldf.columns.values)

def pickXoutofHList(indata,samplesize,hlist):
    outdata = indata[indata['CPID'].isin(hlist[:samplesize])]
    return outdata

noEVtriallist = [pickXoutofHList(csv2g_final3,(i+1)*20,trialhardlist) for i in range(40)]

noEVtriallist8001000 = [pickXoutofHList(csv2g_final3,(i+1+40)*20,trialhardlist) for i in range(10)]

#these lists are later used to test the effect of fleet size on the algorithm's performance