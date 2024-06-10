#this file introduces a number of functions that are used to transform the EV charging data
import pandas as pd

#converting the column in the DfT dataset to pandas datetime format
def add_pdtime(x):
    x['StartPd'] = pd.to_datetime(x['StartDate'] + ' ' + x['StartTime'])
    x['EndPd'] = pd.to_datetime(x['EndDate'] + ' ' + x['EndTime'])
    return x

def add_days(x): #may need extra clause that makes sure 2018 adds 365
    x['sDay'] = pd.to_datetime(x["StartDate"], format='%Y-%m-%d').dt.dayofyear
    x['eDay'] = pd.to_datetime(x["EndDate"], format='%Y-%m-%d').dt.dayofyear
    return x

def add_mins(x):
    x['sMins'] = pd.to_datetime(x["StartTime"], format='%H:%M:%S').dt.minute
    x['eMins'] = pd.to_datetime(x["EndTime"], format='%H:%M:%S').dt.minute
    return x

def add_totmins(x):
    x['sTotMins'] = x['StartPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)
    x['eTotMins'] = x['EndPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)
    return x

def add_settl(x): #needed?
    x['sSettl'] = (x['StartPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    x['eSettl'] = (x['EndPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    return x

def add_settl_gen(x,org_name,outname):
    x[outname] = (x[org_name].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    return x

def add_speed(x): #speed in kWh/h --> kW #if one wanted the energy change per settlement, they could simply compute speed/2
    x['speed'] = x['Energy'] / ((x['EndPd'].sub(x['StartPd']).view('int64') // (60*10**9)) /60)
    return x

def add_maxspeed(x):
    x['maxspeed'] = x.groupby(['CPID'])['speed'].transform(max)
    return x

def add_maxbattery(x):
    x['maxbattery'] = x.groupby(['CPID'])['Energy'].transform(max)
    return x

def min_max_speed(x,minspeed):
    y = x.copy(deep=True)
    y['maxspeeed_clip'] = y['maxspeed']
    y['maxspeeed_clip'].clip(minspeed, inplace = True)
    return y

def min_max_battery(x,minbat):
    y = x.copy(deep=True)
    y['maxbattery'].clip(minbat, inplace = True)
    return y

#clean --> remove more than t_max hours, remove double occupied chargers
def remove48(x,t_max):
    y = x.drop(x[ (x['EndPd'] - x['StartPd']).astype('timedelta64[s]').astype('int64')/3600 > t_max].index) #removes all rows that have a charging time above t_max hours #to be tested
    return y

