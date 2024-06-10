import pandas as pd
import numpy as np

from auxfunc_data import add_settl_gen

def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0): #source: https://stackoverflow.com/questions/25909984/missing-data-insert-rows-in-pandas-and-fill-with-nan
    return df\
      .merge(how='right', on=field,
            right = pd.DataFrame({field:np.arange(range_from, range_to, range_step)}))\
      .sort_values(by=field).reset_index().fillna(fill_with).drop(['index'], axis=1)

def dropunnecessary(x):
    x.drop(columns = ['CPID','ChargingEvent','PluginDuration','StartPd','EndPd','speed'])
    return x

def red_energy_batprot(x,perc): #this can be problematic because sometimes the x% of battery capacity might be larger than the energy to be charged --> need to account for that by taking these charging processes out?
    x['Energy_perc'] = x['Energy'] - (x['maxbattery']*(1-perc))
    #this bottom bit is a first measure to prevent --> will be dropped for now (this also makes most sense) --> later should be included in direct charging --> has been included in DC
    #x.loc[x['Energy'] < (x['maxbattery']*(1-perc)), 'Energy_perc'] = x['Energy'] #this shouldn't really be an issue because these rows be dropped in the function below anyway but I commented it out for now
    x['Energy_perc_bool'] = x['Energy'] < (x['maxbattery']*(1-perc))
    x['Energy_perc'].clip(0, inplace = True)
    return x

def drop_bat_bigger_E(x): #drop cases in which battery end charge is bigger than total charge #not needed
    y = x[x['Energy_perc_bool'] == False]
    return y

def spread_energyS(x): #i think this is correct --> but need to check?
    #x = red_energy_batprot(x, perce) 
    #x1 = drop_bat_bigger_E(x)
    y = x.loc[x.index.repeat(x['Energy_perc'] // (x['maxspeeed_clip']/2) + 1)]
    y['ones1'] = 1
    y['sSettl_cumsum'] = y['ones1'].groupby(y.index).cumsum() + y['sSettl']
    y['new_energy'] = y['maxspeeed_clip'] / 2
    y['new_en_cumsum'] = y['new_energy'].groupby(y.index).cumsum()
    y.loc[y['new_en_cumsum'].gt(y['Energy_perc']), 'new_energy'] = (y.loc[y['new_en_cumsum'].gt(y['Energy_perc']),'Energy_perc']
                                                                    - y.loc[y['new_en_cumsum'].gt(y['Energy_perc']),'new_en_cumsum']
                                                                    + y.loc[y['new_en_cumsum'].gt(y['Energy_perc']),'maxspeeed_clip']/2)
    return y

def spread_energyS_L(x): #this should pretty much stay the same --> test
    #x = red_energy_batprot(x, per) 
    #x1 = drop_bat_bigger_E(x)
    y = x.loc[x.index.repeat((x['maxbattery'] - x['Energy']) // (x['maxspeeed_clip']/2) + 1)]
    y['ones1'] = 1
    y['sSettl_cumsum'] = y['ones1'].groupby(y.index).cumsum() + y['sSettl']
    y['new_neg_energy'] = y['maxspeeed_clip'] / 2
    y['new_neg_en_cumsum'] = y['new_neg_energy'].groupby(y.index).cumsum()
    y.loc[y['new_neg_en_cumsum'].gt(y['maxbattery'] - y['Energy']), 'new_neg_energy'] = (y.loc[y['new_neg_en_cumsum'].gt(y['maxbattery']- y['Energy']),'maxbattery']
                                                                                         - y.loc[y['new_neg_en_cumsum'].gt(y['maxbattery'] - y['Energy']),'Energy']
                                                                                         - y.loc[y['new_neg_en_cumsum'].gt(y['maxbattery'] - y['Energy']),'new_neg_en_cumsum']
                                                                                         + y.loc[y['new_neg_en_cumsum'].gt(y['maxbattery'] - y['Energy']),'maxspeeed_clip']/2)
    return y

def spread_energyE(x,percen): #v problem (energy needing to go up again when its still going down) --> sort in different function (this function will only be applied to non-v-shapes)
    #x = red_energy_batprot(x, percen) 
    #x1 = drop_bat_bigger_E(x)
    x['dir_diff'] = pd.to_timedelta((((percen)*x['maxbattery']/x['maxspeeed_clip'])*60), unit='m') #time at the end that is reserved for direct charging
    x['EndPd_dir'] = x['FlexEndPd'] - pd.to_timedelta(((percen)*x['maxbattery']/x['maxspeeed_clip']*60), unit='m') #and the resulting time stamp at which variable charging therefore has to end
    
    x.loc[x['Energy_perc_bool'] == True, 'EndPd_dir'] = x['FlexEndPd'] - pd.to_timedelta(((x['maxbattery']-x['Energy'])/x['maxspeeed_clip']*60), unit='m') #added in state of near insomnia
    
    z = add_settl_gen(x,'EndPd_dir','DirEndSettl') #resulting settlement
    #z = red_energy_batprot(y, percen)
    #replace with settlement based counting method #z2 = z.loc[z.index.repeat((z['maxbattery'] * percen) // (z['maxspeeed_clip']/2) + 1)] #+max_battery was added later
    z2 = z.loc[z.index.repeat(z['FlexEndSettl'] - z['DirEndSettl'] + 1)] #+max_battery was added later #when is FlexEndSettl added?
    z2['ones1'] = 1
    z2['eSettl_cumsum'] = z2['DirEndSettl'] + z2['ones1'].groupby(z2.index).cumsum() -1
    z2['new_energy'] = z2['maxspeeed_clip'] / 2

    z2.loc[z2['eSettl_cumsum'] == z2['DirEndSettl'], 'new_energy'] = ceil_dt(z2.loc[z2['eSettl_cumsum'] == z2['DirEndSettl'],'EndPd_dir'])/pd.to_timedelta(0.5,unit='h') * z2.loc[z2['eSettl_cumsum'] == z2['DirEndSettl'],'maxspeeed_clip'] /2
    z2['new_en_cumsum'] = z2['new_energy'].groupby(z2.index).cumsum()

    #below also altered late at night
    z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery']*percen)) & (z2['Energy_perc_bool'] == False), 'new_energy'] = (z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery']*percen)) & (z2['Energy_perc_bool'] == False),'maxbattery'] * percen
                                                                                          - z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery']*percen)) & (z2['Energy_perc_bool'] == False),'new_en_cumsum']
                                                                                          + z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery']*percen)) & (z2['Energy_perc_bool'] == False),'maxspeeed_clip']/2)
    
    z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery'] - z2['Energy'])) & (z2['Energy_perc_bool'] == True), 'new_energy'] = (z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery'] - z2['Energy'])) & (z2['Energy_perc_bool'] == True),'maxbattery']
                                                                                          - z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery'] - z2['Energy'])) & (z2['Energy_perc_bool'] == True),'Energy']
                                                                                          - z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery'] - z2['Energy'])) & (z2['Energy_perc_bool'] == True),'new_en_cumsum']
                                                                                          + z2.loc[(z2['new_en_cumsum'].gt(z2['maxbattery'] - z2['Energy'])) & (z2['Energy_perc_bool'] == True),'maxspeeed_clip']/2)
    z2.loc[(z2['DirEndSettl'] == z2['FlexEndSettl']) & (z2['Energy_perc_bool'] == False), 'new_energy'] = z2.loc[z2['DirEndSettl'] == z2['FlexEndSettl'], 'maxbattery'] * percen
    z2.loc[(z2['DirEndSettl'] == z2['FlexEndSettl']) & (z2['Energy_perc_bool'] == True), 'new_energy'] = z2.loc[z2['DirEndSettl'] == z2['FlexEndSettl'], 'maxbattery'] - z2.loc[z2['DirEndSettl'] == z2['FlexEndSettl'], 'Energy'] 
    return z2 #maxbattery was added later

def concatenate_LB(St_LB,En_LB):
    St_LB['new_energy'] = St_LB.loc[:,'new_neg_energy']*-1
    St_LB['genSettl'] = St_LB['sSettl_cumsum']
    En_LB['genSettl'] = En_LB['eSettl_cumsum']
    outdf = pd.concat([St_LB[['genSettl','new_energy']],En_LB[['genSettl','new_energy']]],ignore_index=True)
    return outdf

def aggregatedalreadyS(x):
    aggregation_functions = {'maxbattery': 'sum', 'maxspeed': 'sum', 'Energy': 'sum'}
    y = x.copy(deep=True)
    y = y.groupby(y['sSettl']).aggregate(aggregation_functions)
    return y

def aggregatedalreadyE(x):
    aggregation_functions = {'maxbattery': 'sum', 'maxspeed': 'sum', 'Energy': 'sum'}
    y = x.copy(deep=True)
    y = y.groupby(y['eSettl']).aggregate(aggregation_functions)
    return y

def aggregatedalreadyS_spread(x):
    aggregation_functions = {'new_energy': 'sum'}
    y = x.copy(deep=True)
    y = y.groupby(y['sSettl_cumsum']).aggregate(aggregation_functions)
    return y

def aggregatedalreadyE_spread(x):
    aggregation_functions = {'new_energy': 'sum'}
    y = x.copy(deep=True)
    y = y.groupby(y['genSettl']).aggregate(aggregation_functions)
    return y

#new since October 2023

#CCCV and direct charging

def createDF_directcharge(x,threshold):
    x['direct_charge'] = 0
    x.loc[x['speed'] > x['maxspeed'] * threshold,'direct_charge'] = 1
    y = x[x['direct_charge'] == 1].copy()
    return y

def createDF_flexiblecharge(x,threshold):
    x['direct_charge'] = 0
    x.loc[x['speed'] > x['maxspeed'] * threshold,'direct_charge'] = 1
    y = x[x['direct_charge'] == 0].copy()
    return y

def flexender(x,end_dir_charge_perc,CVspeed): #this adds the time at which an EV would have to stop charging to honour charging the last few percent SOC (1-"end_dir_charge") at CVspeed% of its maxspeed 
    x = red_energy_batprot(x, end_dir_charge_perc)
    x['FlexEndPd'] = x['EndPd'] - pd.to_timedelta(((x['maxbattery']*(1-end_dir_charge_perc))/(x['maxspeeed_clip']*CVspeed)), "hours") #maxspeed here also needs to be divided by two --> not the case
    x.loc[x['Energy_perc_bool'] == True,'FlexEndPd'] = x['EndPd'] - pd.to_timedelta(((x['Energy'])/(x['maxspeeed_clip']*CVspeed)), "hours") #add line that adds different FlexEndPd for charging processes where the end charge is more than the charged energy (Energy_perc_bool == True)
    return x 

#this considers the starting time and ensures that charges which only consist of "end-phase" charging are dropped
def make_flexpd_before_start_direct(x): #this gives a dataframe of all inflexible (or "direct") charging because of the CCCV period before the end
    y = x[x['inflexstart2'] > x['FlexEndPd']].copy()
    return y

def make_flexpd_after_start_flexible(x): #this gives a dataframe of all flexible (or "indirect") charging even with CCCV period before the end
    y = x[x['inflexstart2'] < x['FlexEndPd']].copy()
    return y

#now lets add the fact that the discretisation only starts on the next time interval def inflexender(x): #this adds the earliest possible time at which an EV could stop charging to reach CC full energy ("end_dir_charge")
def inflexender2(x): #this adds the earliest possible time at which an EV could stop charging to reach CC full energy ("end_dir_charge")
    #x = red_energy_batprot(x, percen) #moved to flexender
    x['inflexstart2'] = pd.to_datetime('2017-01-01 00:00:00') + pd.to_timedelta(x['sSettl']/2,"hours") + pd.to_timedelta(x["Energy_perc"].div(x['maxspeeed_clip']),"hours") 
    return x

def add_flexendsettl(dfvx):
    dfvx['FlexEndSettl']= (dfvx['FlexEndPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    return dfvx

#bottom v-shape
def add_vshape_col(x,perc):
    x['v-shape'] = 0
    filter_a = pd.to_timedelta(((x['maxbattery'] - x['Energy'] + x['maxbattery']*perc)/x['maxspeeed_clip']),'hours') > (x['FlexEndPd'] - (pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + pd.to_timedelta(x['sSettl']/2,unit='h')))
    filter_b = pd.to_timedelta((((x['maxbattery'] - x['Energy'])*2)/x['maxspeeed_clip']),'hours') > (x['FlexEndPd'] - (pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + pd.to_timedelta(x['sSettl']/2,unit='h')))
    x.loc[(x['Energy_perc_bool'] == False) & filter_a,'v-shape'] = 1
    x.loc[(x['Energy_perc_bool'] == True) & filter_b,'v-shape'] = 1
    return x

def add_vbottom_time(x):
    x['vbot_time_delt'] = (x['FlexEndPd'] - 
                           (pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + #sSettl pdtime
                            pd.to_timedelta(x['sSettl']/2,unit='h')) - #sSettl pdtime continued
                            pd.to_timedelta(x['Energy_perc']/x['maxspeeed_clip'], "hours"))/2
    x['vbot_time_abs'] = (pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + #sSettl pdtime
                            pd.to_timedelta(x['sSettl']/2,unit='h') + #sSettl pdtime continued
                            x['vbot_time_delt']) #additional discharge time to get to bottom
    x['vbot_settl'] = (x['vbot_time_abs'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1 #
    return x

def add_vbottom_energy(x):
    x['neg_vbottom_energy'] = x['vbot_time_delt']/ np.timedelta64(1, 'h') * x['maxspeeed_clip']
    return x

#add bottom v-shape
#not sure if bullet points below were actually followed like that precisely
#1. for ssettl+1 to vbot_settl-1 subtract maxspeed/2
#2. for vbot_settl add Energy_perc + (1.) - (3.) - (4.) ### or -(neg_vbottom_energy%maxspeed) + ((vbot_settl + 1) - vbot_time_abs)/30 * maxspeed/2
#3. for vbot_settl+1 to settl_flexendpd -1 add maxspeed/2
#4. for settl_flexendpd add maxspeed/2 * (FlexEndPd - flexendsettl)/30 mins

#left side of v
def left_v(x):
    y = x.loc[x.index.repeat(x['vbot_settl'] - x['sSettl'])] #works 
    y['ones1'] = 1
    y['sSettl_cumsum'] = y['ones1'].groupby(y.index).cumsum() + y['sSettl']
    y['new_energy_vb'] = y['maxspeeed_clip'] / 2
    y['new_en_cumsum_vb'] = y['new_energy_vb'].groupby(y.index).cumsum()
    y.loc[y['new_en_cumsum_vb'].gt(y['neg_vbottom_energy']), 'new_energy_vb'] = (y.loc[y['new_en_cumsum_vb'].gt(y['neg_vbottom_energy']),'neg_vbottom_energy']
                                                                    - y.loc[y['new_en_cumsum_vb'].gt(y['neg_vbottom_energy']),'new_en_cumsum_vb']
                                                                    + y.loc[y['new_en_cumsum_vb'].gt(y['neg_vbottom_energy']),'maxspeeed_clip']/2)
    y.loc[y['vbot_settl'] == y['sSettl'], 'new_energy_vb'] = y.loc[y['vbot_settl'] == y['sSettl'],'neg_vbottom_energy'] #need to test this briefly
    return y

#right side of v
def right_v(x):
    y1 = x.loc[x.index.repeat(x['FlexEndSettl'] - x['vbot_settl'] + 1)] #works 
    y1['ones1'] = 1
    y1['sSettl_cumsum'] = y1['ones1'].groupby(y1.index).cumsum() + y1['vbot_settl'] - 1
    y1['new_energy_vb2'] = y1['maxspeeed_clip'] / 2
    y1.loc[y1['sSettl_cumsum'] == y1['vbot_settl'], 'new_energy_vb2'] = ceil_dt(y1.loc[y1['sSettl_cumsum'] == y1['vbot_settl'],'vbot_time_abs'])/pd.to_timedelta(0.5,unit='h') * y1.loc[y1['sSettl_cumsum'] == y1['vbot_settl'],'maxspeeed_clip'] /2
    y1['new_en_cumsum_vb2'] = y1['new_energy_vb2'].groupby(y1.index).cumsum()
    y1.loc[y1['new_en_cumsum_vb2'].gt(y1['neg_vbottom_energy'] + y1['Energy_perc']), 'new_energy_vb2'] = (y1.loc[y1['new_en_cumsum_vb2'].gt(y1['neg_vbottom_energy'] + y1['Energy_perc']),'neg_vbottom_energy']
                                                                    + y1.loc[y1['new_en_cumsum_vb2'].gt(y1['neg_vbottom_energy'] + y1['Energy_perc']),'Energy_perc']
                                                                    - y1.loc[y1['new_en_cumsum_vb2'].gt(y1['neg_vbottom_energy'] + y1['Energy_perc']),'new_en_cumsum_vb2']
                                                                    + y1.loc[y1['new_en_cumsum_vb2'].gt(y1['neg_vbottom_energy'] + y1['Energy_perc']),'maxspeeed_clip']/2)
    y1.loc[y1['vbot_settl'] == y1['FlexEndSettl'], 'new_energy_vb2'] = y1.loc[y1['vbot_settl'] == y1['FlexEndSettl'],'neg_vbottom_energy'] + y1.loc[y1['vbot_settl'] == y1['FlexEndSettl'],'Energy_perc']
    return y1

#power spread
def spread_power(x):
    z2int = x.loc[x.index.repeat(x['FlexEndSettl'] - x['sSettl'])]
    z2int['ones1'] = 1
    z2int['sSettl_cumsum'] = z2int['sSettl'] + z2int['ones1'].groupby(z2int.index).cumsum()
    z2int['power'] = z2int['maxspeeed_clip']
    z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'power'] = (pd.to_timedelta(0.5,unit='h') - ceil_dt(z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'FlexEndPd']))/pd.to_timedelta(0.5,unit='h') * z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'maxspeeed_clip']
    return z2int

#ev number spread
def spread_noev(x):
    z2int = x.loc[x.index.repeat(x['FlexEndSettl'] - x['sSettl'])]
    z2int['ones1_ev'] = 1
    z2int['sSettl_cumsum'] = z2int['sSettl'] + z2int['ones1_ev'].groupby(z2int.index).cumsum()
    #z2int['power'] = z2int['maxspeeed_clip']
    #z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'power'] = (pd.to_timedelta(0.5,unit='h') - ceil_dt(z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'FlexEndPd']))/pd.to_timedelta(0.5,unit='h') * z2int.loc[z2int['sSettl_cumsum'] == z2int['FlexEndSettl'],'maxspeeed_clip']
    return z2int


###direct charging

#concat dataframes
def conc_dir_dfs(dc1,dc2,flexdc):
    #create start time for direct charging (end time is always "EndPd")
    dc1['dir_stpd'] = dc1['StartPd']
    dc2['dir_stpd'] = dc2['StartPd']
    flexdc['dir_stpd'] = flexdc['FlexEndPd']

    #create direct charge energy column
    dc1['dir_energyF'] = dc1['Energy']
    dc2['dir_energyF'] = dc2['Energy']
    flexdc['dir_energyF_int'] = flexdc['maxbattery']*(1-0.9)
    flexdc['dir_energyF'] = flexdc['dir_energyF_int'].clip(None,flexdc['Energy'])

    #concat
    dir_ch_totx = pd.concat([dc1[['dir_stpd','dir_energyF','EndPd']],dc2[['dir_stpd','dir_energyF','EndPd']],flexdc[['dir_stpd','dir_energyF','EndPd']]])

    return dir_ch_totx

#spread timeline for direct charging
def timeline_dir_charge(dir_ch_tot):
    dir_ch_tot['dir_speed'] = dir_ch_tot['dir_energyF']/((dir_ch_tot['EndPd'] - dir_ch_tot['dir_stpd']).dt.seconds/3600) #find average speed for direct charge
    dir_ch_tot['sSettl'] = (dir_ch_tot['dir_stpd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    dir_ch_tot['eSettl'] = (dir_ch_tot['EndPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') // (60*10**9)) // 30 + 1
    dir_ch_tot['sSettl_float'] = (dir_ch_tot['dir_stpd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') / (60*10**9)) / 30
    dir_ch_tot['eSettl_float'] = (dir_ch_tot['EndPd'].sub(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')).view('int64') / (60*10**9)) / 30 + 1
    y = dir_ch_tot.loc[dir_ch_tot.index.repeat(dir_ch_tot['eSettl'] - dir_ch_tot['sSettl']+1)]
    y['ones1'] = 1
    y['sSettl_cumsum'] = y['ones1'].groupby(y.index).cumsum() + y['sSettl'] -1
    y['fEn'] = 0
    y.loc[y['sSettl'] == y['sSettl_cumsum'],'fEn'] = y.loc[y['sSettl'] == y['sSettl_cumsum'],"dir_speed"] / 2 * (y.loc[y['sSettl'] == y['sSettl_cumsum'],"sSettl"] - y.loc[y['sSettl'] == y['sSettl_cumsum'],"sSettl_float"])
    y.loc[y['eSettl'] == y['sSettl_cumsum'],'fEn'] = y.loc[y['eSettl'] == y['sSettl_cumsum'],"dir_speed"] / 2 * (y.loc[y['sSettl'] == y['sSettl_cumsum'],"eSettl_float"] - y.loc[y['sSettl'] == y['sSettl_cumsum'],"eSettl"])
    y.loc[(y['sSettl'] < y['sSettl_cumsum']) & (y['eSettl'] > y['sSettl_cumsum']),'fEn'] = y.loc[(y['sSettl'] < y['sSettl_cumsum']) & (y['eSettl'] > y['sSettl_cumsum']),"dir_speed"] / 2
    
    return y

#turn into time series
def dir_charge_series(x):
    aggregation_functions = {'fEn': 'sum'}
    y = x.copy(deep=True)
    y = y.groupby(y['sSettl_cumsum']).aggregate(aggregation_functions)
    return y

#auxiliary
def ceil_dt(dt):
    return (pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') - dt) % pd.to_timedelta(0.5,unit='h')