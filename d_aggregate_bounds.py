import pandas as pd
import auxfunc_aggregation as pcst

#applies proposed aggregation procedure to EV dataset
def aggfunction(trial,end_dir_cha_perc = 0.8, end_dir_cha_power_perc = 0.5):

    #trainI = pcst.dropunnecessary(trial1000)

    ### 1a. get flexible & direct charging DF based on maxspeed
    flex_df = pcst.createDF_flexiblecharge(trial,end_dir_cha_perc)
    dir_ch1 = pcst.createDF_directcharge(trial,end_dir_cha_perc)

    ### 1b. shorten times in flex DF because of CCCV constraints & add column with earliest possible end time
    int2 = pcst.inflexender2(pcst.flexender(flex_df,end_dir_cha_perc,end_dir_cha_power_perc))

    ### 1c. make charges that violate new times inflexible but keep rest flexi
    flex_df2 = pcst.make_flexpd_after_start_flexible(int2)
    dir_ch2 = pcst.make_flexpd_before_start_direct(int2)

    ### 2. Upper Bound

    #spread
    dfUB_spread = pcst.spread_energyS(flex_df2)

    #transform
    aggUB = pcst.aggregatedalreadyS_spread(dfUB_spread)
    finUB = pcst.fill_missing_range(aggUB,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')


    ### 3. Lower Bound

    #a. create df for v-shape bottom & without v-shape
    flex_df2_vbott = pcst.add_vshape_col(pcst.add_flexendsettl(flex_df2),end_dir_cha_perc)
    dfv = flex_df2_vbott.loc[flex_df2_vbott['v-shape'] == 1] #tested and approved
    dfnv = flex_df2_vbott.loc[flex_df2_vbott['v-shape'] == 0]

    #add info about bottom v
    dfvx = pcst.add_vbottom_energy(pcst.add_vbottom_time(dfv)) #tested and approved

    #b. spread

    #spread non-v-shape left
    dfLB_spread_left_nv = pcst.spread_energyS_L(dfnv)

    #spread non-v-shape right
    dfLB_spread_right_nv = pcst.spread_energyE(dfnv,end_dir_cha_perc) #there are some nasty things here

    #spread v-shape lower bound left
    dfLB_spread_left_v = pcst.left_v(dfvx)

    #spread v-shape lower bound right
    dfLB_spread_right_v = pcst.right_v(dfvx)

    #c. transform
    #nvleft
    lnv = dfLB_spread_left_nv.copy(deep=True)
    aggLB_lnv = lnv.groupby(lnv['sSettl_cumsum']).aggregate({'new_neg_energy':'sum'})
    finLB_lnv = pcst.fill_missing_range(aggLB_lnv,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')
    #nvright
    rnv = dfLB_spread_right_nv.copy(deep=True)
    aggLB_rnv = rnv.groupby(rnv['eSettl_cumsum']).aggregate({'new_energy':'sum'})
    finLB_rnv = pcst.fill_missing_range(aggLB_rnv,'eSettl_cumsum',1,17521).drop(columns='eSettl_cumsum')
    #vleft
    lv = dfLB_spread_left_v.copy(deep=True)
    aggLB_lv = lv.groupby(lv['sSettl_cumsum']).aggregate({'new_energy_vb':'sum'})
    finLB_lv = pcst.fill_missing_range(aggLB_lv,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')
    #vright
    rv = dfLB_spread_right_v.copy(deep=True)
    aggLB_rv = rv.groupby(rv['sSettl_cumsum']).aggregate({'new_energy_vb2':'sum'})
    finLB_rv = pcst.fill_missing_range(aggLB_rv,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')

    #add together lower bound
    LB_series = finLB_rv['new_energy_vb2'] + finLB_rnv['new_energy'] - finLB_lv['new_energy_vb'] - finLB_lnv['new_neg_energy']

    #4. Power Bound

    #spread
    z2int = pcst.spread_power(flex_df2)

    #aggregate
    pBint = z2int.copy(deep=True)
    pBint2 = pBint.groupby(pBint['sSettl_cumsum']).aggregate({'power':'sum'})
    pBfin = pcst.fill_missing_range(pBint2,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')

    #4a. Number of EVs
    z2int_noEVs = pcst.spread_noev(flex_df2)
    pBnvint = z2int_noEVs.copy(deep=True)
    pBnVint2 = pBnvint.groupby(pBint['sSettl_cumsum']).aggregate({'ones1_ev':'sum'})
    pBfinV = pcst.fill_missing_range(pBnVint2,'sSettl_cumsum',1,17521).drop(columns='sSettl_cumsum')

    # 5. Direct Charging
    #3. assort direct charge
    int1x = pcst.conc_dir_dfs(dir_ch1,dir_ch2,flex_df2) #concat all three dc dfs
    int2x = pcst.timeline_dir_charge(int1x) #create df with spread out charge times
    fin3x = pcst.dir_charge_series(int2x) #create timeseries with charge
    fin4x = pcst.fill_missing_range(fin3x,'sSettl_cumsum',1,17521) #complete timeline with zeroes
    #fin4xplus = pcst.fill_missing_range(fin3x,'sSettl_cumsum',1,iTrainE.index[-1]+1)

    #6. price data
    #from fP4_pricedata import pricelist #not the quickest way to do this
    priceseries = pd.read_csv("../Raw data/wholesaleprice.csv",header=None)[0]

    #7. weather data
    temp_df = pd.read_excel('../Raw data/metoffice_hadcet_data_meantemp_daily_2017.xlsx')
    precip_df = pd.read_excel('../Raw data/metoffice_hadukp_data_daily_HadEWP_daily_totals_2017.xlsx')
    temp_ser= pd.Series(temp_df.loc[0:365,'Value'].repeat(48).values)
    precip_ser = pd.Series(precip_df.loc[0:365,'Value'].repeat(48).values)

    #8. bank holidays
    bh = [2,104,107,121,149,240,359,360]
    bh_ser = [0]*365*48
    for i in bh:
        bh_ser[(i-1)*48:(i*48)] = [1]*48
    bh_ser = pd.Series(bh_ser)

    #9. day of week
    settl_list = [m for m in range(17520)]
    pdtimelist = [(pd.to_datetime('2017-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S') + pd.to_timedelta(n/2,unit='h')).weekday() for n in settl_list]
    week_ser = pd.Series(pdtimelist)

    #put together into 1 dataframe
    df_data = {'LB':LB_series,'UB':finUB['new_energy'],'Power':pBfin['power'],'No_EVs':pBfinV['ones1_ev'],'DCharge':fin4x['fEn'],'Price':priceseries, 'Temperature':temp_ser, 'Precipitation': precip_ser, 'BankHoliday': bh_ser, 'Weekday': week_ser}
    df_datac = {'LB':LB_series.cumsum(),'UB':finUB['new_energy'].cumsum(),'Power':pBfin['power'],'No_EVs':pBfinV['ones1_ev'],'DCharge':fin4x['fEn'],'Price':priceseries, 'Temperature':temp_ser, 'Precipitation': precip_ser, 'BankHoliday': bh_ser, 'Weekday': week_ser}
    df_datab = {'LB':LB_series,'UB':finUB['new_energy'],'LBc':LB_series.cumsum(),'UBc':finUB['new_energy'].cumsum(),'Power':pBfin['power'],'No_EVs':pBfinV['ones1_ev'],'DCharge':fin4x['fEn'],'Price':priceseries, 'Temperature':temp_ser, 'Precipitation': precip_ser, 'BankHoliday': bh_ser, 'Weekday': week_ser}

    df_nc = pd.DataFrame(df_data)
    df_c = pd.DataFrame(df_datac)
    df_b = pd.DataFrame(df_datab)
    return df_b

