import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as sps

#creates dataframe with shifted columns to train prediction model
def xyfunc(dfy,topred,frompred,train=True):
    if train == True:
        topred_list = [frompred+topred+48*i for i in range(12240//48) if frompred+topred+48*i <= 12240]
    elif train == False:
        topred_list = [frompred+topred+48*i + 12240 for i in range(5280//48) if frompred+topred+48*i + 12240 <= 17520]
    dfy['UB_loc'] = dfy['UBc'].shift(topred+1)
    dfy['LB_loc'] = dfy['LBc'].shift(topred+1)
    dfy['Power_loc'] = dfy['Power'].shift(topred+1)
    dfy.fillna(0)
    df_out = dfy.loc[dfy.index.isin(topred_list)].fillna(0)
    dfy = dfy.drop(columns=['UB_loc','LB_loc'])
    df_out.loc[:,'UB_loc_c'] = df_out.loc[:,'UBc'] - df_out.loc[:,'UB_loc']
    df_out.loc[:,'LB_loc_c'] = df_out.loc[:,'LBc'] - df_out.loc[:,'LB_loc']
    df_out.loc[:,'deter_dist'] = df_out.loc[:,'UB_loc'] - df_out.loc[:,'LB_loc']
    return df_out

#creates distributions for any prediction
def create_preds_dist(df_from,df_to,auc_settl,no_settl,distlist): #day = day of year (starting from 1)
    no_scen = len(distlist)
    df_from['diffUBLB'] = df_from['UBc'] - df_from['LBc']
    df_to['diffUBLB'] = df_to['UBc'] - df_to['LBc']
    
    df_list_trn = [] #creates list of dataframes with a frame for each settlement in 14-23 (66)
    #df_list_tst = [] #same but for training dataset
    for i in range(no_settl):
        dfx = xyfunc(df_from,i,auc_settl,train=True)
        #dfy = xyfunc(df_to,i,auc_settl,train=False)
        df_list_trn.append(dfx)
        #df_list_tst.append(dfy)

    #predict in sample
    model_list_UB = []
    model_list_LB = []
    model_list_dLB = []
    model_list_P = []
    for ii in range(len(df_list_trn)):
        model_list_UB.append(smf.ols(formula='UB_loc_c ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        model_list_dLB.append(smf.ols(formula='diffUBLB ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        model_list_LB.append(smf.ols(formula='LB_loc_c ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        model_list_P.append(smf.ols(formula='Power ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        df_list_trn[ii]['UB_loc_c_pred'] = smf.ols(formula='UB_loc_c ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit().predict(df_list_trn[ii])
        df_list_trn[ii]['LB_loc_c_pred'] = smf.ols(formula='LB_loc_c ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit().predict(df_list_trn[ii])
        df_list_trn[ii]['Power_pred'] = smf.ols(formula='Power ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit().predict(df_list_trn[ii])
        df_list_trn[ii]['diff_pred'] = smf.ols(formula='diffUBLB ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit().predict(df_list_trn[ii])
        df_list_trn[ii]['UBresid'] = df_list_trn[ii]['UB_loc_c'] - df_list_trn[ii]['UB_loc_c_pred']
        df_list_trn[ii]['LBresid'] = df_list_trn[ii]['LB_loc_c'] - df_list_trn[ii]['LB_loc_c_pred']
        df_list_trn[ii]['Powerresid'] = df_list_trn[ii]['Power'] - df_list_trn[ii]['Power_pred']
        df_list_trn[ii]['diffresid'] = df_list_trn[ii]['diffUBLB'] - df_list_trn[ii]['diff_pred']
        df_list_trn[ii]['dayofauc'] = (df_list_trn[ii].index - (auc_settl-1) - ii)//48

    #create scenarios from predictions (this should be used for both train and test data)
    intvalmatrixUB = np.zeros([no_settl,no_scen])
    intvalmatrixP = np.zeros([no_settl,no_scen])
    intvalmatrixdLB = np.zeros([no_settl,no_scen])
    quantiles = np.cumsum(np.array(distlist))[:-1]
    for i in range(no_settl):
        #this create scenarios from distributions --> later add a proper distribution here
        mux, stdx = sps.norm.fit(df_list_trn[i]['UBresid'])
        muy2, stdy2 = sps.norm.fit(df_list_trn[i]['diffresid'])
        muz, stdz = sps.norm.fit(df_list_trn[i]['Powerresid'])
        distx = sps.norm(loc=mux, scale=stdx)
        disty2 = sps.norm(loc=muy2, scale=stdy2)
        distz = sps.norm(loc=muz, scale=stdz)
        #print(quantiles)
        ppfsx = distx.ppf(quantiles)
        ppfsy2 = disty2.ppf(quantiles)
        ppfsz = distz.ppf(quantiles)  # boundaries
        ppfsox = np.append(np.append(-np.inf,ppfsx),np.inf)
        ppfsoy2 = np.append(np.append(-np.inf,ppfsy2),np.inf)
        ppfsoz = np.append(np.append(-np.inf,ppfsz),np.inf)
        ex1 = []
        ey2 = []
        ez1 = []
        for i2 in range (no_scen):
            ex1.append(distx.expect(lb = ppfsox[i2],ub = ppfsox[i2+1])/distlist[i2])
            ey2.append(disty2.expect(lb = ppfsoy2[i2],ub = ppfsoy2[i2+1])/distlist[i2])
            ez1.append(distz.expect(lb = ppfsoz[i2],ub = ppfsoz[i2+1])/distlist[i2])

        intvalmatrixUB[i] =  np.array(ex1)
        intvalmatrixP[i] = np.array(ez1)
        intvalmatrixdLB[i] = np.array(ey2)
    
    return intvalmatrixUB,intvalmatrixdLB,intvalmatrixP

#creates predictions for the whole test period
def create_forecasts(df_from,df_to,auc_settl,no_settl): #the predictions are created starting at auc_settl but the reference LB is from auc_settl - 1
    df_from['diffUBLB'] = df_from['UBc'] - df_from['LBc']
    df_to['diffUBLB'] = df_to['UBc'] - df_to['LBc']
    
    df_list_trn = [] #creates list of dataframes with a frame for each settlement in 14-23 (66)
    df_list_tst = [] #same but for training dataset
    for i in range(no_settl):
        dfx = xyfunc(df_from,i,auc_settl,train=True)
        dfy = xyfunc(df_to,i,auc_settl,train=False)
        df_list_trn.append(dfx)
        df_list_tst.append(dfy)

    #predict in sample
    model_list_UB = []
    model_list_dLB = []
    model_list_P = []
    for ii in range(len(df_list_trn)):
        model_list_UB.append(smf.ols(formula='UB_loc_c ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        model_list_dLB.append(smf.ols(formula='diffUBLB ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
        model_list_P.append(smf.ols(formula='Power ~ deter_dist + Temperature + Precipitation + BankHoliday + C(Weekday)', data=df_list_trn[ii]).fit())
    
    for ii in range(len(df_list_tst)):
        df_list_tst[ii]['UB_loc_c_pred'] = model_list_UB[ii].predict(df_list_tst[ii])
        df_list_tst[ii]['diff_pred'] = model_list_dLB[ii].predict(df_list_tst[ii])
        df_list_tst[ii]['Power_pred'] = model_list_P[ii].predict(df_list_tst[ii])
        df_list_tst[ii]['dayofauc'] = (df_list_tst[ii].index - (auc_settl-1) - ii)//48

    #get prediction for day of

    #if test/train
    predlistUB = []
    predlistdLB = []
    predlistP = []
    for day in range(256,365):
        UBpred = []
        dpred = []
        powerpred = []
        for i in range(len(df_list_tst)):
            UBpred.append(df_list_tst[i].loc[(df_list_tst[i]['dayofauc'] == (day-1)), 'UB_loc_c_pred'].iat[0] + df_list_tst[i].loc[(df_list_tst[i]['dayofauc'] == (day-1)), 'deter_dist'].iat[0])
            powerpred.append(df_list_tst[i].loc[(df_list_tst[i]['dayofauc'] == (day-1)), 'Power_pred'].iat[0])
            dpred.append(df_list_tst[i].loc[(df_list_tst[i]['dayofauc'] == (day-1)), 'diff_pred'].iat[0])
        predlistUB.append(UBpred)
        predlistdLB.append(dpred)
        predlistP.append(powerpred)
    forecastUB = np.array(predlistUB)
    forecastdLB = np.array(predlistdLB)
    forecastP = np.array(predlistP)
    return forecastUB,forecastdLB,forecastP
