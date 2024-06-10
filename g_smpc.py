#global imports
import pandas as pd
pd.set_option("mode.chained_assignment", None)

#local imports
import e_predmodel as epm
import f_stoch_opt as fso

#proposed SMPC algorithm
def run_conti_SMPC_nzb_AVg_wMILP_GUROB_noEVs(df_tr,df_te,distr_list_S1,distr_list_S2,final_settl,penaltyinput,risk_input,filename,eficaz=0.9,nPR = 0.000308, dPR = 0.001414, nrpr = 0.2,LBUB_diff_thresh=20):
    
    #get S1 scenario predictions once and for all [3][66][no_scen]
    s1_scen_pred = epm.create_preds_dist(df_tr,df_te,29,65,distr_list_S1)

    #get S2 scenario predictions once and for all (more complex) [48][3][18][no_scen]
    s2_scen_pred = [epm.create_preds_dist(df_tr,df_te,i,17,distr_list_S2) for i in range(48)]

    #get S1 point forecasts once and for all ([3][110][66])
    s1_forecasts = epm.create_forecasts(df_tr,df_te,28,66) #should be 28,66 --> need to change stuff in stoch_opt tho

    #get S2 point forecasts once and for all ([48][3][110][18?])
    s2_forecasts = [epm.create_forecasts(df_tr,df_te,i,18) for i in range(48)] #should be 18

    #set everything else up for the run
    lintNR = [0] * 46
    lintPR = [0] * 46
    sol_V2Gg = [0,0]
    sol_G2Vg = [0,0]
    #sol_V2G = []
    #sol_G2V = []
    sol_PR = [[0] * 46]
    sol_NR = [[0] * 46]
    sol_pen = []
    energy00 = ((df_te.loc[12239,'LBc'] + df_te.loc[12239,'UBc'])/2)
    energy0s = []
    energy0s.append(energy00)
    print(energy00)
    #actual run
    for settl in range(12240,final_settl):
        
        day_rn = ((settl-28)//48)+1-255
        print(day_rn)
        day_rn1 = ((settl-28)//48)+1
        print(day_rn1)

        #S12day
        day_rnS12 = settl//48
        print("day_rnS12: ", day_rnS12)

        #S2day

        #forecast day
        day_rn2 = (settl//48)-255
        print("day_rn2: ", day_rn2)
        
        energy0i = energy00 - df_te.loc[settl-1,'LBc']
        if (settl-28)%48 == 0:
            
            res_S1 = fso.optS1_nzb_AVg_MILP_gurob_noEVs(day_rnS12, distr_list_S1, df_te, [s1_forecasts[i][day_rn2][1:] for i in range(3)], s1_scen_pred,pen=penaltyinput, effy=eficaz,risk_avers=risk_input,energy0=energy0i,prePR=lintPR[:18],preNR=lintNR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],nightPR=nPR,dayPR=dPR,NRPRratio=nrpr,LBUB_diff_threshold=LBUB_diff_thresh)

            #save reserve commitments to intermediate list
            lintPR.extend(res_S1[4][0][-48:])
            lintNR.extend(res_S1[4][1][-48:])
        
            #append results to solution list
        
            #reserve
            sol_PR.append(res_S1[4][0])
            sol_NR.append(res_S1[4][1])

            #actual charging
            sol_V2Gg.append(res_S1[1])
            sol_G2Vg.append(res_S1[0])


            energy00 += (res_S1[0]*eficaz - res_S1[1]/eficaz) #vehicle side

            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        else:
            res_S2 = fso.optS2_nzb_AVg_wMILP_gurob_noEVs(day_rnS12,settl%48,df_te,distr_list_S2,[s2_forecasts[(settl%48)][i][day_rn2][1:] for i in range(3)], s2_scen_pred[settl%48],energy0=energy0i,pen=penaltyinput,efficiency=eficaz,preNR=lintNR[:18],prePR=lintPR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],LBUB_diff_threshold=LBUB_diff_thresh)
            #actual charging
            sol_V2Gg.append(res_S2[1])
            sol_G2Vg.append(res_S2[0])

            energy00 += (res_S2[0]*eficaz - res_S2[1]/eficaz) #vehicle side
            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        #debugging
        #print("Continuous NR list: ",lintNR)#print("Intake NR list: ", lintNR[-18:])
        #print("Continuous PR list: ",lintPR)#print("Intake PR list: ", lintPR[-18:])
        lintNR = lintNR[1:]
        lintPR = lintPR[1:]
        energy0s.append(energy00)

    #for the sake of completeness
    sol_V2Gg = sol_V2Gg[2:]
    sol_G2Vg = sol_G2Vg[2:]

        #only intermediary --> take out later
    sol_PR.append([0]*24)
    sol_NR.append([0]*24)

        #save results
    solPR_f = []
    solNR_f = []
    for i in sol_PR:
        solPR_f.extend(i[-48:])
        #print(i)
        #print(len(i))
    for i in sol_NR:
        solNR_f.extend(i[-48:])

    #traj_PR_f = [a - b*0.5*0.9 for a, b in zip(energy0s, solPR_f)]
    #traj_NR_f = [a + b*0.5*0.9 for a, b in zip(energy0s, solNR_f)]
    #sol_pen.extend([0]*46)
    #energy0s.extend([None]*46)
    
    solPR_f.insert(0,0)
    solNR_f.insert(0,0)
    sol_pen.insert(0,0)
    sol_G2Vg.insert(0,0)
    sol_V2Gg.insert(0,0)
    
    # dictionary of lists 
    print(len(sol_pen),len(solPR_f[:(final_settl-12239)]),len(solNR_f[:(final_settl-12239)]),len(energy0s),len(sol_G2Vg),len(sol_V2Gg))
    dict = {'Penalties': sol_pen, 'PR': solPR_f[:(final_settl-12239)], 'NR': solNR_f[:(final_settl-12239)], 'energy_traj': energy0s,'G2Vg':sol_G2Vg,'V2Gg':sol_V2Gg} 
    dfcsv = pd.DataFrame(dict)
    dfcsv['LBc'] = df_te.loc[12240:final_settl,'LBc'].values
    dfcsv['UBc'] = df_te.loc[12240:final_settl,'UBc'].values
    dfcsv['Power'] = df_te.loc[12240:final_settl,'Power'].values
    dfcsv['Price'] = df_te.loc[12240:final_settl,'Price'].values
    dfcsv.to_csv("../noEV_results/" + str(filename) + ".csv")

    #s11 = fso.optS1(256, distr_list_S1, df_te, [s1_forecasts[i][0] for i in range(3)], s1_scen_pred)
    #s12 = fso.optS2(256,28,df_te,distr_list_S2,[s2_forecasts[28][i][0] for i in range(3)], s2_scen_pred[28])

    return dfcsv

#deterministic predictions benchmark
def run_conti_SMPC_nzb_AVg_wMILP_GUROB_noEVs_BM1a(df_tr,df_te,distr_list_S1,distr_list_S2,final_settl,penaltyinput,risk_input,filename,eficaz=0.9,nPR = 0.000308, dPR = 0.001414, nrpr = 0.2,LBUB_diff_thresh=20):

    #get S1 point forecasts once and for all ([3][110][66])
    s1_forecasts = epm.create_forecasts(df_tr,df_te,28,66) #should be 28,66 --> need to change stuff in stoch_opt tho

    #get S2 point forecasts once and for all ([48][3][110][18?])
    s2_forecasts = [epm.create_forecasts(df_tr,df_te,i,18) for i in range(48)] #should be 18

    #set everything else up for the run
    lintNR = [0] * 46
    lintPR = [0] * 46
    sol_V2Gg = [0,0]
    sol_G2Vg = [0,0]
    #sol_V2G = []
    #sol_G2V = []
    sol_PR = [[0] * 46]
    sol_NR = [[0] * 46]
    sol_pen = []
    energy00 = ((df_te.loc[12239,'LBc'] + df_te.loc[12239,'UBc'])/2)
    energy0s = []
    energy0s.append(energy00)
    print(energy00)
    #actual run
    wvi = 60
    for settl in range(12240,final_settl):
        
        day_rn = ((settl-28)//48)+1-255
        print(day_rn)
        day_rn1 = ((settl-28)//48)+1
        print(day_rn1)

        #S12day
        day_rnS12 = settl//48
        print("day_rnS12: ", day_rnS12)

        #S2day

        #forecast day
        day_rn2 = (settl//48)-255
        print("day_rn2: ", day_rn2)
        
        energy0i = energy00 - df_te.loc[settl-1,'LBc']
        if (settl-28)%48 == 0:
            
            res_S1 = fso.optS1_nzb_AVg_MILP_gurob_noEVs_BM1a(day_rnS12, df_te, [s1_forecasts[i][day_rn2][1:] for i in range(3)], pen=penaltyinput, effy=eficaz,risk_avers=risk_input,energy0=energy0i,prePR=lintPR[:18],preNR=lintNR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],nightPR=nPR,dayPR=dPR,NRPRratio=nrpr,LBUB_diff_threshold=LBUB_diff_thresh)#,willvarin=wvi-40)
            #wvi = res_S1[5]
            #save reserve commitments to intermediate list
            lintPR.extend(res_S1[4][0][-48:])
            lintNR.extend(res_S1[4][1][-48:])
        
            #append results to solution list
        
            #reserve
            sol_PR.append(res_S1[4][0])
            sol_NR.append(res_S1[4][1])

            #actual charging
            sol_V2Gg.append(res_S1[1])
            sol_G2Vg.append(res_S1[0])


            energy00 += (res_S1[0]*eficaz - res_S1[1]/eficaz) #vehicle side

            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        else:
            res_S2 = fso.optS2_nzb_AVg_wMILP_gurob_noEVs_BM1a(day_rnS12,settl%48,df_te,[s2_forecasts[(settl%48)][i][day_rn2][1:] for i in range(3)],energy0=energy0i,pen=penaltyinput,efficiency=eficaz,preNR=lintNR[:18],prePR=lintPR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],LBUB_diff_threshold=LBUB_diff_thresh)#,willvarin=wvi-40)
            #wvi = res_S2[5]
            #actual charging
            sol_V2Gg.append(res_S2[1])
            sol_G2Vg.append(res_S2[0])

            energy00 += (res_S2[0]*eficaz - res_S2[1]/eficaz) #vehicle side
            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        #debugging
        #print("Continuous NR list: ",lintNR)#print("Intake NR list: ", lintNR[-18:])
        #print("Continuous PR list: ",lintPR)#print("Intake PR list: ", lintPR[-18:])
        lintNR = lintNR[1:]
        lintPR = lintPR[1:]
        energy0s.append(energy00)

    #for the sake of completeness
    sol_V2Gg = sol_V2Gg[2:]
    sol_G2Vg = sol_G2Vg[2:]

        #only intermediary --> take out later
    sol_PR.append([0]*24)
    sol_NR.append([0]*24)

        #save results
    solPR_f = []
    solNR_f = []
    for i in sol_PR:
        solPR_f.extend(i[-48:])
        #print(i)
        #print(len(i))
    for i in sol_NR:
        solNR_f.extend(i[-48:])

    #traj_PR_f = [a - b*0.5*0.9 for a, b in zip(energy0s, solPR_f)]
    #traj_NR_f = [a + b*0.5*0.9 for a, b in zip(energy0s, solNR_f)]
    #sol_pen.extend([0]*46)
    #energy0s.extend([None]*46)
    
    solPR_f.insert(0,0)
    solNR_f.insert(0,0)
    sol_pen.insert(0,0)
    sol_G2Vg.insert(0,0)
    sol_V2Gg.insert(0,0)
    
    # dictionary of lists 
    print(len(sol_pen),len(solPR_f[:(final_settl-12239)]),len(solNR_f[:(final_settl-12239)]),len(energy0s),len(sol_G2Vg),len(sol_V2Gg))
    dict = {'Penalties': sol_pen, 'PR': solPR_f[:(final_settl-12239)], 'NR': solNR_f[:(final_settl-12239)], 'energy_traj': energy0s,'G2Vg':sol_G2Vg,'V2Gg':sol_V2Gg} 
    dfcsv = pd.DataFrame(dict)
    dfcsv['LBc'] = df_te.loc[12240:final_settl,'LBc'].values
    dfcsv['UBc'] = df_te.loc[12240:final_settl,'UBc'].values
    dfcsv['Power'] = df_te.loc[12240:final_settl,'Power'].values
    dfcsv['Price'] = df_te.loc[12240:final_settl,'Price'].values
    
    #dfcsv.to_csv("C:/Users/jt3022/OneDrive - Imperial College London/Output/Projects/2. EV SMPC/4. Results/fP7/" + str(filename) + ".csv")

    dfcsv.to_csv("../noEV_results/detpred1a" + str(filename) + ".csv")
    #s11 = fso.optS1(256, distr_list_S1, df_te, [s1_forecasts[i][0] for i in range(3)], s1_scen_pred)
    #s12 = fso.optS2(256,28,df_te,distr_list_S2,[s2_forecasts[28][i][0] for i in range(3)], s2_scen_pred[28])

    return dfcsv

#perfect foresight benchmark
def run_conti_SMPC_nzb_AVg_wMILP_GUROB_noEVs_BM1b(df_tr,df_te,distr_list_S1,distr_list_S2,final_settl,penaltyinput,risk_input,filename,eficaz=0.9,nPR = 0.000308, dPR = 0.001414, nrpr = 0.2,LBUB_diff_thresh=20):
    
    #set everything else up for the run
    lintNR = [0] * 46
    lintPR = [0] * 46
    sol_V2Gg = [0,0]
    sol_G2Vg = [0,0]
    #sol_V2G = []
    #sol_G2V = []
    sol_PR = [[0] * 46]
    sol_NR = [[0] * 46]
    sol_pen = []
    energy00 = ((df_te.loc[12239,'LBc'] + df_te.loc[12239,'UBc'])/2)
    energy0s = []
    energy0s.append(energy00)
    print(energy00)
    #actual run
    wvi = 60
    for settl in range(12240,final_settl):
        
        day_rn = ((settl-28)//48)+1-255
        print(day_rn)
        day_rn1 = ((settl-28)//48)+1
        print(day_rn1)

        #S12day
        day_rnS12 = settl//48
        print("day_rnS12: ", day_rnS12)

        #S2day

        #forecast day
        day_rn2 = (settl//48)-255
        print("day_rn2: ", day_rn2)
        
        energy0i = energy00 - df_te.loc[settl-1,'LBc']
        if (settl-28)%48 == 0:
            
            res_S1 = fso.optS1_nzb_AVg_MILP_gurob_noEVs_BM1b(day_rnS12, df_te, pen=penaltyinput, effy=eficaz,risk_avers=risk_input,energy0=energy0i,prePR=lintPR[:18],preNR=lintNR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],nightPR=nPR,dayPR=dPR,NRPRratio=nrpr,LBUB_diff_threshold=LBUB_diff_thresh)#,willvarin=wvi-40)
            #wvi = res_S1[5]
            #save reserve commitments to intermediate list
            lintPR.extend(res_S1[4][0][-48:])
            lintNR.extend(res_S1[4][1][-48:])
        
            #append results to solution list
        
            #reserve
            sol_PR.append(res_S1[4][0])
            sol_NR.append(res_S1[4][1])

            #actual charging
            sol_V2Gg.append(res_S1[1])
            sol_G2Vg.append(res_S1[0])


            energy00 += (res_S1[0]*eficaz - res_S1[1]/eficaz) #vehicle side

            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        else:
            res_S2 = fso.optS2_nzb_AVg_wMILP_gurob_noEVs_BM1b(day_rnS12,settl%48,df_te,energy0=energy0i,pen=penaltyinput,efficiency=eficaz,preNR=lintNR[:18],prePR=lintPR[:18],energy_prev_G2V=sol_G2Vg[-2:],energy_prev_V2G=sol_V2Gg[-2:],LBUB_diff_threshold=LBUB_diff_thresh)#,willvarin=wvi-40)
            #wvi = res_S2[5]
            #actual charging
            sol_V2Gg.append(res_S2[1])
            sol_G2Vg.append(res_S2[0])

            energy00 += (res_S2[0]*eficaz - res_S2[1]/eficaz) #vehicle side
            sol_pen.append(res_S2[2].loc[(0,0),'Pen'])
        #debugging
        #print("Continuous NR list: ",lintNR)#print("Intake NR list: ", lintNR[-18:])
        #print("Continuous PR list: ",lintPR)#print("Intake PR list: ", lintPR[-18:])
        lintNR = lintNR[1:]
        lintPR = lintPR[1:]
        energy0s.append(energy00)

    #for the sake of completeness
    sol_V2Gg = sol_V2Gg[2:]
    sol_G2Vg = sol_G2Vg[2:]

        #only intermediary --> take out later
    sol_PR.append([0]*24)
    sol_NR.append([0]*24)

        #save results
    solPR_f = []
    solNR_f = []
    for i in sol_PR:
        solPR_f.extend(i[-48:])
        #print(i)
        #print(len(i))
    for i in sol_NR:
        solNR_f.extend(i[-48:])

    #traj_PR_f = [a - b*0.5*0.9 for a, b in zip(energy0s, solPR_f)]
    #traj_NR_f = [a + b*0.5*0.9 for a, b in zip(energy0s, solNR_f)]
    #sol_pen.extend([0]*46)
    #energy0s.extend([None]*46)
    
    solPR_f.insert(0,0)
    solNR_f.insert(0,0)
    sol_pen.insert(0,0)
    sol_G2Vg.insert(0,0)
    sol_V2Gg.insert(0,0)
    
    # dictionary of lists 
    print(len(sol_pen),len(solPR_f[:(final_settl-12239)]),len(solNR_f[:(final_settl-12239)]),len(energy0s),len(sol_G2Vg),len(sol_V2Gg))
    dict = {'Penalties': sol_pen, 'PR': solPR_f[:(final_settl-12239)], 'NR': solNR_f[:(final_settl-12239)], 'energy_traj': energy0s,'G2Vg':sol_G2Vg,'V2Gg':sol_V2Gg} 
    dfcsv = pd.DataFrame(dict)
    dfcsv['LBc'] = df_te.loc[12240:final_settl,'LBc'].values
    dfcsv['UBc'] = df_te.loc[12240:final_settl,'UBc'].values
    dfcsv['Power'] = df_te.loc[12240:final_settl,'Power'].values
    dfcsv['Price'] = df_te.loc[12240:final_settl,'Price'].values
    
    #dfcsv.to_csv("C:/Users/jt3022/OneDrive - Imperial College London/Output/Projects/2. EV SMPC/4. Results/fP7/" + str(filename) + ".csv")

    dfcsv.to_csv("../noEV_results/perffore1b" + str(filename) + ".csv")
    #s11 = fso.optS1(256, distr_list_S1, df_te, [s1_forecasts[i][0] for i in range(3)], s1_scen_pred)
    #s12 = fso.optS2(256,28,df_te,distr_list_S2,[s2_forecasts[28][i][0] for i in range(3)], s2_scen_pred[28])

    return dfcsv