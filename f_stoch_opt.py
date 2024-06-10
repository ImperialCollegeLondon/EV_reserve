#global imports
import pulp
import pandas as pd
pd.set_option("mode.chained_assignment", None)
import numpy as np
import time

#stage 1 optimisation for proposed SMPC algorithm
def optS1_nzb_AVg_MILP_gurob_noEVs(day, problist, dfout, forecast, dist_mat,
        pen = 0.050223125,effy = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        risk_avers = 0.5,cvar_alph=0.1, activation_maxshare = 0.9, nightPR = 0.000437595, dayPR = 0.002008925, NRPRratio = 0.3,bigM = 1000000,bigM2 = 1000000,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)

        #create boundary matrix -> rewrite
        scenUBi =  np.array(forecast[0])[:, np.newaxis] + dist_mat[0] 
        scenLBi = np.array(forecast[1])[:, np.newaxis] + dist_mat[1]
        scenPi = np.array(forecast[2])[:, np.newaxis] + dist_mat[2]

        #for cases with low numbers of vehicles
        #scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        

        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = len(forecast[0]) + 1
        no_scen = len(problist)
        lamb = 1 - risk_avers
        #print(tot_time_len,no_scen,lamb)

        #insert not replace
        scenUB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+27,'LBc'],repeats=no_scen),scenUBi))
        scenLB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+28,'LBc'],repeats=no_scen),scenLBi)) #X#X#X#tick
        scenP = np.vstack((np.repeat(dfout.loc[day*48+28,'Power'],repeats=no_scen),scenPi))

        #import wholesale prices
        pricelist = dfout.loc[day*48+28:day*48+28+tot_time_len,'Price'].values.tolist()

        #list with reserve prices --> optimise
        reservepricelist = []
        for i in range(18):
                reservepricelist.append(dayPR)
        for i in range(16):
                reservepricelist.append(nightPR)
        for i in range(32):
                reservepricelist.append(dayPR)

        #initiate model
        modelx = pulp.LpProblem("Mean-CVaR",pulp.LpMaximize)

        #create scenario and time index
        sc_idx = [sc for sc in range(no_scen**3)]
        prob_idx = [problist[i]*problist[j]*problist[k] for i in range(no_scen) for j in range(no_scen) for k in range(no_scen)]
        t_idx = [t for t in range(tot_time_len)]

        #create two main variables
        e_X = pulp.LpVariable("Mean",   lowBound=None, cat='Continuous')
        cVaR = pulp.LpVariable("CVaR",   lowBound=None, cat='Continuous')
        binPR = pulp.LpVariable.dicts("bin_PR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,
        binNR = pulp.LpVariable.dicts("bin_NR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,

        #create other technical variables
        eNergyV2G = pulp.LpVariable.dicts("eV2G", ( (sc,t) for sc in sc_idx for t in t_idx),lowBound=0,cat='Continuous')
        eNergyG2V = pulp.LpVariable.dicts("eG2V", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')
        pR = pulp.LpVariable.dicts("ePR", ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        nR = pulp.LpVariable.dicts("eNR",  ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        ePen = pulp.LpVariable.dicts("ePen", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')

        #create other statistical variables

        #loss deviation
        VarDev = pulp.LpVariable.dicts("VarDev", ( (sc) for sc in sc_idx ), lowBound=0, cat='Continuous') #this has a lower bound of zero to ensure that only losses that are greater than VaR are included in the CVaR

        #value at risk
        VaR = pulp.LpVariable("VaR",   lowBound=None, cat='Continuous')

        #define objective function
        modelx += lamb*e_X - (1-lamb)*cVaR

        #constraints

        #all variables are defined on the grid side which means that they have to be adjusted within the constraints
        #pmax or "rated power" is assumed to refer to the power output (either on grid or vehicle side, depending on whether its V2G or G2V)

        inv_eff = 1/effy #because pulp does not like division operators

        #specify eX
        modelx += pulp.lpSum([(
        pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000)  * (-1) for t in t_idx]) + #/ effy
        pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
        pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
        pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
        pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
        pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick

        ) *prob_idx[sc] for sc in sc_idx]) == e_X

        #specify CVaR through VardDeV and VaR
        for sc in sc_idx:
                modelx += (-1)*(
                        pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000) * (-1) for t in t_idx]) + #/ effy
                        pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
                        pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
                        pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
                        pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
                        pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
                ) - VaR <= VarDev[sc]

        modelx += VaR + (1/cvar_alph)*pulp.lpSum([ VarDev[sc]*prob_idx[sc] for sc in sc_idx]) == cVaR

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #energy and power constraints
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(tot_time_len):
                                        
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t1]*effy - eNergyV2G[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + energy0 <= scenUB[i10][i7] #i10 or i10+1 --> i10+1 because of how range works (range(0) would give us nothing)
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t2]*effy - eNergyV2G[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        modelx += eNergyG2V[scen_id,i10] + eNergyV2G[scen_id,i10]*inv_eff <= (scenP[i10][i9]/2) #needs to be other way around (I think) --> compare to how it was when all variables were vehicle side
                                        
                                        if i10 > 1:

                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)])) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10] * 0.5 * activation_maxshare * effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10] * 0.5 * activation_maxshare / effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10]+ pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10]+ pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 1:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM)#i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 0:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen[scen_id,i10] * inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        else:
                                               raise ValueError("incontinuous time series")
                                        
        #first reserve commitments are given
        for i11 in range(len(prePR)):
                modelx += nR[i11] == preNR[i11]
                modelx += pR[i11] == prePR[i11]
                if preNR[i11] == 0:
                        modelx += binNR[i11] == 1
                else:
                        modelx += binNR[i11] == 0
                if prePR[i11] == 0:
                        modelx += binPR[i11] == 1
                else:
                        modelx += binPR[i11] == 0

        #other reserve commitments have to be same for service windows (4 settlements)
        for i12 in range(tot_time_len - len(prePR)):
                if i12 % 4 == 0:
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+1]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+1]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+2]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+2]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+3]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+3]

                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+1]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+1]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+2]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+2]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+3]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+3]

        #ensure first period is the same in all scenarios --> later becomes real charging
        for ij in sc_idx: #all boundary scenarios
                        if ij != 0:
                                modelx += eNergyG2V[0,0] == eNergyG2V[ij,0]
                                modelx += eNergyV2G[0,0] == eNergyV2G[ij,0]

        #binary constraints
        for i13 in range(tot_time_len - len(prePR)):
                modelx += bigM2*(1-binNR[i13+len(prePR)]) >= nR[i13+len(prePR)]
                modelx += bigM2*(1-binPR[i13+len(prePR)]) >= pR[i13+len(prePR)]

        solver = pulp.GUROBI_CMD(options=[("MIPgap", 0.00004), ("TimeLimit", "1500")])
        #modelx.solve(pulp.getSolver('GUROBI_CMD'))
        modelx.solve(solver)

        modelstatus = pulp.LpStatus[modelx.status]
        print(modelstatus)

        #process results
        vV2G = dict()
        vG2V = dict()
        vNR = dict()
        vPR = dict()
        vPen = dict()

        for variable in modelx.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eNR" in variable.name:
                        vNR[int(variable.name[4:])] = variable.varValue
                elif "ePR" in variable.name:
                        vPR[int(variable.name[4:])] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "VarDev" in variable.name:
                        #print(variable.name,variable.varValue)
                        pass
                else:
                        print(variable.name,variable.varValue)
        
        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)

        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*effy - df_temp['V2G']/effy
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)

        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick --> simply name row D and add another row with LB (=UB-D)
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("C:/Users/jt3022/OneDrive - Imperial College London/Output/Projects/2. EV SMPC/4. Results/noEVs/plans/day"+str(day+1)+"settl"+timestamp+"_S1.csv")


        vPRl = [vPR[i] for i in range(66)]
        vNRl = [vNR[i] for i in range(66)]
        temp_dict = {'PR': vPRl, 'NR': vNRl}
        df_temp2 = pd.DataFrame(temp_dict)
        df_temp2.to_csv("C:/Users/jt3022/OneDrive - Imperial College London/Output/Projects/2. EV SMPC/4. Results/noEVs/plans/day"+str(day+1)+"settl"+timestamp+"_S1res.csv")

        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,[vPRl,vNRl]

#stage 2 optimisation for proposed SMPC algorithm
def optS2_nzb_AVg_wMILP_gurob_noEVs(day,cur_settl,dfout,problist,forecast,dist_mat,no_settl_pl = 18,pen = 0.050223125,efficiency = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        activation_maxshare = 0.9,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)
        #create boundary matrix -> rewrite
        scenUBi = np.array(forecast[0])[:, np.newaxis] + dist_mat[0]
        scenLBi = np.array(forecast[1])[:, np.newaxis] + dist_mat[1]
        scenPi = np.array(forecast[2])[:, np.newaxis] + dist_mat[2]
        
        #for cases with few vehicles
        #scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        
        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = len(forecast[0]) + 1
        no_scen = len(problist)

        #insert (not replace) first row of matrices with (no_scen*) real boundaries
        scenUB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl-1,'LBc'],repeats=no_scen),scenUBi))
        scenLB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl,'LBc'],repeats=no_scen),scenLBi)) #X#X#X# tick
        scenP = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'Power'],repeats=no_scen),scenPi))

        print(scenUB)
        
        print(scenLB)

        print(scenP)

        #import wholesale prices
        pricelist = dfout.loc[day*48+cur_settl:day*48+cur_settl+no_settl_pl-1,'Price'].values.tolist()
        
        #init model
        model_simp_S2 = pulp.LpProblem("Mean_S2", pulp.LpMaximize)

        #Define technical variables
        eNergyV2G_S2 = pulp.LpVariable.dicts("eV2G_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl)),lowBound=0,cat='Continuous')
        eNergyG2V_S2 = pulp.LpVariable.dicts("eG2V_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')
        ePen_S2 = pulp.LpVariable.dicts("ePen_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')

        #Model Objective
        #model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[i5,j5] * (pricelist[j5]/1000) / efficiency * (-1)  for i5 in range(no_scen**3) for j5 in range(no_settl_pl)] )/(no_scen**3) + 
        #      pulp.lpSum([(eNergyV2G_S2[i4,j4] * (pricelist[j4]/1000) * efficiency) for i4 in range(no_scen**3) for j4 in range(no_settl_pl)] )/(no_scen**3)   +
        #      pulp.lpSum([ePen_S2[i6,j6] * pen * (-1) for i6 in range(no_scen**3) for j6 in range(no_settl_pl)] )/(no_scen**3) +
        #      (sum(pricelist)/len(pricelist)) * pulp.lpSum([pulp.lpSum([eNergyG2V_S2[j9,j10] - eNergyV2G_S2[j9,j10] for j10 in range(no_settl_pl)]) - scenLB[no_settl_pl-1][j9] for j9 in range(no_scen)]))#reward for energy added at the end to make rollover make more sense
        
        inv_eff = 1/efficiency

        model_simp_S2 += pulp.lpSum([
                pulp.lpSum([eNergyG2V_S2[sc,t] * (pricelist[t]/1000) * (-1) for t in range(no_settl_pl)]) + #/ efficiency
                pulp.lpSum([eNergyV2G_S2[sc,t] * (pricelist[t]/1000) for t in range(no_settl_pl)]) + #* efficiency #try 999
                pulp.lpSum([ePen_S2[sc,t] * pen * (-1) for t in range(no_settl_pl)]) +
                pulp.lpSum([pulp.lpSum([eNergyG2V_S2[sc,t] * efficiency - eNergyV2G_S2[sc,t] * inv_eff for t in range(no_settl_pl)]) - (scenUB[no_settl_pl-1][(sc//(no_scen**2))] - scenLB[no_settl_pl-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
        for sc in range(no_scen**3)])/(no_scen**3)

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #constraints #make quicker by using numpy array that I use later?
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(no_settl_pl):
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[scen_id,t1]*efficiency - 
                                                                eNergyV2G_S2[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + #i10 or i10+1
                                                                energy0 <= 
                                                                scenUB[i10][i7])
                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t2]*efficiency - eNergyV2G_S2[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= scenUB[i10][i7] - scenLB[i10][i8] #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        model_simp_S2 += eNergyG2V_S2[scen_id,i10] + eNergyV2G_S2[scen_id,i10]*inv_eff <= scenP[i10][i9]/2
                                        
                                        if preNR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10]+ pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                else:
                                                        raise ValueError("incontinuous time series")
                                                
                                        if prePR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10]+ pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                else:
                                                        raise ValueError("incontinuous time series")

        #other reserve commitments have to be same for service windows (4 settlements)
        for ij in range(no_scen**3): #all boundary scenarios
                        if ij != 0:
                                model_simp_S2 += eNergyG2V_S2[0,0] == eNergyG2V_S2[ij,0]
                                model_simp_S2 += eNergyV2G_S2[0,0] == eNergyV2G_S2[ij,0]
                                #could do the same for penalty but it shouldn't make a difference --> shouldn't do

        #print(model_simp_S2)

        #solve
        pulp.LpSolverDefault.msg = 1
        solver = pulp.GUROBI_CMD(gapRel = 0.00004, timeLimit = 1500)
        #model_simp_S2.solve(pulp.getSolver('GUROBI_CMD'))
        model_simp_S2.solve(solver)

        modelstatus = pulp.LpStatus[model_simp_S2.status]
        
        print(modelstatus)

        #Process results
        vV2G = dict()
        vG2V = dict()
        vPen = dict()

        for variable in model_simp_S2.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "VarDev" in variable.name:
                        pass
                else:
                        #print("Other Variable: ",variable.name)
                        pass

        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)
        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*efficiency - df_temp['V2G']/efficiency
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)
        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))
        df_temp['Price'] = pricelist * no_scen**3
        df_temp['NR'] = [prePR[i10] for i10 in range(no_settl_pl)] * no_scen**3
        df_temp['PR'] = [preNR[i10] for i10 in range(no_settl_pl)] * no_scen**3

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("../stage12results/day"+str(day+1)+"settl"+str(cur_settl)+timestamp+"_S2.csv")
        #vPRl = [k[1] for k in list(sorted(vPR.items()))]
        #vNRl = [k[1] for k in list(sorted(vNR.items()))]
        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,pricelist#,scenUB,scenLB,scenP#,[vPRl,vNRl] #X#X#X# not sure

#stage 1 optimisation for determinstic predictions benchmark
def optS1_nzb_AVg_MILP_gurob_noEVs_BM1a(day, dfout, forecast,
        problist =[1],pen = 0.050223125,effy = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        risk_avers = 0.5,cvar_alph=0.1, activation_maxshare = 0.9, nightPR = 0.000437595, dayPR = 0.002008925, NRPRratio = 0.3,bigM = 1000000,bigM2 = 1000000,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)

        #no need for adaptation
        
        #create boundary matrix -> rewrite
        scenUBi =  np.array(forecast[0])[:, np.newaxis]
        scenLBi = np.array(forecast[1])[:, np.newaxis]
        scenPi = np.array(forecast[2])[:, np.newaxis]

        #for cases with low numbers of vehicles
        scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        

        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = len(forecast[0]) + 1
        no_scen = 1
        lamb = 1 - risk_avers
        #print(tot_time_len,no_scen,lamb)

        #insert not replace
        scenUB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+27,'LBc'],repeats=no_scen),scenUBi))
        scenLB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+28,'LBc'],repeats=no_scen),scenLBi)) #X#X#X#tick
        scenP = np.vstack((np.repeat(dfout.loc[day*48+28,'Power'],repeats=no_scen),scenPi))

        print(scenUB)
        
        print(scenLB)

        print(scenP)


        #import wholesale prices
        pricelist = dfout.loc[day*48+28:day*48+28+tot_time_len,'Price'].values.tolist()

        #list with reserve prices --> optimise
        reservepricelist = []
        for i in range(18):
                reservepricelist.append(dayPR)
        for i in range(16):
                reservepricelist.append(nightPR)
        for i in range(32):
                reservepricelist.append(dayPR)

        #initiate model
        modelx = pulp.LpProblem("Mean-CVaR",pulp.LpMaximize)

        #create scenario and time index
        sc_idx = [sc for sc in range(no_scen**3)]
        prob_idx = [problist[i]*problist[j]*problist[k] for i in range(no_scen) for j in range(no_scen) for k in range(no_scen)]
        t_idx = [t for t in range(tot_time_len)]

        #create two main variables
        e_X = pulp.LpVariable("Mean",   lowBound=None, cat='Continuous')
        #cVaR = pulp.LpVariable("CVaR",   lowBound=None, cat='Continuous')
        binPR = pulp.LpVariable.dicts("bin_PR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,
        binNR = pulp.LpVariable.dicts("bin_NR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,

        #create other technical variables
        eNergyV2G = pulp.LpVariable.dicts("eV2G", ( (sc,t) for sc in sc_idx for t in t_idx),lowBound=0,cat='Continuous')
        eNergyG2V = pulp.LpVariable.dicts("eG2V", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')
        pR = pulp.LpVariable.dicts("ePR", ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        nR = pulp.LpVariable.dicts("eNR",  ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        ePen = pulp.LpVariable.dicts("ePen", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')

        #create other statistical variables

        #loss deviation
        #VarDev = pulp.LpVariable.dicts("VarDev", ( (sc) for sc in sc_idx ), lowBound=0, cat='Continuous') #this has a lower bound of zero to ensure that only losses that are greater than VaR are included in the CVaR

        #value at risk
        #VaR = pulp.LpVariable("VaR",   lowBound=None, cat='Continuous')

        #define objective function
        modelx += e_X #lamb*e_X - (1-lamb)*cVaR

        #constraints

        #all variables are defined on the grid side which means that they have to be adjusted within the constraints
        #pmax or "rated power" is assumed to refer to the power output (either on grid or vehicle side, depending on whether its V2G or G2V)

        inv_eff = 1/effy #because pulp does not like division operators

        #specify eX
        modelx += pulp.lpSum([(
        pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000)  * (-1) for t in t_idx]) + #/ effy
        pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
        pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
        pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
        pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
        pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick

        ) *prob_idx[sc] for sc in sc_idx]) == e_X

        #specify CVaR through VardDeV and VaR
        #for sc in sc_idx:
        #        modelx += (-1)*(
        #                pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000) * (-1) for t in t_idx]) + #/ effy
        #                pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
        #                pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
        #                pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
        #                pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
        #                pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
        #        ) - VaR <= VarDev[sc]

        #modelx += VaR + (1/cvar_alph)*pulp.lpSum([ VarDev[sc]*prob_idx[sc] for sc in sc_idx]) == cVaR

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #energy and power constraints
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(tot_time_len):
                                        
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t1]*effy - eNergyV2G[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + energy0 <= scenUB[i10][i7] #i10 or i10+1 --> i10+1 because of how range works (range(0) would give us nothing)
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t2]*effy - eNergyV2G[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        modelx += eNergyG2V[scen_id,i10] + eNergyV2G[scen_id,i10]*inv_eff <= (scenP[i10][i9]/2) #needs to be other way around (I think) --> compare to how it was when all variables were vehicle side
                                        
                                        if i10 > 1:

                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)])) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10] * 0.5 * activation_maxshare * effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10] * 0.5 * activation_maxshare / effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10]+ pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10]+ pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 1:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM)#i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 0:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen[scen_id,i10] * inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        else:
                                               raise ValueError("incontinuous time series")
                                        
        #first reserve commitments are given
        for i11 in range(len(prePR)):
                modelx += nR[i11] == preNR[i11]
                modelx += pR[i11] == prePR[i11]
                if preNR[i11] == 0:
                        modelx += binNR[i11] == 1
                else:
                        modelx += binNR[i11] == 0
                if prePR[i11] == 0:
                        modelx += binPR[i11] == 1
                else:
                        modelx += binPR[i11] == 0

        #other reserve commitments have to be same for service windows (4 settlements)
        for i12 in range(tot_time_len - len(prePR)):
                if i12 % 4 == 0:
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+1]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+1]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+2]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+2]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+3]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+3]

                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+1]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+1]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+2]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+2]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+3]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+3]

        #ensure first period is the same in all scenarios --> later becomes real charging
        for ij in sc_idx: #all boundary scenarios
                        if ij != 0:
                                modelx += eNergyG2V[0,0] == eNergyG2V[ij,0]
                                modelx += eNergyV2G[0,0] == eNergyV2G[ij,0]

        #binary constraints
        for i13 in range(tot_time_len - len(prePR)):
                modelx += bigM2*(1-binNR[i13+len(prePR)]) >= nR[i13+len(prePR)]
                modelx += bigM2*(1-binPR[i13+len(prePR)]) >= pR[i13+len(prePR)]

        solver = pulp.GUROBI_CMD(options=[("MIPgap", 0.00004), ("TimeLimit", "1500")])
        #modelx.solve(pulp.getSolver('GUROBI_CMD'))
        modelx.solve(solver)

        modelstatus = pulp.LpStatus[modelx.status]
        print(modelstatus)

        #process results
        vV2G = dict()
        vG2V = dict()
        vNR = dict()
        vPR = dict()
        vPen = dict()

        for variable in modelx.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eNR" in variable.name:
                        vNR[int(variable.name[4:])] = variable.varValue
                elif "ePR" in variable.name:
                        vPR[int(variable.name[4:])] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                #elif "VarDev" in variable.name:
                        #print(variable.name,variable.varValue)
                        #pass
                else:
                        print(variable.name,variable.varValue)
        
        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)

        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*effy - df_temp['V2G']/effy
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)

        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick --> simply name row D and add another row with LB (=UB-D)
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("../stage12results/day"+str(day+1)+"settl"+timestamp+"_S1.csv")


        vPRl = [vPR[i] for i in range(66)]
        vNRl = [vNR[i] for i in range(66)]
        temp_dict = {'PR': vPRl, 'NR': vNRl}
        df_temp2 = pd.DataFrame(temp_dict)
        df_temp2.to_csv("../stage12results/day"+str(day+1)+"settl"+timestamp+"_S1res.csv")

        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,[vPRl,vNRl]

#stage 2 optimisation for determinstic predictions benchmark
def optS2_nzb_AVg_wMILP_gurob_noEVs_BM1a(day,cur_settl,dfout,forecast,problist =[1],no_settl_pl = 18,pen = 0.050223125,efficiency = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        activation_maxshare = 0.9,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)
        #create boundary matrix -> rewrite
        scenUBi = np.array(forecast[0])[:, np.newaxis]
        scenLBi = np.array(forecast[1])[:, np.newaxis]
        scenPi = np.array(forecast[2])[:, np.newaxis]
        
        #for cases with few vehicles
        #scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        
        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = len(forecast[0]) + 1
        no_scen = len(problist)

        #insert (not replace) first row of matrices with (no_scen*) real boundaries
        scenUB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl-1,'LBc'],repeats=no_scen),scenUBi))
        scenLB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl,'LBc'],repeats=no_scen),scenLBi)) #X#X#X# tick
        scenP = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'Power'],repeats=no_scen),scenPi))

        print(scenUB)
        
        print(scenLB)

        print(scenP)

        #import wholesale prices
        pricelist = dfout.loc[day*48+cur_settl:day*48+cur_settl+no_settl_pl-1,'Price'].values.tolist()
        
        #init model
        model_simp_S2 = pulp.LpProblem("Mean_S2", pulp.LpMaximize)

        #Define technical variables
        eNergyV2G_S2 = pulp.LpVariable.dicts("eV2G_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl)),lowBound=0,cat='Continuous')
        eNergyG2V_S2 = pulp.LpVariable.dicts("eG2V_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')
        ePen_S2 = pulp.LpVariable.dicts("ePen_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')

        #Model Objective
        #model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[i5,j5] * (pricelist[j5]/1000) / efficiency * (-1)  for i5 in range(no_scen**3) for j5 in range(no_settl_pl)] )/(no_scen**3) + 
        #      pulp.lpSum([(eNergyV2G_S2[i4,j4] * (pricelist[j4]/1000) * efficiency) for i4 in range(no_scen**3) for j4 in range(no_settl_pl)] )/(no_scen**3)   +
        #      pulp.lpSum([ePen_S2[i6,j6] * pen * (-1) for i6 in range(no_scen**3) for j6 in range(no_settl_pl)] )/(no_scen**3) +
        #      (sum(pricelist)/len(pricelist)) * pulp.lpSum([pulp.lpSum([eNergyG2V_S2[j9,j10] - eNergyV2G_S2[j9,j10] for j10 in range(no_settl_pl)]) - scenLB[no_settl_pl-1][j9] for j9 in range(no_scen)]))#reward for energy added at the end to make rollover make more sense
        
        inv_eff = 1/efficiency

        model_simp_S2 += pulp.lpSum([
                pulp.lpSum([eNergyG2V_S2[sc,t] * (pricelist[t]/1000) * (-1) for t in range(no_settl_pl)]) + #/ efficiency
                pulp.lpSum([eNergyV2G_S2[sc,t] * (pricelist[t]/1000) for t in range(no_settl_pl)]) + #* efficiency #try 999
                pulp.lpSum([ePen_S2[sc,t] * pen * (-1) for t in range(no_settl_pl)]) +
                pulp.lpSum([pulp.lpSum([eNergyG2V_S2[sc,t] * efficiency - eNergyV2G_S2[sc,t] * inv_eff for t in range(no_settl_pl)]) - (scenUB[no_settl_pl-1][(sc//(no_scen**2))] - scenLB[no_settl_pl-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
        for sc in range(no_scen**3)])/(no_scen**3)

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #constraints #make quicker by using numpy array that I use later?
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(no_settl_pl):
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[scen_id,t1]*efficiency - 
                                                                eNergyV2G_S2[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + #i10 or i10+1
                                                                energy0 <= 
                                                                scenUB[i10][i7])
                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t2]*efficiency - eNergyV2G_S2[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= scenUB[i10][i7] - scenLB[i10][i8] #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        model_simp_S2 += eNergyG2V_S2[scen_id,i10] + eNergyV2G_S2[scen_id,i10]*inv_eff <= scenP[i10][i9]/2
                                        
                                        if preNR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10]+ pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                else:
                                                        raise ValueError("incontinuous time series")
                                                
                                        if prePR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10]+ pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                else:
                                                        raise ValueError("incontinuous time series")

        #other reserve commitments have to be same for service windows (4 settlements)
        for ij in range(no_scen**3): #all boundary scenarios
                        if ij != 0:
                                model_simp_S2 += eNergyG2V_S2[0,0] == eNergyG2V_S2[ij,0]
                                model_simp_S2 += eNergyV2G_S2[0,0] == eNergyV2G_S2[ij,0]
                                #could do the same for penalty but it shouldn't make a difference --> shouldn't do

        #print(model_simp_S2)

        #solve
        pulp.LpSolverDefault.msg = 1
        solver = pulp.GUROBI_CMD(gapRel = 0.00004, timeLimit = 1500)
        #model_simp_S2.solve(pulp.getSolver('GUROBI_CMD'))
        model_simp_S2.solve(solver)

        modelstatus = pulp.LpStatus[model_simp_S2.status]
        
        print(modelstatus)

        #Process results
        vV2G = dict()
        vG2V = dict()
        vPen = dict()

        for variable in model_simp_S2.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "VarDev" in variable.name:
                        pass
                else:
                        #print("Other Variable: ",variable.name)
                        pass

        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)
        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*efficiency - df_temp['V2G']/efficiency
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)
        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))
        df_temp['Price'] = pricelist * no_scen**3
        df_temp['NR'] = [prePR[i10] for i10 in range(no_settl_pl)] * no_scen**3
        df_temp['PR'] = [preNR[i10] for i10 in range(no_settl_pl)] * no_scen**3

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("../stage12results/day"+str(day+1)+"settl"+str(cur_settl)+timestamp+"_S2.csv")
        #vPRl = [k[1] for k in list(sorted(vPR.items()))]
        #vNRl = [k[1] for k in list(sorted(vNR.items()))]
        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,pricelist#,scenUB,scenLB,scenP#,[vPRl,vNRl] #X#X#X# not sure

#stage 1 optimisation for perfect foresight benchmark
def optS1_nzb_AVg_MILP_gurob_noEVs_BM1b(day, dfout, 
        problist =[1],pen = 0.050223125,effy = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        risk_avers = 0.5,cvar_alph=0.1, activation_maxshare = 0.9, nightPR = 0.000437595, dayPR = 0.002008925, NRPRratio = 0.3,bigM = 1000000,bigM2 = 1000000,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)

        #no need for adaptation
        
        #create boundary matrix -> rewrite
        #scenUBi =  np.array(forecast[0])[:, np.newaxis]
        #scenLBi = np.array(forecast[1])[:, np.newaxis]
        #scenPi = np.array(forecast[2])[:, np.newaxis]

        #for cases with low numbers of vehicles
        #scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        #scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        #scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        

        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = 66
        no_scen = 1
        lamb = 1 - risk_avers
        #print(tot_time_len,no_scen,lamb)

        #insert not replace
        #scenUB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+27,'LBc'],repeats=no_scen),scenUBi))
        #scenLB = np.vstack((np.repeat(dfout.loc[day*48+28,'UBc'] - dfout.loc[day*48+28,'LBc'],repeats=no_scen),scenLBi)) #X#X#X#tick
        #scenP = np.vstack((np.repeat(dfout.loc[day*48+28,'Power'],repeats=no_scen),scenPi))

        scenUB = np.vstack(dfout.loc[day*48+28:day*48+28+66,'UBc'] - dfout.loc[day*48+27,'LBc'])
        scenLB = np.vstack(dfout.loc[day*48+28:day*48+28+66,'UBc'] - dfout.loc[day*48+28:day*48+28+66,'LBc'])
        scenP = np.vstack(dfout.loc[day*48+28:day*48+28+66,'Power'])

        print(scenUB)
        
        print(scenLB)

        print(scenP)


        #import wholesale prices
        pricelist = dfout.loc[day*48+28:day*48+28+tot_time_len,'Price'].values.tolist()

        #list with reserve prices --> optimise
        reservepricelist = []
        for i in range(18):
                reservepricelist.append(dayPR)
        for i in range(16):
                reservepricelist.append(nightPR)
        for i in range(32):
                reservepricelist.append(dayPR)

        #initiate model
        modelx = pulp.LpProblem("Mean-CVaR",pulp.LpMaximize)

        #create scenario and time index
        sc_idx = [sc for sc in range(no_scen**3)]
        prob_idx = [problist[i]*problist[j]*problist[k] for i in range(no_scen) for j in range(no_scen) for k in range(no_scen)]
        t_idx = [t for t in range(tot_time_len)]

        #create two main variables
        e_X = pulp.LpVariable("Mean",   lowBound=None, cat='Continuous')
        #cVaR = pulp.LpVariable("CVaR",   lowBound=None, cat='Continuous')
        binPR = pulp.LpVariable.dicts("bin_PR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,
        binNR = pulp.LpVariable.dicts("bin_NR",((t) for t in t_idx),lowBound=0,upBound=1,cat='Integer') #lowBound=0,upBound=1,

        #create other technical variables
        eNergyV2G = pulp.LpVariable.dicts("eV2G", ( (sc,t) for sc in sc_idx for t in t_idx),lowBound=0,cat='Continuous')
        eNergyG2V = pulp.LpVariable.dicts("eG2V", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')
        pR = pulp.LpVariable.dicts("ePR", ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        nR = pulp.LpVariable.dicts("eNR",  ( (t) for t in t_idx), lowBound=0, cat='Continuous')
        ePen = pulp.LpVariable.dicts("ePen", ( (sc,t) for sc in sc_idx for t in t_idx ),lowBound=0,cat='Continuous')

        #create other statistical variables

        #loss deviation
        #VarDev = pulp.LpVariable.dicts("VarDev", ( (sc) for sc in sc_idx ), lowBound=0, cat='Continuous') #this has a lower bound of zero to ensure that only losses that are greater than VaR are included in the CVaR

        #value at risk
        #VaR = pulp.LpVariable("VaR",   lowBound=None, cat='Continuous')

        #define objective function
        modelx += e_X #lamb*e_X - (1-lamb)*cVaR

        #constraints

        #all variables are defined on the grid side which means that they have to be adjusted within the constraints
        #pmax or "rated power" is assumed to refer to the power output (either on grid or vehicle side, depending on whether its V2G or G2V)

        inv_eff = 1/effy #because pulp does not like division operators

        #specify eX
        modelx += pulp.lpSum([(
        pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000)  * (-1) for t in t_idx]) + #/ effy
        pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
        pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
        pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
        pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
        pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick

        ) *prob_idx[sc] for sc in sc_idx]) == e_X

        #specify CVaR through VardDeV and VaR
        #for sc in sc_idx:
        #        modelx += (-1)*(
        #                pulp.lpSum([eNergyG2V[sc,t] * (pricelist[t]/1000) * (-1) for t in t_idx]) + #/ effy
        #                pulp.lpSum([eNergyV2G[sc,t] * (pricelist[t]/1000) for t in t_idx]) + #* effy
        #                pulp.lpSum([ePen[sc,t] * pen * (-1) for t in t_idx]) +
        #                pulp.lpSum([pR[t] * reservepricelist[t] for t in t_idx]) + #* effy
        #                pulp.lpSum([nR[t] * reservepricelist[t]*NRPRratio for t in t_idx]) + #/ effy
        #                pulp.lpSum([pulp.lpSum([eNergyG2V[sc,t]*effy - eNergyV2G[sc,t]*inv_eff for t in t_idx]) - (scenUB[tot_time_len-1][(sc//(no_scen**2))] - scenLB[tot_time_len-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
        #        ) - VaR <= VarDev[sc]

        #modelx += VaR + (1/cvar_alph)*pulp.lpSum([ VarDev[sc]*prob_idx[sc] for sc in sc_idx]) == cVaR

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #energy and power constraints
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(tot_time_len):
                                        
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t1]*effy - eNergyV2G[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + energy0 <= scenUB[i10][i7] #i10 or i10+1 --> i10+1 because of how range works (range(0) would give us nothing)
                                        modelx += pulp.lpSum([eNergyG2V[scen_id,t2]*effy - eNergyV2G[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        modelx += eNergyG2V[scen_id,i10] + eNergyV2G[scen_id,i10]*inv_eff <= (scenP[i10][i9]/2) #needs to be other way around (I think) --> compare to how it was when all variables were vehicle side
                                        
                                        if i10 > 1:

                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)])) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10] * 0.5 * activation_maxshare * effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10] * 0.5 * activation_maxshare / effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10]+ pulp.lpSum([eNergyG2V[scen_id,i10a] - eNergyV2G[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10]+ pulp.lpSum([eNergyV2G[scen_id,i10a] - eNergyG2V[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 1:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM)#i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] + eNergyG2V[scen_id,0] - eNergyV2G[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] + eNergyV2G[scen_id,0] - eNergyG2V[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen[scen_id,i10]*inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        
                                        elif i10 == 0:
                                                #if reserve goes outside boundaries, penalty applies
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t3]*effy - eNergyV2G[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * effy * 0.5 * activation_maxshare - ePen[scen_id,i10]*0.5*activation_maxshare*effy + energy0 <= (scenUB[i10][i7] + binNR[i10]*bigM) #i10 or i10+1
                                                modelx += pulp.lpSum([eNergyG2V[scen_id,t4]*effy - eNergyV2G[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen[scen_id,i10]*0.5*activation_maxshare/effy + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8] - binPR[i10]*bigM) #i10 or i10+1 #X#X#X# tick
                                        
                                                #pen for power also
                                                modelx += ((nR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) * inv_eff - ePen[scen_id,i10]) <= (scenP[i10][i9] + binNR[i10]*bigM)
                                                modelx += ((pR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen[scen_id,i10] * inv_eff) <= (scenP[i10][i9] + binPR[i10]*bigM)
                                        else:
                                               raise ValueError("incontinuous time series")
                                        
        #first reserve commitments are given
        for i11 in range(len(prePR)):
                modelx += nR[i11] == preNR[i11]
                modelx += pR[i11] == prePR[i11]
                if preNR[i11] == 0:
                        modelx += binNR[i11] == 1
                else:
                        modelx += binNR[i11] == 0
                if prePR[i11] == 0:
                        modelx += binPR[i11] == 1
                else:
                        modelx += binPR[i11] == 0

        #other reserve commitments have to be same for service windows (4 settlements)
        for i12 in range(tot_time_len - len(prePR)):
                if i12 % 4 == 0:
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+1]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+1]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+2]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+2]
                        modelx += pR[i12+len(prePR)] == pR[i12+len(prePR)+3]
                        modelx += binPR[i12+len(prePR)] == binPR[i12+len(prePR)+3]

                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+1]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+1]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+2]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+2]
                        modelx += nR[i12+len(prePR)] == nR[i12+len(prePR)+3]
                        modelx += binNR[i12+len(prePR)] == binNR[i12+len(prePR)+3]

        #ensure first period is the same in all scenarios --> later becomes real charging
        for ij in sc_idx: #all boundary scenarios
                        if ij != 0:
                                modelx += eNergyG2V[0,0] == eNergyG2V[ij,0]
                                modelx += eNergyV2G[0,0] == eNergyV2G[ij,0]

        #binary constraints
        for i13 in range(tot_time_len - len(prePR)):
                modelx += bigM2*(1-binNR[i13+len(prePR)]) >= nR[i13+len(prePR)]
                modelx += bigM2*(1-binPR[i13+len(prePR)]) >= pR[i13+len(prePR)]

        solver = pulp.GUROBI_CMD(options=[("MIPgap", 0.00004), ("TimeLimit", "1500")])
        #modelx.solve(pulp.getSolver('GUROBI_CMD'))
        modelx.solve(solver)

        modelstatus = pulp.LpStatus[modelx.status]
        print(modelstatus)

        #process results
        vV2G = dict()
        vG2V = dict()
        vNR = dict()
        vPR = dict()
        vPen = dict()

        for variable in modelx.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eNR" in variable.name:
                        vNR[int(variable.name[4:])] = variable.varValue
                elif "ePR" in variable.name:
                        vPR[int(variable.name[4:])] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                #elif "VarDev" in variable.name:
                        #print(variable.name,variable.varValue)
                        #pass
                else:
                        print(variable.name,variable.varValue)
        
        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)

        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*effy - df_temp['V2G']/effy
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)

        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick --> simply name row D and add another row with LB (=UB-D)
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("../stage12results/day"+str(day+1)+"settl"+timestamp+"_S1.csv")


        vPRl = [vPR[i] for i in range(66)]
        vNRl = [vNR[i] for i in range(66)]
        temp_dict = {'PR': vPRl, 'NR': vNRl}
        df_temp2 = pd.DataFrame(temp_dict)
        df_temp2.to_csv("../stage12results/day"+str(day+1)+"settl"+timestamp+"_S1res.csv")

        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,[vPRl,vNRl]

#stage 2 optimisation for perfect foresight benchmark
def optS2_nzb_AVg_wMILP_gurob_noEVs_BM1b(day,cur_settl,dfout,problist =[1],no_settl_pl = 18,pen = 0.050223125,efficiency = 0.9,energy0 = 0,prePR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],preNR=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], energy_prev_G2V = [0,0], energy_prev_V2G = [0,0],
        activation_maxshare = 0.9,LBUB_diff_threshold=0.1):
        
        print("energy0: ",energy0)
        print("prevG2V: ",energy_prev_G2V)
        print("prevV2G: ",energy_prev_V2G)
        #create boundary matrix -> rewrite
        #scenUBi = np.array(forecast[0])[:, np.newaxis]
        #scenLBi = np.array(forecast[1])[:, np.newaxis]
        #scenPi = np.array(forecast[2])[:, np.newaxis]
        
        #for cases with few vehicles
        #scenUBi[scenUBi<LBUB_diff_threshold] = LBUB_diff_threshold
        #scenLBi[scenLBi<LBUB_diff_threshold] = LBUB_diff_threshold
        #scenPi[scenPi<LBUB_diff_threshold] = LBUB_diff_threshold
        
        #print(scenLBi)
        #print(scenPi)

        #setting a few parameters
        tot_time_len = 18 #len(forecast[0]) + 1
        no_scen = len(problist)

        #insert (not replace) first row of matrices with (no_scen*) real boundaries
        #scenUB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl-1,'LBc'],repeats=no_scen),scenUBi))
        #scenLB = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'UBc'] - dfout.loc[day*48+cur_settl,'LBc'],repeats=no_scen),scenLBi)) #X#X#X# tick
        #scenP = np.vstack((np.repeat(dfout.loc[day*48+cur_settl,'Power'],repeats=no_scen),scenPi))

        scenUB = np.vstack(dfout.loc[day*48+cur_settl:day*48+cur_settl+18,'UBc'] - dfout.loc[day*48+cur_settl-1,'LBc'])
        scenLB = np.vstack(dfout.loc[day*48+cur_settl:day*48+cur_settl+18,'UBc'] - dfout.loc[day*48+cur_settl:day*48+cur_settl+18,'LBc']) #X#X#X# tick
        scenP = np.vstack(dfout.loc[day*48+cur_settl:day*48+cur_settl+18,'Power'])

        print(scenUB)
        
        print(scenLB)

        print(scenP)

        #import wholesale prices
        pricelist = dfout.loc[day*48+cur_settl:day*48+cur_settl+no_settl_pl-1,'Price'].values.tolist()
        
        #init model
        model_simp_S2 = pulp.LpProblem("Mean_S2", pulp.LpMaximize)

        #Define technical variables
        eNergyV2G_S2 = pulp.LpVariable.dicts("eV2G_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl)),lowBound=0,cat='Continuous')
        eNergyG2V_S2 = pulp.LpVariable.dicts("eG2V_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')
        ePen_S2 = pulp.LpVariable.dicts("ePen_S2", ( (i1,j1) for i1 in range(no_scen*no_scen*no_scen) for j1 in range(no_settl_pl) ),lowBound=0,cat='Continuous')

        #Model Objective
        #model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[i5,j5] * (pricelist[j5]/1000) / efficiency * (-1)  for i5 in range(no_scen**3) for j5 in range(no_settl_pl)] )/(no_scen**3) + 
        #      pulp.lpSum([(eNergyV2G_S2[i4,j4] * (pricelist[j4]/1000) * efficiency) for i4 in range(no_scen**3) for j4 in range(no_settl_pl)] )/(no_scen**3)   +
        #      pulp.lpSum([ePen_S2[i6,j6] * pen * (-1) for i6 in range(no_scen**3) for j6 in range(no_settl_pl)] )/(no_scen**3) +
        #      (sum(pricelist)/len(pricelist)) * pulp.lpSum([pulp.lpSum([eNergyG2V_S2[j9,j10] - eNergyV2G_S2[j9,j10] for j10 in range(no_settl_pl)]) - scenLB[no_settl_pl-1][j9] for j9 in range(no_scen)]))#reward for energy added at the end to make rollover make more sense
        
        inv_eff = 1/efficiency

        model_simp_S2 += pulp.lpSum([
                pulp.lpSum([eNergyG2V_S2[sc,t] * (pricelist[t]/1000) * (-1) for t in range(no_settl_pl)]) + #/ efficiency
                pulp.lpSum([eNergyV2G_S2[sc,t] * (pricelist[t]/1000) for t in range(no_settl_pl)]) + #* efficiency #try 999
                pulp.lpSum([ePen_S2[sc,t] * pen * (-1) for t in range(no_settl_pl)]) +
                pulp.lpSum([pulp.lpSum([eNergyG2V_S2[sc,t] * efficiency - eNergyV2G_S2[sc,t] * inv_eff for t in range(no_settl_pl)]) - (scenUB[no_settl_pl-1][(sc//(no_scen**2))] - scenLB[no_settl_pl-1][((sc//no_scen)%no_scen)])]) * (sum(pricelist)/len(pricelist))/1000 #X#X#X# tick
        for sc in range(no_scen**3)])/(no_scen**3)

        #print previous
        #print(energy_prev_V2G[0])
        #print(energy_prev_V2G[1])
        #print(energy_prev_G2V[0])
        #print(energy_prev_G2V[1])
        #print(energy_prev_V2G[0] + energy_prev_V2G[1] - energy_prev_G2V[0] - energy_prev_G2V[1])

        #constraints #make quicker by using numpy array that I use later?
        for i7 in range(no_scen): #upper boundary scenarios
                for i8 in range(no_scen): #lower boundary scenarios
                        for i9 in range(no_scen): #power boundary scenarios
                                #scenario number
                                scen_id = i7*no_scen*no_scen+i8*no_scen+i9
                                for i10 in range(no_settl_pl):
                                        #cumulative charging trajectory needs to stay within scenario boundaries
                                        model_simp_S2 += (pulp.lpSum([eNergyG2V_S2[scen_id,t1]*efficiency - 
                                                                eNergyV2G_S2[scen_id,t1]*inv_eff for t1 in range(i10+1)]) + #i10 or i10+1
                                                                energy0 <= 
                                                                scenUB[i10][i7])
                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t2]*efficiency - eNergyV2G_S2[scen_id,t2]*inv_eff for t2 in range(i10+1)]) + energy0 >= scenUB[i10][i7] - scenLB[i10][i8] #i10 or i10+1 #X#X#X# tick
                                        
                                        #always within power
                                        model_simp_S2 += eNergyG2V_S2[scen_id,i10] + eNergyV2G_S2[scen_id,i10]*inv_eff <= scenP[i10][i9]/2
                                        
                                        if preNR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10]+ pulp.lpSum([eNergyG2V_S2[scen_id,i10a] - eNergyV2G_S2[scen_id,i10a] for i10a in range(i10-2,i10)]))*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] + eNergyG2V_S2[scen_id,0] - eNergyV2G_S2[scen_id,0] + energy_prev_G2V[1] - energy_prev_V2G[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t3]*efficiency - eNergyV2G_S2[scen_id,t3]*inv_eff for t3 in range(i10+1)]) + (preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1]) *efficiency * 0.5 * activation_maxshare - ePen_S2[scen_id,i10]*0.5*activation_maxshare*efficiency + energy0 <= (scenUB[i10][i7]) #i10 or i10+1
                                                        
                                                        #pen for power also
                                                        model_simp_S2 += ((preNR[i10] - energy_prev_V2G[0] - energy_prev_V2G[1] + energy_prev_G2V[0] + energy_prev_G2V[1])*inv_eff - ePen_S2[scen_id,i10]) <= (scenP[i10][i9])
                                                        
                                                else:
                                                        raise ValueError("incontinuous time series")
                                                
                                        if prePR[i10] > 0:

                                                if i10 > 1:

                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10]+ pulp.lpSum([eNergyV2G_S2[scen_id,i10a] - eNergyG2V_S2[scen_id,i10a] for i10a in range(i10-2,i10)])) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 1:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] + eNergyV2G_S2[scen_id,0] - eNergyG2V_S2[scen_id,0] + energy_prev_V2G[1] - energy_prev_G2V[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                
                                                elif i10 == 0:
                                                        #if reserve goes outside boundaries, penalty applies
                                                        model_simp_S2 += pulp.lpSum([eNergyG2V_S2[scen_id,t4]*efficiency - eNergyV2G_S2[scen_id,t4]*inv_eff for t4 in range(i10+1)]) - (prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) * inv_eff * 0.5 * activation_maxshare + ePen_S2[scen_id,i10]*0.5*activation_maxshare/efficiency + energy0 >= (scenUB[i10][i7] - scenLB[i10][i8]) #i10 or i10+1 #X#X#X# tick
                                                
                                                        #pen for power also
                                                        model_simp_S2 += ((prePR[i10] - energy_prev_G2V[0] - energy_prev_G2V[1] + energy_prev_V2G[0] + energy_prev_V2G[1]) - ePen_S2[scen_id,i10]*inv_eff) <= (scenP[i10][i9])
                                                else:
                                                        raise ValueError("incontinuous time series")

        #other reserve commitments have to be same for service windows (4 settlements)
        for ij in range(no_scen**3): #all boundary scenarios
                        if ij != 0:
                                model_simp_S2 += eNergyG2V_S2[0,0] == eNergyG2V_S2[ij,0]
                                model_simp_S2 += eNergyV2G_S2[0,0] == eNergyV2G_S2[ij,0]
                                #could do the same for penalty but it shouldn't make a difference --> shouldn't do

        #print(model_simp_S2)

        #solve
        pulp.LpSolverDefault.msg = 1
        solver = pulp.GUROBI_CMD(gapRel = 0.00004, timeLimit = 1500)
        #model_simp_S2.solve(pulp.getSolver('GUROBI_CMD'))
        model_simp_S2.solve(solver)

        modelstatus = pulp.LpStatus[model_simp_S2.status]
        
        print(modelstatus)

        #Process results
        vV2G = dict()
        vG2V = dict()
        vPen = dict()

        for variable in model_simp_S2.variables():
                if "eV2G" in variable.name:
                        vV2G[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "eG2V" in variable.name:
                        vG2V[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "ePen" in variable.name:
                        vPen[(int(variable.name[variable.name.find(start:='_(')+len(start):variable.name.find(',_')]),int(variable.name[variable.name.find(start:=',_')+len(start):variable.name.find(')')]))] = variable.varValue
                elif "VarDev" in variable.name:
                        pass
                else:
                        #print("Other Variable: ",variable.name)
                        pass

        #create multiindex dataframe
        s_tempV2G = pd.Series(vV2G.values(), index=pd.MultiIndex.from_tuples(list(vV2G.keys()),names=('scenario','timeperiod')), name='V2G').sort_index()
        s_tempG2V = pd.Series(vG2V.values(), index=pd.MultiIndex.from_tuples(list(vG2V.keys()),names=('scenario','timeperiod')), name='G2V').sort_index()
        s_tempPen = pd.Series(vPen.values(), index=pd.MultiIndex.from_tuples(list(vPen.keys()),names=('scenario','timeperiod')), name='Pen').sort_index()
        df_temp=pd.concat([s_tempV2G,s_tempG2V,s_tempPen],axis=1)
        df_temp['o_charge'] = df_temp['G2V'] - df_temp['V2G']
        df_temp['o_charge2'] = df_temp['G2V']*efficiency - df_temp['V2G']/efficiency
        df_temp['energy_traj'] = df_temp.groupby(level=0)['o_charge'].cumsum() + energy0
        df_temp['energy_traj2'] = df_temp.groupby(level=0)['o_charge2'].cumsum() + energy0

        df_temp.reset_index(inplace=True)
        df_temp['UB'] = pd.Series(np.tile(scenUB, (no_scen*no_scen, 1)).flatten('F'))
        df_temp['D'] = pd.Series(np.tile(np.tile(scenLB, (no_scen, 1)).flatten('F'),no_scen)) #X#X#X# tick
        df_temp['LB'] = df_temp['UB'] - df_temp['D']
        df_temp['P'] = pd.Series(np.tile(scenP.flatten('F'),no_scen*no_scen))
        df_temp['Price'] = pricelist * no_scen**3
        df_temp['NR'] = [prePR[i10] for i10 in range(no_settl_pl)] * no_scen**3
        df_temp['PR'] = [preNR[i10] for i10 in range(no_settl_pl)] * no_scen**3

        df_temp.set_index(['scenario', 'timeperiod'], inplace=True)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        df_temp.to_csv("../stage12results/day"+str(day+1)+"settl"+str(cur_settl)+timestamp+"_S2.csv")
        #vPRl = [k[1] for k in list(sorted(vPR.items()))]
        #vNRl = [k[1] for k in list(sorted(vNR.items()))]
        return df_temp.loc[(0,0),'G2V'],df_temp.loc[(0,0),'V2G'],df_temp,modelstatus,pricelist#,scenUB,scenLB,scenP#,[vPRl,vNRl] #X#X#X# not sure

