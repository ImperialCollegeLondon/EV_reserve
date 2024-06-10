#this data cleaning step is quite computationally expensive, so that it should be executed once and afterwards the cleaned data file can be used
import pandas as pd
import piso
import auxfunc_data as ctf

csv_in = pd.read_csv('C:/Users/jt3022/OneDrive - Imperial College London/Output/Projects/2. EV SMPC/3. Methodology/Data/Raw EV/chargepoint analysis.csv')

csv_int1 = ctf.add_pdtime(csv_in) #adds column with pandas time
csv_int2 = ctf.remove48(csv_int1,336) #any charging process that lasts longer than two weeks is removed

csv_int2['isOverlap'] = 0
for i in csv_int2['CPID'].unique():
    iii = csv_int2.loc[csv_int2['CPID'] == i]
    ii = pd.IntervalIndex.from_arrays(iii["StartPd"], iii["EndPd"])
    csv_int2.loc[csv_int2['CPID'] == i,"isOverlap"] = piso.adjacency_matrix(ii).any(axis=1).astype(int).values

#drop
csv_out = csv_int2.drop(csv_int2[(csv_int2['isOverlap'] == 1)].index)

#resave
csv_out.to_csv("../Raw data/chargepoint analysis wo doubles.csv")