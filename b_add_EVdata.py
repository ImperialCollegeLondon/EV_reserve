import pandas as pd

import auxfunc_data as ctf

csv_in = pd.read_csv('../Raw data/chargepoint analysis wo doubles.csv') # previously "chargepoint analysis"

#for implementation with cleaned file
csv_in2 = csv_in.drop(columns=['StartPd', 'EndPd'])
csv2a_pdtime = ctf.add_pdtime(csv_in2)

csv2c_speed = ctf.add_speed(csv2a_pdtime) #adds column with charging speed (energy/time(hours)) (=kW)
csv2d_maxspeed = ctf.add_maxspeed(csv2c_speed) #adds column with maximum speed for each charger
csv2e_maxbat = ctf.add_maxbattery(csv2d_maxspeed) #adds column with maximum energy for each charger
csv2f_settl = ctf.add_settl(csv2e_maxbat) #adds settlements
csv2g_settldrop = csv2f_settl.drop(columns=['StartDate', 'StartTime','EndDate','EndTime'])

#add minimal energy capacity and rated power
csv2h_guarantspeed = ctf.min_max_speed(csv2g_settldrop,7)
csv2g_final3 = ctf.min_max_battery(csv2h_guarantspeed,16)

#print(csv2g_final)
