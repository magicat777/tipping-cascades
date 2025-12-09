import time
import numpy as np
import networkx as nx
import glob
import re
import os

import csv
from netCDF4 import Dataset

#plotting imports
import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
sns.set(font_scale=2.5)
sns.set_style("whitegrid")

#self programmed code imports
import sys
sys.path.append('pycascades/modules/gen')
sys.path.append('pycascades/modules/core')

from net_factory import from_nxgraph
#from gen.net_factory import from_nxgraph
from amazon import generate_network
from functions_amazon import global_functions
from evolve import evolve, NoEquilibrium
from tipping_element import cusp
from coupling import linear_coupling




sys_var = np.array(sys.argv[2:])
year = sys_var[0]
start_file = sys_var[1]
#Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
adapt_fact = np.loadtxt("probabilistic_ensemble/drought_start_sample_save/{}.txt".format(str(start_file).zfill(3)))	



################################GLOBAL VARIABLES################################
no_cpl_dummy = 0
if no_cpl_dummy == 0: #no_cpl_dummy can be used to shut on or off the coupling; If True then there is no coupling, False with normal coupling
    no_cpl_dummy = False 
elif no_cpl_dummy == 1:
    no_cpl_dummy = True
else:
    print("Uncorrect value given for no_cpl_dummy, namely: {}".format(no_cpl_dummy))
    die

rain_fact = 1.0        #This variable can be used to evaluate tipping points in the Amazon rainforest by artifically reducing the rain
################################################################################


#GLOBAL VARIABLES
#the first two datasets are required to compute the critical values
data_crit_1 = np.sort(np.array(glob.glob("average_network/hydrological_era5_average/average_both_1deg*.nc")))[9:]
data_crit_2 = np.sort(np.array(glob.glob("average_network/hydrological_era5_average/average_both_1deg*.nc")))[:9]
data_crit = np.concatenate((data_crit_1, data_crit_2))
data_crit_std_1 = np.sort(np.array(glob.glob("average_network/hydrological_era5_average/average_std_both_1deg*.nc")))[9:]
data_crit_std_2 = np.sort(np.array(glob.glob("average_network/hydrological_era5_average/average_std_both_1deg*.nc")))[:9]
data_crit_std = np.concatenate((data_crit_std_1, data_crit_std_2))

#evaluated drought (or not drought) year
hydro_1 = np.sort(np.array(glob.glob("average_network/era5_new_network_data/*{}*.nc".format(int(year)-1))))[9:]
hydro_2 = np.sort(np.array(glob.glob("average_network/era5_new_network_data/*{}*.nc".format(int(year)))))[:9]
data_eval = np.concatenate((hydro_1, hydro_2))



###MAIN - PREPARATION###
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)
#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]

resolution_type = "1deg"
year_type = year



field = np.loadtxt("results/droughts/unstable_amaz_{}_{}_adaptsample{}_field.txt".format(resolution_type, year_type, str(start_file).zfill(3)))


#ACTUAL VALUES for rainfall and mcwd from evaluated data
rain = global_functions.Rain(data_eval, rain_fact) 
mcwd = global_functions.Mcwd(data_eval, rain_fact)

#CRITICAL DATA for critical rainfall and mcwd
rain_crit = global_functions.Rain_crit(data_crit, data_crit_std, adapt_fact)
mcwd_crit = global_functions.Mcwd_crit(data_crit, data_crit_std, adapt_fact)

#Continuous variables
diff_rain = np.subtract(rain, rain_crit) #all negative values are considered to be tipped
diff_mcwd = np.subtract(mcwd_crit, mcwd) #all negative values are considered to be tipped





np.savetxt("results/droughts_tipping_reason/rain_diff_{}_{}_adaptsample{}_rainfact{}_field.txt".format(resolution_type, 
    year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), diff_rain)
np.savetxt("results/droughts_tipping_reason/mcwd_diff_{}_{}_adaptsample{}_rainfact{}_field.txt".format(resolution_type, 
    year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), diff_mcwd)


#First plotting procedure
print("First plotting procedure")
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

lat = np.unique(lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) #why do we need to append lat[-1]+lat[-1]-lat[-2]???
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])

vals_rain = np.empty((lat.size,lon.size))
vals_rain[:,:] = np.nan

vals_mcwd = np.empty((lat.size,lon.size))
vals_mcwd[:,:] = np.nan


for idx,x in enumerate(lat):
    for idy,y in enumerate(lon):
        if (x,y) in tuples:
            vals_rain[idx, idy] = diff_rain[tuples.index((x,y))]
            vals_mcwd[idx, idy] = diff_mcwd[tuples.index((x,y))]




plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

#FIRST PLOT
plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
cmap = plt.get_cmap('seismic_r')


#colorbar limits need to be set
max_color = np.amax([np.amax(diff_rain), -np.amin(diff_rain)])
plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals_rain, cmap=cmap, vmin=-max_color, vmax=max_color)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Unstable Amazon')


plt.savefig("results/droughts_tipping_reason/rain_diff_{}_{}_adaptsample{}_rainfact{}.png".format(resolution_type, 
    year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
plt.savefig("results/droughts_tipping_reason/rain_diff_{}_{}_adaptsample{}_rainfact{}.pdf".format(resolution_type, 
    year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
#plt.show()
plt.clf()
plt.close()



#SECOND PLOT
plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
cmap = plt.get_cmap('seismic_r')

#colorbar limits need to be set
max_color = np.amax([np.amax(diff_mcwd), -np.amin(diff_mcwd)])
plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals_mcwd, cmap=cmap, vmin=-max_color, vmax=max_color)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Unstable Amazon')

plt.savefig("results/droughts_tipping_reason/mcwd_diff_{}_{}_adaptsample{}_rainfact{}.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
plt.savefig("results/droughts_tipping_reason/mcwd_diff_{}_{}_adaptsample{}_rainfact{}.pdf".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
#plt.show()
plt.clf()
plt.close()




#thresholded values (0: no tipping, 1: rainfall reason, 2: mcwd reason, 3: both, 4: additional tipping due to network)
np.place(diff_rain, diff_rain > 0.0, 0.0)
np.place(diff_rain, diff_rain < 0.0, 1.0) #only critical values for rainfall are masked with a 1.0

np.place(diff_mcwd, diff_mcwd > 0.0, 0.0)
np.place(diff_mcwd, diff_mcwd < 0.0, 2.0) #only critical values for mcwd are masked with a 2.0

both_diff = np.add(diff_rain, diff_mcwd) #only critical values for rainfall AND mcwd are masked with a 3.0

cpl_effect = np.subtract(both_diff, field)
np.place(cpl_effect, cpl_effect == -1.0, 4.0)
np.place(cpl_effect, cpl_effect != 4.0, 0.0) #only additionally tipped values due to network effects are masked with a 4.0




#complete field
complete_discrete = np.add(cpl_effect, both_diff)
np.savetxt("results/droughts_tipping_reason/complete_discrete_diff_{}_{}_adaptsample{}_rainfact{}_field.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), complete_discrete)



#Second plotting procedure
print("Second plotting procedure")
dataset = data_crit[0]
net_data = Dataset(dataset)
#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

lat = np.unique(lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) #why do we need to append lat[-1]+lat[-1]-lat[-2]???
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])


vals_complete_discrete = np.empty((lat.size,lon.size))
vals_complete_discrete[:,:] = np.nan

for idx,x in enumerate(lat):
    for idy,y in enumerate(lon):
        if (x,y) in tuples:
            vals_complete_discrete[idx, idy] = complete_discrete[tuples.index((x,y))]

#THIRD PLOT
plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
cmap = plt.get_cmap('inferno_r')

plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals_complete_discrete, cmap=cmap)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Unstable Amazon')

plt.savefig("results/droughts_tipping_reason/complete_discrete_diff_{}_{}_adaptsample{}_rainfact{}.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
plt.savefig("results/droughts_tipping_reason/complete_discrete_diff_{}_{}_adaptsample{}_rainfact{}.pdf".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*rain_fact))), bbox_inches='tight')
#plt.show()
plt.clf()
plt.close()


print("Finish")