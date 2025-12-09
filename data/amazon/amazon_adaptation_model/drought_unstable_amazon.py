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
from evolve import evolve, NoEquilibrium
from tipping_element import cusp
from coupling import linear_coupling




sys_var = np.array(sys.argv[2:])
year = sys_var[0]
start_file = sys_var[1]


################################GLOBAL VARIABLES################################
no_cpl_dummy = 0
if no_cpl_dummy == 0: #no_cpl_dummy can be used to shut on or off the coupling; If True then there is no coupling, False with normal coupling
    no_cpl_dummy = False 
elif no_cpl_dummy == 1:
    no_cpl_dummy = True
else:
    print("Uncorrect value given for no_cpl_dummy, namely: {}".format(no_cpl_dummy))
    die


adapt_fact = np.loadtxt("probabilistic_ensemble/drought_start_sample_save/{}.txt".format(str(start_file).zfill(3)))		  #Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
rain_fact = 1.0                                                                                               #This variable can be used to evaluate tipping points in the Amazon rainforest by artifically reducing the rain
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


#Tipping function
def tip( net , initial_state ):
    ev = evolve( net , initial_state )

    tolerance = 0.01
    t_step = 1
    #dc = 0.01
    realtime_break = 30000
    
    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
        print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    ev.equilibrate(tolerance, t_step, realtime_break)

    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, net.get_number_tipped(ev.get_timeseries()[1][-1,:]), net.get_tip_states(ev.get_timeseries()[1][-1])[:], ev.get_timeseries()[1][-1,:]
    



###MAIN - PREPARATION###
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)
#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]

resolution_type = "1deg"
year_type = year

#Network is created using the monthly data, the critical mcwd, coupling switch and the rain factor
net = generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact)


###MAIN - PREPARATION###
init_state = np.zeros(net.number_of_nodes())
init_state.fill(-1) #initial state should be -1 instead of 0 everywhere

#Without the source node tipped
info = tip(net, init_state)
conv_time = info[0]
casc_size = info[1] 
unstable_amaz = info[2]

#absolute values are saved here to secure also smaller changes that are not tipping, but still relevant
abs_values = info[3]


np.savetxt("results/droughts/unstable_amaz_{}_{}_adaptsample{}_field.txt".format(resolution_type, year_type, str(start_file).zfill(3)), unstable_amaz)
np.savetxt("results/droughts/unstable_amaz_{}_{}_adaptsample{}_total.txt".format(resolution_type, year_type, str(start_file).zfill(3)), [conv_time, casc_size])


#save absolute values
abs_values = np.array(abs_values)
np.savetxt("results/droughts/abs_values_unstable_amaz_{}_{}_adaptsample{}_field.txt".format(resolution_type, year_type, str(start_file).zfill(3)), abs_values)



#plotting procedure
print("Plotting sequence")
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

lat = np.unique(lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) #why do we need to append lat[-1]+lat[-1]-lat[-2]???
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])
vals = np.empty((lat.size,lon.size))
vals[:,:] = np.nan

for idx,x in enumerate(lat):
    for idy,y in enumerate(lon):
        if (x,y) in tuples:
            p = unstable_amaz[tuples.index((x,y))]
            vals[idx,idy] = p


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
cmap = plt.get_cmap('rainbow')

plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Unstable Amazon')

plt.savefig("results/droughts/unstable_amaz_{}_{}_adaptsample{}.png".format(resolution_type, year_type, str(start_file).zfill(3)), bbox_inches='tight')
plt.savefig("results/droughts/unstable_amaz_{}_{}_adaptsample{}.pdf".format(resolution_type, year_type, str(start_file).zfill(3)), bbox_inches='tight')

#plt.show()
plt.clf()
plt.close()


print("Finish")