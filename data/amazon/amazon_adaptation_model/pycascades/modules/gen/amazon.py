import sys

from tipping_network import tipping_network
from tipping_element import cusp
from coupling import linear_coupling

#NW - my functions
from functions_amazon import global_functions


from netCDF4 import Dataset



import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
sns.set_style("whitegrid")


def generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact): #Network with data file, mean rainfall, critical rainfall, coupling dummy [shut on or off coupling], reduced rainfall
    net = tipping_network()

    #get longitude and latitude values, for that use first network
    net_data = Dataset(data_eval[0])
    lon_x = net_data.variables["lon"][:]
    lat_y = net_data.variables["lat"][:]


    #ACTUAL VALUES for rainfall and mcwd from evaluated data
    rain = global_functions.Rain(data_eval, rain_fact) 
    mcwd = global_functions.Mcwd(data_eval, rain_fact)


    #CRITICAL DATA for critical rainfall and mcwd
    rain_crit = global_functions.Rain_crit(data_crit, data_crit_std, adapt_fact)
    mcwd_crit = global_functions.Mcwd_crit(data_crit, data_crit_std, adapt_fact)
    #Note: Critical rainfall and MCWD values are very far away from usual values meaning that the natural variability is large!!

    #MEAN VALUES for rainfall and Mcwd from averaged data; no rain_fact != 1.0 should be used here
    rain_average = global_functions.Rain(data_crit, 1.0)
    mcwd_average = global_functions.Mcwd(data_crit, 1.0)

    for idx, val in enumerate(lon_x):
        #constants that are necessary to set up the tipping network
        rain_current = rain[idx] #rainfall in a certain cell
        rain_critical = rain_crit[idx] #critical rainfall of a certain cell
        rain_mean = rain_average[idx] #mean rainfall in that certain cell

        mcwd_current = mcwd[idx] #mcwd in a certain cell
        mcwd_critical = mcwd_crit[idx] #critical mcwd of a certain cell
        mcwd_mean = mcwd_average[idx] #mean rainfall in that certain cell

        a = 1 #prefactor in the cubic term of the CUSP-DEQ
        b = 1 #prefactor in the linear term of the CUSP-DEQ


        element = cusp( a = -1, b = 1, 
            c = global_functions.Amazon_CUSPc(a, b, rain_mean, rain_critical, rain_current, mcwd_mean, mcwd_critical, mcwd_current), 
            x_0 = 0)
        net.add_element(element)
        net.node[idx]['pos'] = (val, lat_y[idx])
    
    
    flows_xy_total = []
    for data in data_eval:
        data_file = Dataset(data)
        flows_xy_total.append(data_file.variables["network"][:, :])
    flows_xy_total = np.array(flows_xy_total)


    #compute total moisture recycling within a year
    flows_xy = Dataset(data_eval[0]).variables["network"][:, :]
    it = np.nditer(flows_xy, flags=['multi_index'])

    couplings = []
    while not it.finished:
        if not it.multi_index[0] == it.multi_index[1]:
            """
            In 'it.multi_index[0]' and 'it.multi_index[1]' we have the receiving[0] and the starting cell[1] of the moisture transport
            'it[0]' gives the amount of moisture transport 
            'rain[it.multi_index[0]]' the rainfall of the receiving cell and 'rainfall[multi_index[1]]' the rainfall of the starting cell
            Note: Since MATLAB and python are row and respectively column major, the index 0 is the receiving cell and the index 1 the starting cell
            """
            #print(it.multi_index[0], it.multi_index[1], it[0], rain[it.multi_index[0]], rain[it.multi_index[1]])


            ###########
            #need to get value for each month
            monthly = []
            for i in range(0, len(flows_xy_total)):
                monthly.append(flows_xy_total[i, it.multi_index[0], it.multi_index[1]])

            appender = [it.multi_index[1], it.multi_index[0], np.sum(monthly)]
            for i in monthly:
                appender.append(i)
            #print(appender)
            
            couplings.append([i for i in appender])
            #print(couplings)
            

        it.iternext()


    couplings = np.array(couplings)
    couplings = np.array(sorted(couplings, key=lambda x: x[2]))


    #sort out zero values in array
    cpl_hist = np.array(couplings).T[2]
    start_idx = len(couplings) - np.count_nonzero(cpl_hist)#count length of zeros
    couplings = np.array(couplings[start_idx:]) # sort out where couplings are zero



    """
    #test histogram of moisture recycling
    plt.grid(True)
    plt.hist(cpl_hist[start_idx:], bins = 100)
    plt.show()
    plt.clf()
    die
    """

    #Resolution dependent coupling limit: Setting where couplings below cpl_limit mm/year are neglected  
    cpl_limit = 1.0 # time to compute the new network sensitively depends on the coupling limit; 
    #takes about 2 minutes if cpl_limit = 10 mm/yr, would take about 30 minutes for cpl_limit = 3 mm/yr as before, and approximately 1-3 hours for cpl_limit = 1.0
    
    #kk = 0 #test variable
    #maxlen = len(np.where(cpl_hist > cpl_limit)[0])
    #print(maxlen)

    for cpl in couplings:
        if cpl[2] > cpl_limit:


            #print(kk/maxlen)
            #kk+=1

            if no_cpl_dummy == True:
                coupling_object = linear_coupling(strength = 0.0, x_0 = -1.0)
            else:
                #get the difference between rainfall in the respective cell after[rain_new] and before[rain_old] tipping
                #[cpl[i] for i in range(3, len(cpl))] goes through all 12 months without the indices and  the mean annual precipitation
                delta_rain = global_functions.Rain_moisture_delta_only([cpl[i] for i in range(3, len(cpl))]) #delta_rain is a negative number
                rain_old = rain[int(cpl[1])]
                rain_new = rain_old + delta_rain

                #Here the moisture recycling value needs to be recomputed into a change of MCWD;
                #it is necessary to give the rainfall moisture recycling value along that the cell receives (cpl[2])
                mcwd_new = global_functions.Mcwd_moisture(data_eval, rain_fact, int(cpl[1]), [cpl[i] for i in range(3, len(cpl))])
                mcwd_old = mcwd[int(cpl[1])] #int(cpl[1]) gives the id of the node that receives moisture recycling
                delta_mcwd = mcwd_new - mcwd_old


                #current, mean and critical values are required for the rainfall and mcwd
                rain_current = rain[int(cpl[1])]
                rain_mean = rain_average[int(cpl[1])]
                rain_critical = rain_crit[int(cpl[1])]

                mcwd_current = mcwd[int(cpl[1])]
                mcwd_mean = mcwd_average[int(cpl[1])]
                mcwd_critical = mcwd_crit[int(cpl[1])]

     
                if delta_mcwd < 0.:
                    print("Error: The new MCWD can only be larger than the old one: Negative couplings are forbidden")
                    die

                #Amount of change of MCWD is the coupling strength
                coupling_object = linear_coupling(strength = 
                    global_functions.Amazon_cpl(a, b, rain_mean, rain_critical, rain_current, delta_rain, mcwd_mean, mcwd_critical, mcwd_current, delta_mcwd), 
                    x_0 = -1.0)

            net.add_coupling( int(cpl[0]), int(cpl[1]), coupling_object ) #needs to be saved as integer here

    print("Amazon rainforest network generated! Restriction: Only moisture recycling links above {} mm/yr are considered".format(cpl_limit))
    d = net.number_of_edges() / net.number_of_nodes()
    average_clustering = nx.average_clustering(net)
    #print("Average degree: ", d)
    #print("Average clustering: ", average_clustering)
    return net