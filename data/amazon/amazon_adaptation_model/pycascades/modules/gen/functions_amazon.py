import numpy as np
from netCDF4 import Dataset
import warnings

"""
Here all global functions are stored - the functions are up to "choice": Here, linear functions are used
"""


class global_functions():
    def Rain(datasets, rain_fact):
        """
        Computation of rainfall values; all datasets are necessary and a potential rain factor
        """
        rain = []
        for data in datasets:
            net_data = Dataset(data)
            #get mean value for rainfall
            rain_dataset = np.multiply(rain_fact, net_data.variables["rain"][:])
            if len(rain) == 0:
                rain = rain_dataset
            else:
                rain = np.add(rain, rain_dataset)

        rain = np.array(rain)
        return rain


    def Rain_crit(data_crit, data_crit_std, adapt_fact):
        """
        Computation of critical rainfall values; all datasets are necessary and a potential rain factor
        """
        rain_crit = []
        for i in range(0, len(data_crit)):
            net_data = Dataset(data_crit[i])
            net_data_std = Dataset(data_crit_std[i])

            #get mean value for rainfall
            if type(adapt_fact) is np.ndarray:
                rain_dataset = np.subtract(net_data.variables["rain"][:], np.multiply(net_data_std.variables["rain"][:], adapt_fact[:]))
            else:
                rain_dataset = np.subtract(net_data.variables["rain"][:], np.multiply(net_data_std.variables["rain"][:], adapt_fact))

            #values smaller zero do not make sense
            rain_dataset[rain_dataset < 0.0] = 0.0
            if len(rain_crit) == 0:
                rain_crit = rain_dataset
            else:
                rain_crit = np.add(rain_crit, rain_dataset)

        rain_crit = np.array(rain_crit)


        #here also absolute limits can be set
        global_limit_rain = 0.0
        rain_crit[rain_crit < global_limit_rain] = global_limit_rain
        return rain_crit


    def Mcwd(datasets, rain_fact):
        """
        Computation of MCWD; all datasets are necessary and a potential rain factor
        """
        mcwd_array = []
        for data in datasets:
            net_data = Dataset(data)
            #get mean value for rainfall
            rain = np.multiply(rain_fact, net_data.variables["rain"][:])
            evap = net_data.variables["evap"][:]

            #computation of MCWD
            diff = np.subtract(evap, rain)
            mcwd_array.append(diff)
        mcwd_array = np.array(mcwd_array)


        mcwd = []
        for i in range(0, len(mcwd_array[0])): #necessary for assessing an mcwd value for each cell
            mcwd_probe = []
            for j in mcwd_array:
                mcwd_probe.append(j[i])
            mcwd_probe = np.array(mcwd_probe)
            mcwd_sign = np.sign(mcwd_probe)

            mcwd_real = []
            diff = 0.
            for j in range(0, len(mcwd_probe)):
                if mcwd_sign[j] < 0.:
                    diff = 0.
                else:
                    diff += mcwd_probe[j]
                mcwd_real.append(diff)
            mcwd_real = np.array(mcwd_real)

            mcwd.append(np.amax(mcwd_real))
        mcwd = np.array(mcwd)
        return mcwd


    def Mcwd_crit(data_crit, data_crit_std, adapt_fact):
        """
        Computation of critical MCWD values; all datasets are necessary and a potential rain factor
        """
        mcwd_array = []
        for i in range(0, len(data_crit)):
            net_data = Dataset(data_crit[i])
            net_data_std = Dataset(data_crit_std[i])
            #get mean value for rainfall
            if type(adapt_fact) is np.ndarray:
                rain = np.subtract(net_data.variables["rain"][:], np.multiply(net_data_std.variables["rain"][:], adapt_fact[:]))
            else:
                rain = np.subtract(net_data.variables["rain"][:], np.multiply(net_data_std.variables["rain"][:], adapt_fact))
            #values smaller zero do not make sense
            rain[rain<0.0] = 0.0

            evap = net_data.variables["evap"][:]

            #computation of MCWD
            diff = np.subtract(evap, rain)
            mcwd_array.append(diff)
        mcwd_array = np.array(mcwd_array)


        mcwd_crit = []
        for i in range(0, len(mcwd_array[0])): #necessary for assessing an mcwd_crit value for each cell
            mcwd_probe = []
            for j in mcwd_array:
                mcwd_probe.append(j[i])
            mcwd_probe = np.array(mcwd_probe)
            mcwd_sign = np.sign(mcwd_probe)

            mcwd_real = []
            diff = 0.
            for j in range(0, len(mcwd_probe)):
                if mcwd_sign[j] < 0.:
                    diff = 0.
                else:
                    diff += mcwd_probe[j]
                mcwd_real.append(diff)
            mcwd_real = np.array(mcwd_real)

            mcwd_crit.append(np.amax(mcwd_real))
        mcwd_crit = np.array(mcwd_crit)

        #here also absolute limits can be set
        global_limit_mcwd = 0.0
        mcwd_crit[mcwd_crit < global_limit_mcwd] = global_limit_mcwd
        return mcwd_crit        
      


    # c = c(rainfall) where tipping occurs at sqrt((4*b**3)/(27*a)) ~ 0.38/sqrt(b**3/a)
    # Linear function through two points maps rainfall --> c, where x-values represent rainfall values and y-values represent CUSP-c values
    def Amazon_CUSPc(a, b, rain_mean, rain_critical, rain_current, mcwd_mean, mcwd_critical, mcwd_current):
        # only returns a value in case the rainfall is higher than the mean rainfall, 
        # otherwise return 0.0 (lower cap), N.B.: No upper cap
        if mcwd_critical < mcwd_mean:
            print("""Error: The mean seasonality value is below the average.
                This does not make sense since on average the Amazon rainforest would be tipped - this is not what is observed...""")
            die

        #critical ground truth values for rain and mcwd
        #for c_rain direct computation - no exceptions need to be defined
        c_rain = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(rain_critical - rain_mean)*(rain_current - rain_mean)
        #for mcwd it can be the case that this cell has zero mcwd each year which results in: mcwd_critical = 0.0 and mcwd_mean = 0.0
        #this exception must be handled here
        if mcwd_critical == 0.0 and mcwd_mean == 0.0:
            c_mcwd = 0.0
        else:
            c_mcwd = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(mcwd_critical - mcwd_mean)*(mcwd_current - mcwd_mean)


        #case distinction to find "correct" criitcal value c
        if 0 < c_rain < np.sqrt(4/27) and 0 < c_mcwd < np.sqrt(4/27):
            c = np.amax([c_rain, c_mcwd]) + (np.sqrt(4/27) - np.amax([c_rain, c_mcwd]))/(np.sqrt(4/27))*np.amin([c_rain, c_mcwd])
        else:
            c = np.amax([c_rain, c_mcwd])

        return c


    def Amazon_cpl(a, b, rain_mean, rain_critical, rain_current, delta_rain, mcwd_mean, mcwd_critical, mcwd_current, delta_mcwd):
        """
        - Needs to be additive to Amazon CUSPc
        - Factor of (1/2) is necessary since delta(state)=2 from -1 to +1 
            opposing to the climate system where this can be included into the coupling constant d 
        - Factor (-1) comes from the negative delta_rain/(mcwd_mean-mcwd_critical) since LESS rain reaches the receiveing cell
        """

        #rain [c, cpl]-values
        c_rain = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(rain_critical - rain_mean)*(rain_current - rain_mean)
        cpl_rain = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(rain_mean - rain_critical)*(1/2)*(-1)*delta_rain

        #mcwd [c, cpl]-values
        #the same exception for c_mcwd and cpl_mcwd must be handled as in the function Amazon_CUSPc()
        if mcwd_critical == 0.0 and mcwd_mean == 0.0:
            c_mcwd = 0.0
            cpl_mcwd = 0.0 
            #this could also be defined in another way since the resilience of this cell should be very low. Now it is assumed to be high
            #another solution would be to set cpl_mcwd to np.sqrt(4/27)
        else:
            c_mcwd = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(mcwd_critical - mcwd_mean)*(mcwd_current - mcwd_mean)
            cpl_mcwd = np.sqrt((4*np.abs(b)**3) / (27*np.abs(a)))/(mcwd_mean - mcwd_critical)*(1/2)*(-1)*delta_mcwd


        #computation of coupling - similar to computation of c
        if 0 < c_rain + cpl_rain < np.sqrt(4/27) and 0 < c_mcwd + cpl_mcwd < np.sqrt(4/27):
            index = np.argmax([c_rain, c_mcwd])
            if index  == 0:
                cpl = cpl_rain + (np.sqrt(4/27) - cpl_rain)/(np.sqrt(4/27))*cpl_mcwd
            elif index == 1:
                cpl = cpl_mcwd + (np.sqrt(4/27) - cpl_mcwd)/(np.sqrt(4/27))*cpl_rain
            else:
                print("Wrong index!!")
                die
        else:
            index = np.argmax([c_rain, c_mcwd])
            if index == 0:
                cpl = cpl_rain
            elif index == 1:
                cpl = cpl_mcwd
            else:
                print("Wrong index!!")
                die

        if cpl < 0.0:
            print("Coupling strengths below 0.0 are not allowed")
            die
    
        return cpl



    def Rain_moisture_delta_only(moist_rec_val):
        """
        Computation of moisture recycling value for a year
        """
        rain_moist = - np.sum(moist_rec_val)
        return rain_moist


    def Mcwd_moisture(datasets, rain_fact, id_receiving_cell, moist_rec_val):
        """
        Computation of MCWD with respect to the moisture value of the receiving cell;
        For this computation, we additionally need the moisture recycling value of the receiving cell (moist_rec_val);
        For faster computation, we are only interested in the receving cell, therefore we need the id of the receiving cell (id_receiving_cell)
        to not be forced to compute the whole mcwd network
        """
        mcwd_array = []
        for i in range(0, len(datasets)):
            net_data = Dataset(datasets[i])
            #get mean value for rainfall
            rain = np.multiply(rain_fact, net_data.variables["rain"][id_receiving_cell])
            evap = net_data.variables["evap"][id_receiving_cell]

            #computation of MCWD
            diff = np.subtract(evap, np.subtract(rain, moist_rec_val[i]))
            mcwd_array.append(diff)
        mcwd_array = np.array(mcwd_array)
        mcwd_sign = np.sign(mcwd_array)


        mcwd_real = []
        diff = 0.
        for j in range(0, len(mcwd_array)):
            if mcwd_sign[j] < 0.:
                diff = 0.
            else:
                diff += mcwd_array[j]
            mcwd_real.append(diff)
        mcwd_real = np.array(mcwd_real)
        

        mcwd = np.amax(mcwd_real)
        return mcwd