import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
import matplotlib.pyplot as plt
from pipedream_solver.simulation import Simulation
from pipedream_solver.wq_reaction import WQ_reaction
import matplotlib.colors
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import cm
from collections import defaultdict
sns.set_palette('viridis')

###############################################################################
#   Initializing the model
###############################################################################
# Load the model network and parameter files
input_path = '../data/multi_WQ'
input_file_prefix = 'Input_C_'
measurement_file_prefix = 'Obs_C_'
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')

# Make a list for constituent names for water quality model
name = ['Temp', 'DO', 'DIC','LDOC', 'LPOC', 'RDOC', 'RPOC', 'PIN', 'NH4', 'NO3', 
         'LDON', 'LPON', 'RDON', 'RPON', 'PIP','PO4', 'LDOP', 'LPOP', 'RDOP', 'RPOP',
         'ALG1', 'ALG2', 'ALG3', 'ALG4', 'ISS']

# Make a list for measurement constituents for the data assimilation(Kalman filter)
obs_name = ['Temp', 'DO', 'DIC','LDOC', 'LPOC', 'RDOC', 'RPOC', 'PIN', 'NH4', 'NO3', 
         'LDON', 'LPON', 'RDON', 'RPON', 'PIP','PO4', 'LDOP', 'LPOP', 'RDOP', 'RPOP',
         'ALG1', 'ALG2', 'ALG3', 'ALG4', 'ISS']

# Load the hydraulic time series data
Q_in_file = pd.read_csv(f'{input_path}/Input_Flow.csv', index_col=0)
#H_bc = pd.read_csv(f'{input_path}/boundary_stage.csv', index_col=0)

# Load the waqter quality time series data
input_data = {}
for i in range(0,len(name)):
    input_data[name[i]] = pd.read_csv(f'{input_path}/{input_file_prefix+name[i]}.csv', index_col=0)

# Load the meteorological time series data
MET_data = pd.read_csv(f'{input_path}/Input_MET.csv')

# Load the measurement data for the Kalman Filtering
obs_data = {}
for i in range(0,len(obs_name)):
    obs_data[obs_name[i]] = pd.read_csv(f'{input_path}/{measurement_file_prefix+obs_name[i]}.csv', index_col=0)
    obs_data[obs_name[i]] = pd.DataFrame(obs_data[obs_name[i]]).to_numpy()

# Make instances for the hydraulics & WQ transport: 600sec & 5,10 internal: 40sec for 20 days modeling, 20 internal is unstable.
# When the Kalman Filter is applied: 40sec for 5 days modeling. becomes about 4 times slower
dt = 600
internal_links = 10
SL = SuperLink(superlinks, superjunctions,internal_links=internal_links,
                     bc_method = 'b', min_depth = 10e-3, mobile_elements=True)

# Kalman filter On = 1/Off = 0 or other number
KF = 1

# output options
screen_output = 1 # print the results in screen when t = 1 at the superjunction 5, 11, 17
out_let = 19 # outlet superjunction number for network plot
out_name = ['Temp', 'ALG1', 'PO4', 'NH4', 'NO3', 'DO', 'ISS'] # constituents list for network plot

# Make the WQ reaction instance
WQ_R = WQ_reaction(SL, superjunction_params=superjunction_wq_params,
                   superlink_params=superlink_wq_params, N_constituent=len(name))

# Make the water quality transport instances and apply the initial values
WQ = {}
out_H_j = []
out_c_j = defaultdict(list)
out_c_Ik = defaultdict(list)
out_c_ik = defaultdict(list)
WQ_ini = WQ_R.initial_value()
for i in range(0,len(name)):
    WQ[name[i]] = QualityBuilder(SL, superjunction_params=superjunction_wq_params,
                                 superlink_params=superlink_wq_params)
    zero_initial = 0  # if this value = 1, we can apply the given initial values in wq_reaction.py.
    WQ[name[i]]._c_j = zero_initial*WQ_ini[name[i]]*np.ones(len(WQ[name[i]].c_j))
    WQ[name[i]]._c_Ik = zero_initial*WQ_ini[name[i]]*np.ones(len(WQ[name[i]].c_Ik))
    WQ[name[i]]._c_ik = zero_initial*WQ_ini[name[i]]*np.ones(len(WQ[name[i]].c_ik))
    WQ[name[i]]._c_uk = zero_initial*WQ_ini[name[i]]*np.ones(len(WQ[name[i]].c_uk))
    WQ[name[i]]._c_dk = zero_initial*WQ_ini[name[i]]*np.ones(len(WQ[name[i]].c_dk))
    out_c_j[name[i]].append(WQ[name[i]].c_j)
    out_c_Ik[name[i]].append(WQ[name[i]].c_Ik)
    out_c_ik[name[i]].append(WQ[name[i]].c_ik)

# Variables for the Kalman filtering
WQ_A_k = {}
WQ_B_k = {}

### Visualize the network structure ###
fig, ax = plt.subplots(figsize=(12, 18))
_ = SL.plot_network_2d(ax=ax, junction_kwargs={'s' : 5},
                       superjunction_kwargs={'c' : '0.25'},
                       link_kwargs={'color' : '0.5'})

### Setup end time and initialize the step number 'T'.
t_end = Q_in_file.index[-1]
t_end = 86400*2
T = 0

# Call the numbers of computational elements
N_j = len(WQ['Temp']._c_j)
N_Ik = len(WQ['Temp']._c_Ik)
N_ik = len(WQ['Temp']._c_ik)
N_uk = len(WQ['Temp']._c_uk)
N_dk = len(WQ['Temp']._c_dk)

# Load the correlations for the process noise covariance matrix
# Basically, they are identity matrices. (spatially independent)
Q_spatial = {}
for i in range(0,len(name)):
    Q_spatial[name[i]] = np.eye(N_j,N_j)
    
# If there is additional information, we can apply it.
test_name = ['PO4']
input_pre = 'Q_spatial_'
for i in range(0,len(test_name)):
    Q_spatial[test_name[i]] = pd.read_csv(f'{input_path}/{input_pre+test_name[i]}.csv', index_col=0)
    Q_spatial[test_name[i]] = np.nan_to_num(Q_spatial[test_name[i]]) + np.nan_to_num(Q_spatial[test_name[i]].T) - np.diag(np.diag(Q_spatial[test_name[i]]))
            
# Basically, there are no correlations between water quality species
Q_species = {}
for i in range(0,N_j):
    Q_species[i] = np.zeros((len(name),len(name)))
    
# if there is addional information, we can apply it.
Q_species[14] = pd.read_csv(f'{input_path}/Q_species_sj14.csv', index_col=0)
Q_species[14] = np.nan_to_num(Q_species[14]) + np.nan_to_num(Q_species[14].T) - np.diag(np.diag(Q_species[14]))

# Spin-up hydraulic model to avoid running dry
SL.reposition_junctions()
SL.spinup(n_steps=100)

WQ_R.make_QR_cov(Q_spatial, Q_species)

###############################################################################
#   Simulate unsteady hydraulics, WQ transport, Kalman Filter, and WQ reactions
###############################################################################
with Simulation(SL, dt=dt, t_end=t_end, 
                interpolation_method='linear') as simulation:
    coeffs = simulation.h0321
    
    while SL.t < t_end + 0.001:
        # Calculate the 'Hydraulics'
        Q_in = Q_in_file.iloc[T][:]
        Q_in = np.array(Q_in)
        simulation.step(dt=dt,Q_in = Q_in)
        simulation.model.reposition_junctions()
        
        # Calculate the WQ 'Transport'
        for i in range(0,len(name)):
            c_0j = input_data[name[i]].iloc[T][:]
            c_0j = np.array(c_0j)
            WQ[name[i]].step(dt = dt, c_0j = c_0j)
        
        # Kalman Filtering and Data Assimilation process
        if KF == 1:
            # Calculate the A_k and B_k matrices for the Kalman Filtering
            for i in range(0,len(name)):
                A_k, B_k = WQ[name[i]].KF_Multi_WQ(dt)
                WQ_A_k[name[i]] = A_k
                WQ_B_k[name[i]] = B_k
            # Apply the Kalman Filering and data assimilation
            x_hat = WQ_R.KF_DA(dt, N_j, WQ_A_k, WQ_B_k, WQ, obs_data, T, name, Q_spatial, Q_species)
            # Data assimilation: x_hat(estimation) to WQ[name[i]]._c_j
            for i in range (0,len(name)):
                WQ[name[i]]._c_j = x_hat[i*N_j:(i+1)*N_j]
            # Recalculate the all internals after KF and DA(update)
            for i in range(0,len(name)):
                WQ[name[i]].solve_boundary_states()
                WQ[name[i]].solve_internals_backwards()
        
        # Calculate the WQ 'Reactions'
        WQ_R.calc_hyd_para(dt, name, WQ)
        WQ_R.WQ_reactions(dt, WQ, T, MET_data)
            
        # Save the reaction results to the WQ instances
        for i in range (0,len(name)):
            WQ[name[i]]._c_j = WQ_R.C_all[name[i]][0:N_j]
            WQ[name[i]]._c_Ik = WQ_R.C_all[name[i]][N_j:N_j+N_Ik]
            WQ[name[i]]._c_ik = WQ_R.C_all[name[i]][N_j+N_Ik:N_j+N_Ik+N_ik]
            WQ[name[i]]._c_uk = WQ_R.C_all[name[i]][N_j+N_Ik+N_ik:N_j+N_Ik+N_ik+N_uk]
            WQ[name[i]]._c_dk = WQ_R.C_all[name[i]][N_j+N_Ik+N_ik+N_uk:N_j+N_Ik+N_ik+N_uk+N_dk]    

        if screen_output == 1:
            print("-------------------------")
            print(f'Day = {T*dt/86400:.3f}')
            print("-------------------------")
            for i in range(0,len(name)):
                print(f'{name[i]} \t = {WQ[name[i]].c_j[5] : 3.8f} {WQ[name[i]].c_j[11] : 3.8f} {WQ[name[i]].c_j[17] : 3.8f}')    
        #simulation.record_state()
        simulation.print_progress()

        # save results for plot
        for i in range(0,len(name)):
            out_c_j[name[i]].append(WQ[name[i]].c_j.copy())
            out_c_Ik[name[i]].append(WQ[name[i]].c_Ik.copy())
            out_c_ik[name[i]].append(WQ[name[i]].c_ik.copy())

        out_H_j.append(SL.H_j.copy())
        T += 1

###############################################################################        
#   Plot the outputs
###############################################################################
# 
out_ALG1 = pd.DataFrame(out_c_j['ALG1']) 
plt.figure(1)
plt.figure(figsize= (12,8))
plt.plot(out_ALG1.loc[:, 1:18])
plt.title('Algae 1')
plt.ylabel('Algae (mg/L)')
plt.ylim(0,0.3)
plt.xlabel(f'Time(N time steps) * {dt}sec')

out_PO4 = pd.DataFrame(out_c_j['PO4']) 
plt.figure(2)
plt.figure(figsize= (12,8))
plt.plot(out_PO4.loc[:, 1:18])
plt.title('PO4')
plt.ylabel('PO4 (mg/L)')
plt.ylim(0,0.07)
plt.xlabel(f'Time(N time steps) * {dt}sec')

out_NH4 = pd.DataFrame(out_c_j['NH4']) 
plt.figure(3)
plt.figure(figsize= (12,8))
plt.plot(out_NH4.loc[:, 1:18])
plt.title('NH4')
plt.ylabel('NH4 (mg/L)')
plt.ylim(0,0.1)
plt.xlabel(f'Time(N time steps) * {dt}sec')

out_NO3 = pd.DataFrame(out_c_j['NO3']) 
plt.figure(4)
plt.figure(figsize= (12,8))
plt.plot(out_NO3.loc[:, 1:18])
plt.title('NO3')
plt.ylabel('NO3 (mg/L)')
plt.ylim(0,2)
plt.xlabel(f'Time(N time steps) * {dt}sec')

plt.figure(5)
plt.figure(figsize= (12,8))
plt.plot(out_ALG1.loc[T-1, 1:18])
plt.title('Algae 1')
plt.ylabel('Algae (mg/L)')
plt.ylim(0.15,0.25)
plt.xlabel('Superjunction number')

plt.figure(6)
plt.figure(figsize= (12,8))
plt.plot(out_PO4.loc[T-1, 1:18])
plt.title('PO4')
plt.ylabel('PO4 (mg/L)')
plt.ylim(0,0.05)
plt.xlabel('Superjunction number')

plt.figure(7)
plt.figure(figsize= (12,8))
plt.plot(out_NH4.loc[T-1, 1:18])
plt.title('NH4')
plt.ylabel('NH4 (mg/L)')
plt.ylim(0.04,0.06)
plt.xlabel('Superjunction number')

plt.figure(8)
plt.figure(figsize= (12,8))
plt.plot(out_NO3.loc[T-1, 1:18])
plt.title('NO3')
plt.ylabel('NO3 (mg/L)')
plt.ylim(1.4,1.6)
plt.xlabel('Superjunction number')

out_ALG1 = pd.DataFrame(out_c_j['ISS']) 
plt.figure(9)
plt.figure(figsize= (12,8))
plt.plot(out_ALG1.loc[:, 1:18])
plt.title('Inorganic Suspended Solids')
plt.ylabel('ISS (mg/L)')
plt.ylim(0,25)
plt.xlabel(f'Time(N time steps) * {dt}sec')

# Plot the network output during last 10 time steps: Temperature example
for k in range(0,len(out_name)):
    for j in range(T,T+1):
        superjunction_data = pd.DataFrame(out_c_j[out_name[k]])
        link_data = pd.DataFrame(out_c_ik[out_name[k]])
        junction_data = pd.DataFrame(out_c_Ik[out_name[k]])
    
        outlet = tuple(superjunctions.loc[out_let, ['map_x', 'map_y']].values.ravel().astype(float))
        norm = mcolors.Normalize(vmin=min(link_data.loc[j]), vmax=max(link_data.loc[j]))
        m = cm.ScalarMappable(norm=norm, cmap='jet')
        
        link_slice = link_data.loc[j].values
        superjunction_slice = superjunction_data.loc[j].values
        
        link_colors = [m.to_rgba(i) for i in link_slice]
        superjunction_colors = [m.to_rgba(i) for i in superjunction_slice]
        
        fig, ax = plt.subplots(figsize=(12,18))
        SL.plot_network_2d(ax=ax, junction_kwargs={'s' : 0}, 
                           link_kwargs={'colors' : link_colors, 'linewidths' : (5,)}, 
                           superjunction_kwargs={'c' : superjunction_colors})
        day = (T-1)*600/86400
        plt.annotate(f'{out_name[k]}'+': t = {:2.4}day'.format(day), (0.1, 0.1),
                     xycoords='subfigure fraction', fontsize=20)
        #type = day
        #plt.savefig("{0}.png".format(type))
        plt.scatter(outlet[0], outlet[1], s=200, c='w', zorder=4)
        plt.axis('off')
