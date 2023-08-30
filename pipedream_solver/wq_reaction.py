# Definition of the reaction functions for Pipedream-WQ
from numba import njit
import numpy as np

class WQ_reaction():
    def __init__(self, hydraulics, superjunction_params, superlink_params, N_constituent,
                 junction_params=None, link_params=None):
        self.hydraulics = hydraulics
        ###### WQ parameters and coefficients #########################################
        # All reaction coefficients have unit of "1/day".
        # This part should be moved to the nquality.py with parameter input from the file.
        self.gravity = 9.81
        self.K_CO2 = 0.00
        self.K_SOD = 0.00
        self.CO2_air = 0.0   # We need a conversion factor between concentration of CO2 in air and DIC in water

        self.K_LDOC = 0.000
        self.K_LPOC = 0.00
        self.K_RDOC = 0.000
        self.K_RPOC = 0.000
        self.K_LDOC_to_RDOC = 0.00
        self.K_LPOC_to_RPOC = 0.00
        self.K_s_LPOC = 0.000
        self.K_s_RPOC = 0.000

        self.K_LDON = 0.000
        self.K_LPON = 0.000
        self.K_RDON = 0.000
        self.K_RPON = 0.000
        self.K_LDON_to_RDON = 0.00
        self.K_LPON_to_RPON = 0.00
        self.K_s_LPON = 0.000
        self.K_s_RPON = 0.000

        self.K_LDOP = 0.000
        self.K_LPOP = 0.000
        self.K_RDOP = 0.000
        self.K_RPOP = 0.000
        self.K_LDOP_to_RDOP = 0.000
        self.K_LPOP_to_RPOP = 0.000
        self.K_s_LPOP = 0.000
        self.K_s_RPOP = 0.000

        self.K_NH4 = 0.000
        self.K_NO3 = 0.000
        self.K_s_NO3 = 0.000
        self.K_s_PIN = 0.000
        self.P_d_N = 0.5
        self.P_d_P = 0.5
        self.P_d_C = 0.5

        self.K_ads_PIP = 0.000
        self.K_des_PIP = 0.000
        self.K_ads_PIN = 0.000
        self.K_des_PIN = 0.000
        self.K_s_PIP = 0.000
        self.K_s_PIN = 0.000
        self.K_s_ISS = 0.0
        self.K_s_ALG1 = 0.0
        self.K_s_ALG2 = 0.0
        self.K_s_ALG3 = 0.0
        self.K_s_ALG4 = 0.0

        self.K_h_N = 0.5
        self.K_h_NH4 = 0.2
        self.K_h_P = 0.05
        self.K_h_PIP = 0.2
        self.K_h_PIN = 0.5

        self.delta_O_ALG1_ag = 1.4
        self.delta_O_ALG2_ag = 1.4
        self.delta_O_ALG3_ag = 1.4
        self.delta_O_ALG4_ag = 1.4

        self.delta_O_ALG1_ar = 1.1
        self.delta_O_ALG2_ar = 1.1
        self.delta_O_ALG3_ar = 1.1
        self.delta_O_ALG4_ar = 1.1

        self.delta_C_ALG1_ag = 1.4
        self.delta_C_ALG2_ag = 1.4
        self.delta_C_ALG3_ag = 1.4
        self.delta_C_ALG4_ag = 1.4

        self.delta_C_ALG1_ar = 1.1
        self.delta_C_ALG2_ar = 1.1
        self.delta_C_ALG3_ar = 1.1
        self.delta_C_ALG4_ar = 1.1

        self.delta_C_ALG1_ae = 1.4
        self.delta_C_ALG2_ae = 1.4
        self.delta_C_ALG3_ae = 1.4
        self.delta_C_ALG4_ae = 1.4

        self.delta_C_ALG1_am = 1.1
        self.delta_C_ALG2_am = 1.1
        self.delta_C_ALG3_am = 1.1
        self.delta_C_ALG4_am = 1.1

        self.delta_N_ALG1_ag = 1.4
        self.delta_N_ALG2_ag = 1.4
        self.delta_N_ALG3_ag = 1.4
        self.delta_N_ALG4_ag = 1.4

        self.delta_N_ALG1_ar = 1.1
        self.delta_N_ALG2_ar = 1.1
        self.delta_N_ALG3_ar = 1.1
        self.delta_N_ALG4_ar = 1.1

        self.delta_N_ALG1_ae = 1.4
        self.delta_N_ALG2_ae = 1.4
        self.delta_N_ALG3_ae = 1.4
        self.delta_N_ALG4_ae = 1.4

        self.delta_P_ALG1_am = 1.1
        self.delta_P_ALG2_am = 1.1
        self.delta_P_ALG3_am = 1.1
        self.delta_P_ALG4_am = 1.1
        
        self.delta_P_ALG1_ag = 0.05
        self.delta_P_ALG2_ag = 0.05
        self.delta_P_ALG3_ag = 0.05
        self.delta_P_ALG4_ag = 0.05

        self.delta_P_ALG1_ar = 1.1
        self.delta_P_ALG2_ar = 1.1
        self.delta_P_ALG3_ar = 1.1
        self.delta_P_ALG4_ar = 1.1

        self.delta_P_ALG1_ae = 1.4
        self.delta_P_ALG2_ae = 1.4
        self.delta_P_ALG3_ae = 1.4
        self.delta_P_ALG4_ae = 1.4

        self.delta_N_ALG1_am = 1.1
        self.delta_N_ALG2_am = 1.1
        self.delta_N_ALG3_am = 1.1
        self.delta_N_ALG4_am = 1.1

        self.delta_N_ALG1 = 0.0
        self.delta_N_ALG2 = 0.0
        self.delta_N_ALG3 = 0.0
        self.delta_N_ALG4 = 0.0

        self.delta_P_to_ALG1 = 0.0
        self.delta_P_to_ALG2 = 0.0
        self.delta_P_to_ALG3 = 0.0
        self.delta_P_to_ALG4 = 0.0

        self.delta_O_OC = 2.67
        self.delta_C_OC = 0.50  # temporary value
        self.delta_P_OP = 0.50  # temporary value
        self.delta_N_ON = 0.50  # temporary value
        self.delta_O_NH4 = 4.57
        self.delta_N_to_OC = 0.0

        # temperature modeling
        self.T_si = 15 # annual average temperature of riverbed

        # temperature multipliers
        self.K1_OC = 0.01
        self.K2_OC = 0.99
        self.T1_OC = 0.01
        self.T2_OC = 25
        self.gamma_OC_1 = (1/(self.T2_OC - self.T1_OC))*np.log((self.K2_OC*(1-self.K1_OC))/(self.K1_OC*(1-self.K2_OC)))

        self.K1_ON = 0.01
        self.K2_ON = 0.99
        self.T1_ON = 0.01
        self.T2_ON = 25
        self.gamma_ON_1 = (1/(self.T2_ON - self.T1_ON))*np.log((self.K2_ON*(1-self.K1_ON))/(self.K1_ON*(1-self.K2_ON)))

        self.K1_OP = 0.01
        self.K2_OP = 0.99
        self.T1_OP = 0.01
        self.T2_OP = 25
        self.gamma_OP_1 = (1/(self.T2_OP - self.T1_OP))*np.log((self.K2_OP*(1-self.K1_OP))/(self.K1_OP*(1-self.K2_OP)))

        self.K1_NH4 = 0.01
        self.K2_NH4 = 0.99
        self.T1_NH4 = 0.01
        self.T2_NH4 = 25
        self.gamma_NH4_1 = (1/(self.T2_NH4 - self.T1_NH4))*np.log((self.K2_NH4*(1-self.K1_NH4))/(self.K1_NH4*(1-self.K2_NH4)))

        self.K1_NO3 = 0.01
        self.K2_NO3 = 0.99
        self.T1_NO3 = 0.01
        self.T2_NO3 = 25
        self.gamma_NO3_1 = (1/(self.T2_NO3 - self.T1_NO3))*np.log((self.K2_NO3*(1-self.K1_NO3))/(self.K1_NO3*(1-self.K2_NO3)))
        
        self.K_ag_max_ALG1 = 0.5
        self.K_ar_max_ALG1 = 0.0
        self.K_am_max_ALG1 = 0.0
        self.K_ae_max_ALG1 = 0.02

        self.K_ag_max_ALG2 = 0.3
        self.K_ar_max_ALG2 = 0.0
        self.K_am_max_ALG2 = 0.0
        self.K_ae_max_ALG2 = 0.02

        self.K_ag_max_ALG3 = 0.7
        self.K_ar_max_ALG3 = 0.0
        self.K_am_max_ALG3 = 0.0
        self.K_ae_max_ALG3 = 0.02

        self.K_ag_max_ALG4 = 0.4
        self.K_ar_max_ALG4 = 0.0
        self.K_am_max_ALG4 = 0.0
        self.K_ae_max_ALG4 = 0.02

        self.K1_ALG1 = 0.05
        self.K2_ALG1 = 0.99
        self.K3_ALG1 = 0.99
        self.K4_ALG1 = 0.05
        self.T1_ALG1 = 5.0
        self.T2_ALG1 = 15.0
        self.T3_ALG1 = 25.0
        self.T4_ALG1 = 35.0
        self.gamma1_ALG1 = (1/(self.T2_ALG1-self.T1_ALG1))*np.log((self.K2_ALG1*(1-self.K1_ALG1))/(self.K1_ALG1*(1-self.K2_ALG1)))
        self.gamma2_ALG1 = (1/(self.T4_ALG1-self.T3_ALG1))*np.log((self.K3_ALG1*(1-self.K4_ALG1))/(self.K4_ALG1*(1-self.K3_ALG1)))

        self.K1_ALG2 = 0.01
        self.K2_ALG2 = 0.99
        self.K3_ALG2 = 0.99
        self.K4_ALG2 = 0.05
        self.T1_ALG2 = 5.0
        self.T2_ALG2 = 15.0
        self.T3_ALG2 = 20.0
        self.T4_ALG2 = 35.0
        self.gamma1_ALG2 = (1/(self.T2_ALG2-self.T1_ALG2))*np.log((self.K2_ALG2*(1-self.K1_ALG2))/(self.K1_ALG2*(1-self.K2_ALG2)))
        self.gamma2_ALG2 = (1/(self.T4_ALG2-self.T3_ALG2))*np.log((self.K3_ALG2*(1-self.K4_ALG2))/(self.K4_ALG2*(1-self.K3_ALG2)))

        self.K1_ALG3 = 0.05
        self.K2_ALG3 = 0.99
        self.K3_ALG3 = 0.99
        self.K4_ALG3 = 0.05
        self.T1_ALG3 = 5.0
        self.T2_ALG3 = 15.0
        self.T3_ALG3 = 25.0
        self.T4_ALG3 = 35.0
        self.gamma1_ALG3 = (1/(self.T2_ALG3-self.T1_ALG3))*np.log((self.K2_ALG3*(1-self.K1_ALG3))/(self.K1_ALG3*(1-self.K2_ALG3)))
        self.gamma2_ALG3 = (1/(self.T4_ALG3-self.T3_ALG3))*np.log((self.K3_ALG3*(1-self.K4_ALG3))/(self.K4_ALG3*(1-self.K3_ALG3)))

        self.K1_ALG4 = 0.05
        self.K2_ALG4 = 0.99
        self.K3_ALG4 = 0.99
        self.K4_ALG4 = 0.05
        self.T1_ALG4 = 5.0
        self.T2_ALG4 = 15.0
        self.T3_ALG4 = 25.0
        self.T4_ALG4 = 35.0
        self.gamma1_ALG4 = (1/(self.T2_ALG4-self.T1_ALG4))*np.log((self.K2_ALG4*(1-self.K1_ALG4))/(self.K1_ALG4*(1-self.K2_ALG4)))
        self.gamma2_ALG4 = (1/(self.T4_ALG4-self.T3_ALG4))*np.log((self.K3_ALG4*(1-self.K4_ALG4))/(self.K4_ALG4*(1-self.K3_ALG4)))
        
        self.keb = 0.05
        self.alpha_ISS = 0.01
        self.alpha_POM = 0.4
        self.alpha_ALG = 0.1
        self.K_Lp = 70
        
        # Kalman Filter: measure and process noise(sigma)
        self.N_measure_sigma = 0.05
        self.N_process_sigma = 0.2
        
        self.P_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent))
        self.K_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent))
        self.H_k = np.zeros((self.hydraulics.M*N_constituent,self.hydraulics.M*N_constituent))
        self.N_constituent = N_constituent
        
    def initial_value(self):
        WQ_ini = {}
        WQ_ini['Temp'] = 24.0
        WQ_ini['DO'] = 8.0
        WQ_ini['DIC'] = 0.5
        WQ_ini['LDOC'] = 0.5
        WQ_ini['LPOC'] = 0.6
        WQ_ini['RDOC'] = 1.1
        WQ_ini['RPOC'] = 0.3
        WQ_ini['PIN'] = 0.19
        WQ_ini['NH4'] = 0.05
        WQ_ini['NO3'] = 1.5
        WQ_ini['LDON'] = 0.55
        WQ_ini['LPON'] = 0.40
        WQ_ini['RDON'] = 0.7
        WQ_ini['RPON'] = 0.005
        WQ_ini['PIP'] = 0.011
        WQ_ini['PO4'] = 0.025
        WQ_ini['LDOP'] = 0.02
        WQ_ini['LPOP'] = 0.015
        WQ_ini['RDOP'] = 0.005
        WQ_ini['RPOP'] = 0.05
        WQ_ini['ALG1'] = 0.2
        WQ_ini['ALG2'] = 0.3
        WQ_ini['ALG3'] = 0.05
        WQ_ini['ALG4'] = 0.02
        WQ_ini['ISS'] = 10.0
        return WQ_ini
        
    def calc_hyd_para(self, dt, name, WQ):
        SL = self.hydraulics
        # Merging the concentration of all elements for WQ reaction calculation
        self.C_all = {}
        for i in range(0,len(name)):
            self.C_all[name[i]] = np.concatenate([WQ[name[i]].c_j, WQ[name[i]].c_Ik,
                           WQ[name[i]].c_ik, WQ[name[i]].c_uk, WQ[name[i]].c_dk])
        
        # Calculate the hydraulic parameters for all elements
        self.A_sur = np.concatenate([SL.A_sj, SL._A_SIk, SL.B_ik*SL._dx_ik,
                                SL._B_uk*SL._dx_uk, SL._B_dk*SL._dx_dk])
        self.A_bot = self.A_sur
        H_dep = np.concatenate([SL.H_j-SL._z_inv_j, SL._h_Ik, 
                                SL.A_ik/SL._B_ik, SL._h_uk, SL._h_dk])
        U_vel = np.concatenate([0*SL.H_j, 0*SL._h_Ik, SL.Q_ik/SL.A_ik,
                                SL.Q_uk/SL.A_uk, SL.Q_dk/SL.A_dk])
        self.H_dep = np.where(H_dep<0.1,0.1,H_dep)
        self.U_vel = np.where(U_vel<0.0001,0.0001,U_vel)
        
    def WQ_reactions(self, dt, WQ, T, MET_data):
        
        C_all = self.C_all  # Loading the concentrations at the previous time step
        MET_atemp = MET_data['atemp'].values.astype(np.float64)
        MET_dtemp = MET_data['dtemp'].values.astype(np.float64)
        MET_windspd = MET_data['windspd'].values.astype(np.float64)
        MET_solar = MET_data['solrad'].values.astype(np.float64)    
        H_dep = self.H_dep

        C_new = {}
        ###################################################################
        #   Temperature
        ###################################################################
        
        C_new['Temp'] = RK4_Temp(dt, C_all['Temp'], MET_solar[T], 
                   MET_atemp[T],MET_dtemp[T], MET_windspd[T], self.T_si, H_dep)
       
        # Temperature multipilers : SHOULD BE MOVED TO the wq_reaction.py FILE
        gamma_OC = ( (self.K1_OC*np.exp(self.gamma_OC_1*(C_all['Temp']-self.T1_OC))) 
                    /(1+self.K1_OC*np.exp(self.gamma_OC_1*(C_all['Temp']-self.T1_OC))-self.K1_OC) )
        gamma_ON = ( (self.K1_ON*np.exp(self.gamma_ON_1*(C_all['Temp']-self.T1_ON)))
                    /(1+self.K1_ON*np.exp(self.gamma_ON_1*(C_all['Temp']-self.T1_ON))-self.K1_ON) )
        gamma_OP = ( (self.K1_OP*np.exp(self.gamma_OP_1*(C_all['Temp']-self.T1_OP)))
                    /(1+self.K1_OP*np.exp(self.gamma_OP_1*(C_all['Temp']-self.T1_OP))-self.K1_OP) )
        gamma_NH4 =  ( (self.K1_NH4*np.exp(self.gamma_NH4_1*(C_all['Temp']-self.T1_NH4)))
                      /(1+self.K1_NH4*np.exp(self.gamma_NH4_1*(C_all['Temp']-self.T1_NH4))-self.K1_NH4) )
        gamma_NO3 =  ( (self.K1_NO3*np.exp(self.gamma_NO3_1*(C_all['Temp']-self.T1_NO3)))
                      /(1+self.K1_NO3*np.exp(self.gamma_NO3_1*(C_all['Temp']-self.T1_NO3))-self.K1_NO3) )
        
        ###################################################################
        #   Algae(Phytoplankton)
        ###################################################################
        # Temperature multipliers 
        gamma_r_ALG1 = ((self.K1_ALG1*np.exp(self.gamma1_ALG1*(C_all['Temp'] - self.T1_ALG1)))
                    /(1+self.K1_ALG1*np.exp(self.gamma1_ALG1*(C_all['Temp'] - self.T1_ALG1))-self.K1_ALG1) )
        gamma_f_ALG1 = ((self.K4_ALG1*np.exp(self.gamma2_ALG1*(self.T4_ALG1 - C_all['Temp'])))
                    /(1+self.K4_ALG1*np.exp(self.gamma2_ALG1*(self.T4_ALG1 - C_all['Temp']))-self.K4_ALG1) )
        gamma_ALG1 = gamma_r_ALG1*gamma_f_ALG1
        
        gamma_r_ALG2 = ((self.K1_ALG2*np.exp(self.gamma1_ALG2*(C_all['Temp'] - self.T1_ALG2)))
                    /(1+self.K1_ALG2*np.exp(self.gamma1_ALG2*(C_all['Temp'] - self.T1_ALG2))-self.K1_ALG2) )
        gamma_f_ALG2 = ((self.K4_ALG2*np.exp(self.gamma2_ALG2*(self.T4_ALG2 - C_all['Temp'])))
                    /(1+self.K4_ALG2*np.exp(self.gamma2_ALG2*(self.T4_ALG2 - C_all['Temp']))-self.K4_ALG2) )
        gamma_ALG2 = gamma_r_ALG2*gamma_f_ALG2
        
        
        gamma_r_ALG3 = ((self.K1_ALG3*np.exp(self.gamma1_ALG3*(C_all['Temp'] - self.T1_ALG3)))
                    /(1+self.K1_ALG3*np.exp(self.gamma1_ALG3*(C_all['Temp'] - self.T1_ALG3))-self.K1_ALG3) )
        gamma_f_ALG3 = ((self.K4_ALG3*np.exp(self.gamma2_ALG3*(self.T4_ALG3 - C_all['Temp'])))
                    /(1+self.K4_ALG3*np.exp(self.gamma2_ALG3*(self.T4_ALG3 - C_all['Temp']))-self.K4_ALG3) )
        gamma_ALG3 = gamma_r_ALG3*gamma_f_ALG3
        
        gamma_r_ALG4 = ((self.K1_ALG4*np.exp(self.gamma1_ALG4*(C_all['Temp'] - self.T1_ALG4)))
                    /(1+self.K1_ALG4*np.exp(self.gamma1_ALG4*(C_all['Temp'] - self.T1_ALG4))-self.K1_ALG4) )
        gamma_f_ALG4 = ((self.K4_ALG4*np.exp(self.gamma2_ALG4*(self.T4_ALG4 - C_all['Temp'])))
                    /(1+self.K4_ALG4*np.exp(self.gamma2_ALG4*(self.T4_ALG4 - C_all['Temp']))-self.K4_ALG4) )
        gamma_ALG4 = gamma_r_ALG4*gamma_f_ALG4
        
        # Limiting factors: Nitrogen, Phosphorus, Light
        lambda_N = (C_all['NH4'] +  C_all['NO3'])/(self.K_h_N + C_all['NH4'] +  C_all['NO3'])
        lambda_P = (C_all['PO4'])/(self.K_h_P + C_all['PO4'])
        
        C_all_POM = C_all['LPOC'] + C_all['RPOC'] + C_all['LPON'] + C_all['RPON'] + C_all['LPOP'] + C_all['RPOP']
        C_all_ALG = C_all['ALG1'] + C_all['ALG2'] + C_all['ALG3'] + C_all['ALG4']
        k_e = self.keb + self.alpha_ISS*C_all['ISS'] + self.alpha_POM*C_all_POM + self.alpha_ALG*C_all_ALG

        lambda_light = (2.7182/(k_e*H_dep))*(np.exp(-0.485*24*MET_solar[T]/self.K_Lp)*np.exp(k_e*H_dep) - np.exp(-0.485*24*MET_solar[T]/self.K_Lp))
        lambda_min = min_lambda(lambda_N, lambda_P, lambda_light)
        
        K_ag_ALG1 = gamma_ALG1*lambda_min*self.K_ag_max_ALG1
        K_ag_ALG2 = gamma_ALG2*lambda_min*self.K_ag_max_ALG2
        K_ag_ALG3 = gamma_ALG3*lambda_min*self.K_ag_max_ALG3
        K_ag_ALG4 = gamma_ALG4*lambda_min*self.K_ag_max_ALG4
        
        K_ar_ALG1 = gamma_ALG1*self.K_ar_max_ALG1
        K_ar_ALG2 = gamma_ALG2*self.K_ar_max_ALG2
        K_ar_ALG3 = gamma_ALG3*self.K_ar_max_ALG3
        K_ar_ALG4 = gamma_ALG4*self.K_ar_max_ALG4
        
        K_ae_ALG1 = (1-lambda_light)*gamma_ALG1*self.K_ae_max_ALG1
        K_ae_ALG2 = (1-lambda_light)*gamma_ALG2*self.K_ae_max_ALG2
        K_ae_ALG3 = (1-lambda_light)*gamma_ALG3*self.K_ae_max_ALG3
        K_ae_ALG4 = (1-lambda_light)*gamma_ALG4*self.K_ae_max_ALG4
        
        K_am_ALG1 = gamma_ALG1*self.K_am_max_ALG1
        K_am_ALG2 = gamma_ALG2*self.K_am_max_ALG2
        K_am_ALG3 = gamma_ALG3*self.K_am_max_ALG3
        K_am_ALG4 = gamma_ALG4*self.K_am_max_ALG4
        
        C_new['ALG1'] = RK4_ALG(dt, C_all['ALG1'], K_ag_ALG1, K_ar_ALG1, K_ae_ALG1, K_am_ALG1, self.K_s_ALG1, H_dep)
        C_new['ALG2'] = RK4_ALG(dt, C_all['ALG2'], K_ag_ALG2, K_ar_ALG2, K_ae_ALG2, K_am_ALG2, self.K_s_ALG2, H_dep)
        C_new['ALG3'] = RK4_ALG(dt, C_all['ALG3'], K_ag_ALG3, K_ar_ALG3, K_ae_ALG3, K_am_ALG3, self.K_s_ALG3, H_dep)
        C_new['ALG4'] = RK4_ALG(dt, C_all['ALG4'], K_ag_ALG4, K_ar_ALG4, K_ae_ALG4, K_am_ALG4, self.K_s_ALG4, H_dep)
        
        ###################################################################
        #   Dissolved Oxygen
        ###################################################################
        # Caution: algal parameters should be defined before this step!!!
        K_L = K_L_cal(self.U_vel, H_dep)
        DO_1 = (K_ag_ALG1*self.delta_O_ALG1_ag*C_all['ALG1']
                + K_ag_ALG2*self.delta_O_ALG2_ag*C_all['ALG2']
                + K_ag_ALG3*self.delta_O_ALG3_ag*C_all['ALG3']
                + K_ag_ALG4*self.delta_O_ALG4_ag*C_all['ALG4'] 
                - K_ar_ALG1*self.delta_O_ALG1_ar*C_all['ALG1']
                - K_ar_ALG2*self.delta_O_ALG2_ar*C_all['ALG2']
                - K_ar_ALG3*self.delta_O_ALG3_ar*C_all['ALG3'] 
                - K_ar_ALG4*self.delta_O_ALG4_ar*C_all['ALG4'] )
        DO_2 = self.delta_O_OC*gamma_OC*(- self.K_LDOC*C_all['LDOC']
                                    - self.K_LPOC*C_all['LPOC'] 
                                    - self.K_RDOC*C_all['RDOC'] 
                                    - self.K_RPOC*C_all['RPOC'])
        DO_3 = self.K_NH4*self.delta_O_NH4*gamma_NH4*C_all['NH4'] - self.K_SOD*gamma_OC*H_dep
        C_new['DO'] = RK4_DO(dt, C_all['DO'], C_all['Temp'], K_L, DO_1, DO_2, DO_3)
        
        ###################################################################
        #   Carbon group: DIC, LDOC, LPOC, RDOC, RPOC
        ###################################################################
        # Dissolived Organic Carbon
        DIC_1 = (K_ar_ALG1*self.delta_C_ALG1_ar*C_all['ALG1'] 
                 + K_ar_ALG2*self.delta_C_ALG2_ar*C_all['ALG2']
                 + K_ar_ALG3*self.delta_C_ALG3_ar*C_all['ALG3'] 
                 + K_ar_ALG4*self.delta_C_ALG4_ar*C_all['ALG4']
                 - K_ag_ALG1*self.delta_C_ALG1_ag*C_all['ALG1'] 
                 - K_ag_ALG2*self.delta_C_ALG2_ag*C_all['ALG2']
                 - K_ag_ALG3*self.delta_C_ALG3_ag*C_all['ALG3'] 
                 - K_ag_ALG4*self.delta_C_ALG4_ag*C_all['ALG4'])
        DIC_2 = (self.delta_C_OC*gamma_OC*(self.K_LDOC*C_all['LDOC'] 
                                      + self.K_LPOC*C_all['LPOC']
                                      + self.K_RDOC*C_all['RDOC'] 
                                      + self.K_RPOC*C_all['RPOC'])
                 + self.K_SOD*self.delta_C_OC*gamma_OC/H_dep)
        C_new['DIC'] = RK4_DIC(dt, C_all['DIC'], K_L, self.CO2_air, DIC_1, DIC_2)
        
        # LDOC: Labile Dissolived Organic Carbon
        LDOC_1 = self.P_d_C*(K_am_ALG1*self.delta_C_ALG1_am*C_all['ALG1'] 
                        + K_am_ALG2*self.delta_C_ALG2_am*C_all['ALG2']
                        + K_am_ALG3*self.delta_C_ALG3_am*C_all['ALG3'] 
                        + K_am_ALG4*self.delta_C_ALG4_am*C_all['ALG4']
                        + K_ae_ALG1*self.delta_C_ALG1_ae*C_all['ALG1']
                        + K_ae_ALG2*self.delta_C_ALG2_ae*C_all['ALG2']
                        + K_ae_ALG3*self.delta_C_ALG3_ae*C_all['ALG3'] 
                        + K_ae_ALG4*self.delta_C_ALG4_ae*C_all['ALG4'])
        C_new['LDOC'] = RK4_LDOC(dt, C_all['LDOC'], self.K_LDOC,
                                    self.K_LDOC_to_RDOC, gamma_OC, LDOC_1)
        
        # LPOC: Labile Particulate Organic Carbon
        LPOC_1 = (1-self.P_d_C)*(K_am_ALG1*self.delta_C_ALG1_am*C_all['ALG1'] 
                            + K_am_ALG2*self.delta_C_ALG2_am*C_all['ALG2']
                            + K_am_ALG3*self.delta_C_ALG3_am*C_all['ALG3']
                            + K_am_ALG4*self.delta_C_ALG4_am*C_all['ALG4']
                            + K_ae_ALG1*self.delta_C_ALG1_ae*C_all['ALG1']
                            + K_ae_ALG2*self.delta_C_ALG2_ae*C_all['ALG2']
                            + K_ae_ALG3*self.delta_C_ALG3_ae*C_all['ALG3']
                            + K_ae_ALG4*self.delta_C_ALG4_ae*C_all['ALG4'])
        C_new['LPOC'] = RK4_LPOC(dt, C_all['LPOC'], self.K_LPOC, self.K_LPOC_to_RPOC,
                                    self.K_s_LPOC, H_dep, gamma_OC, LPOC_1)
        
        # RDOC: Refractory Dissolved Organic Carbon
        RDOC_1 = self.K_LDOC_to_RDOC*gamma_OC*C_all['LDOC']
        C_new['RDOC'] = RK4_RDOC(dt, C_all['RDOC'], self.K_RDOC, gamma_OC, RDOC_1)
        
        # RPOC: Refractory Particulate Organic Carbon
        RPOC_1 = self.K_LPOC_to_RPOC*gamma_OC*C_all['LPOC']
        C_new['RPOC'] = RK4_RPOC(dt, C_all['RPOC'], self.K_RPOC, gamma_OC, self.K_s_RPOC, H_dep, RPOC_1)
        
        ###################################################################
        #   Nitogen group: PIN, NH4, NO3, LDON, LPON, RDON, RPON
        ###################################################################
        # PIN: Particulate Inorganic Nitrogen
        PIN_ads = self.K_ads_PIN*(C_all['ISS']/(self.K_h_PIN + C_all['ISS']))
        PIN_1 = PIN_ads*C_all['NH4']
        C_new['PIN'] = RK4_PIN(dt, C_all['PIN'], self.K_des_PIN, self.K_s_PIN, H_dep, PIN_1)
        P_NH4 = ( C_all['NH4']*(C_all['NO3']/((self.K_h_NH4 + C_all['NH4'])*(self.K_h_NH4 + C_all['NO3']))) 
                 + C_all['NH4']*(self.K_h_NH4/((C_all['NH4'] + C_all['NO3'])*(self.K_h_NH4 + C_all['NO3']))) )
        
        # NH4: Ammonium
        NH4_1 = (K_ar_ALG1*self.delta_N_ALG1_ar*C_all['ALG1'] 
                 + K_ar_ALG2*self.delta_N_ALG2_ar*C_all['ALG2']
                 + K_ar_ALG3*self.delta_N_ALG3_ar*C_all['ALG3'] 
                 + K_ar_ALG4*self.delta_N_ALG4_ar*C_all['ALG4']
                 
                 - P_NH4*(K_ag_ALG1*self.delta_N_ALG1_ag*C_all['ALG1'] 
                 + K_ag_ALG2*self.delta_N_ALG2_ag*C_all['ALG2']
                 + K_ag_ALG3*self.delta_N_ALG3_ag*C_all['ALG3'] 
                 + K_ag_ALG4*self.delta_N_ALG4_ag*C_all['ALG4']) )
        NH4_2 = (self.delta_N_ON*gamma_ON*(self.K_LDON*C_all['LDON'] 
                                      + self.K_LPON*C_all['LPON']
                                      + self.K_RDON*C_all['RDON'] 
                                      + self.K_RPON*C_all['RPON']))
        C_new['NH4'] = RK4_NH4(dt, C_all['NH4'], self.K_NH4, gamma_NH4, PIN_ads, NH4_1, NH4_2)
        
        # NO3: Nitrate
        NO3_1 = ( -(1-P_NH4)*(K_ag_ALG1*self.delta_N_ALG1_ag*C_all['ALG1']
                            + K_ag_ALG2*self.delta_N_ALG2_ag*C_all['ALG2']
                            + K_ag_ALG3*self.delta_N_ALG3_ag*C_all['ALG3']
                            + K_ag_ALG4*self.delta_N_ALG4_ag*C_all['ALG4'])
                 + self.K_NH4*gamma_NH4*C_all['NH4'] )
        C_new['NO3'] = RK4_NO3(dt, C_all['NO3'], self.K_NO3, gamma_NO3, self.K_s_NO3, H_dep, NO3_1)
        
        # LDON: Labile Dissolived Organic Nitrogen
        LDON_1 = self.P_d_N*(K_am_ALG1*self.delta_N_ALG1_am*C_all['ALG1'] 
                        + K_am_ALG2*self.delta_N_ALG2_am*C_all['ALG2']
                        + K_am_ALG3*self.delta_N_ALG3_am*C_all['ALG3'] 
                        + K_am_ALG4*self.delta_N_ALG4_am*C_all['ALG4']
                        + K_ae_ALG1*self.delta_N_ALG1_ae*C_all['ALG1']
                        + K_ae_ALG2*self.delta_N_ALG2_ae*C_all['ALG2']
                        + K_ae_ALG3*self.delta_N_ALG3_ae*C_all['ALG3'] 
                        + K_ae_ALG4*self.delta_N_ALG4_ae*C_all['ALG4'])
        C_new['LDON'] = RK4_LDON(dt, C_all['LDON'], self.K_LDON,
                                    self.K_LDON_to_RDON, gamma_ON, LDON_1)
        
        # LPON: Labile Particulate Organic Nitrogen
        LPON_1 = (1-self.P_d_N)*(K_am_ALG1*self.delta_N_ALG1_am*C_all['ALG1'] 
                            + K_am_ALG2*self.delta_N_ALG2_am*C_all['ALG2']
                            + K_am_ALG3*self.delta_N_ALG3_am*C_all['ALG3']
                            + K_am_ALG4*self.delta_N_ALG4_am*C_all['ALG4']
                            + K_ae_ALG1*self.delta_N_ALG1_ae*C_all['ALG1']
                            + K_ae_ALG2*self.delta_N_ALG2_ae*C_all['ALG2']
                            + K_ae_ALG3*self.delta_N_ALG3_ae*C_all['ALG3']
                            + K_ae_ALG4*self.delta_N_ALG4_ae*C_all['ALG4'])
        C_new['LPON'] = RK4_LPON(dt, C_all['LPON'], self.K_LPON, self.K_LPON_to_RPON,
                                    self.K_s_LPON, H_dep, gamma_ON, LPON_1)
        
        # RDON: Refractory Dissolved Organic Nitrogen
        RDON_1 = self.K_LDON_to_RDON*gamma_ON*C_all['LDON']
        C_new['RDON'] = RK4_RDON(dt, C_all['RDON'], self.K_RDON, gamma_ON, RDON_1)
        
        # RPON: Refractory Particulate Organic Nitrogen
        RPON_1 = self.K_LPON_to_RPON*gamma_ON*C_all['LPON']
        C_new['RPON'] = RK4_RPON(dt, C_all['RPON'], self.K_RPON, gamma_ON, self.K_s_RPON, H_dep, RPON_1)
        
        ###################################################################
        #   Phosphorus group: PIP, PO4, LDOP, LPOP, RDOP, RPOP
        ###################################################################
        # PO4: Phosphate
        PO4_ads = self.K_ads_PIP*(C_all['ISS']/(self.K_h_PIP + C_all['ISS']))
        PO4_1 = (K_ar_ALG1*self.delta_P_ALG1_ar*C_all['ALG1'] 
                 + K_ar_ALG2*self.delta_P_ALG2_ar*C_all['ALG2']
                 + K_ar_ALG3*self.delta_P_ALG3_ar*C_all['ALG3'] 
                 + K_ar_ALG4*self.delta_P_ALG4_ar*C_all['ALG4']
                 
                 - K_ag_ALG1*self.delta_P_ALG1_ag*C_all['ALG1'] 
                 - K_ag_ALG2*self.delta_P_ALG2_ag*C_all['ALG2']
                 - K_ag_ALG3*self.delta_P_ALG3_ag*C_all['ALG3'] 
                 - K_ag_ALG4*self.delta_P_ALG4_ag*C_all['ALG4'] )
        PO4_2 = ( self.K_des_PIP*C_all['PIP'] 
                 + self.delta_P_OP*gamma_OP*(self.K_LDOP*C_all['LDOP']
                                             + self.K_LPOP*C_all['LPOP']
                                             + self.K_RDOP*C_all['RDOP'] 
                                             + self.K_RPOP*C_all['RPOP']) )
        C_new['PO4'] = RK4_PO4(dt, C_all['PO4'], PO4_ads, PO4_1, PO4_2)

        # PIP: Particulate Inorganic Phosphorus
        PIP_1 = PO4_ads*C_all['PO4']
        C_new['PIP'] = RK4_PIP(dt, C_all['PIP'], self.K_des_PIP, self.K_s_PIP, H_dep, PIP_1)
        
        # LDOP: Labile Dissolived Organic Nitrogen
        LDOP_1 = self.P_d_N*(K_am_ALG1*self.delta_N_ALG1_am*C_all['ALG1'] 
                        + K_am_ALG2*self.delta_N_ALG2_am*C_all['ALG2']
                        + K_am_ALG3*self.delta_N_ALG3_am*C_all['ALG3'] 
                        + K_am_ALG4*self.delta_N_ALG4_am*C_all['ALG4']
                        + K_ae_ALG1*self.delta_N_ALG1_ae*C_all['ALG1']
                        + K_ae_ALG2*self.delta_N_ALG2_ae*C_all['ALG2']
                        + K_ae_ALG3*self.delta_N_ALG3_ae*C_all['ALG3'] 
                        + K_ae_ALG4*self.delta_N_ALG4_ae*C_all['ALG4'])
        C_new['LDOP'] = RK4_LDOP(dt, C_all['LDOP'], self.K_LDOP,
                                    self.K_LDOP_to_RDOP, gamma_OP, LDOP_1)
        
        # LPOP: Labile Particulate Organic Nitrogen
        LPOP_1 = (1-self.P_d_N)*(K_am_ALG1*self.delta_N_ALG1_am*C_all['ALG1'] 
                            + K_am_ALG2*self.delta_N_ALG2_am*C_all['ALG2']
                            + K_am_ALG3*self.delta_N_ALG3_am*C_all['ALG3']
                            + K_am_ALG4*self.delta_N_ALG4_am*C_all['ALG4']
                            + K_ae_ALG1*self.delta_N_ALG1_ae*C_all['ALG1']
                            + K_ae_ALG2*self.delta_N_ALG2_ae*C_all['ALG2']
                            + K_ae_ALG3*self.delta_N_ALG3_ae*C_all['ALG3']
                            + K_ae_ALG4*self.delta_N_ALG4_ae*C_all['ALG4'])
        C_new['LPOP'] = RK4_LPOP(dt, C_all['LPOP'], self.K_LPOP, self.K_LPOP_to_RPOP,
                                    self.K_s_LPOP, H_dep, gamma_OP, LPOP_1)
        
        # RDOP: Refractory Dissolved Organic Nitrogen
        RDOP_1 = self.K_LDOP_to_RDOP*gamma_OP*C_all['LDOP']
        C_new['RDOP'] = RK4_RDOP(dt, C_all['RDOP'], self.K_RDOP, gamma_OP, RDOP_1)
        
        # RPOP: Refractory Particulate Organic Nitrogen
        RPOP_1 = self.K_LPOP_to_RPOP*gamma_OP*C_all['LPOP']
        C_new['RPOP'] = RK4_RPOP(dt, C_all['RPOP'], self.K_RPOP, gamma_OP, self.K_s_RPOP, H_dep, RPOP_1)
        
        ###################################################################
        #   Inorganic Suspended Solids: ISS
        ###################################################################
        C_new['ISS'] = RK4_ISS(dt, C_all['ISS'], self.K_s_ISS, H_dep)
        
        self.C_all = C_new
    
    def make_QR_cov(self, Q_spatial, Q_species):
        # Define the measurement noise covariance matrix: The measurement noise is the same at all superjunctions and all WQ species.
        R_cov = (self.N_measure_sigma**2)*np.eye(self.hydraulics.M*self.N_constituent,self.hydraulics.M*self.N_constituent)
        self.R_cov = R_cov
       
        # Make the spatial correlation matrices
        sc = np.zeros((self.N_constituent,self.N_constituent,self.hydraulics.M,self.hydraulics.M))
        for i in range(0,self.N_constituent):
            for j in range(0,self.N_constituent):
                for k in range(0,self.hydraulics.M):
                    sc[i][j][k][k] = Q_species[k][i][j]
                    
        # Define the process noise covariance matrix based on spatial correlation for each constituent and inter-species correlation for each superjunction
        Q_cov = np.block([[Q_spatial['Temp'],sc[0][1],sc[0][2],sc[0][3],sc[0][4],sc[0][5],sc[0][6],sc[0][7],sc[0][8],sc[0][9],sc[0][10],sc[0][11],sc[0][12],sc[0][13],
                           sc[0][14],sc[0][15],sc[0][16],sc[0][17],sc[0][18],sc[0][19],sc[0][20],sc[0][21],sc[0][22],sc[0][23],sc[0][24]],
                        [sc[1][0],Q_spatial['DO'],sc[1][2],sc[1][3],sc[1][4],sc[1][5],sc[1][6],sc[1][7],sc[1][8],sc[1][9],sc[1][10],sc[1][11],sc[1][12],sc[1][13],
                         sc[1][14],sc[1][15],sc[1][16],sc[1][17],sc[1][18],sc[1][19],sc[1][20],sc[1][21],sc[1][22],sc[1][23],sc[1][24]],
                        [sc[2][0],sc[2][1],Q_spatial['DIC'],sc[2][3],sc[2][4],sc[2][5],sc[2][6],sc[2][7],sc[2][8],sc[2][9],sc[2][10],sc[2][11],sc[2][12],sc[2][13],
                         sc[2][14],sc[2][15],sc[2][16],sc[2][17],sc[2][18],sc[2][19],sc[2][20],sc[2][21],sc[2][22],sc[2][23],sc[2][24]],
                        [sc[3][0],sc[3][1],sc[3][2],Q_spatial['LDOC'],sc[3][4],sc[3][5],sc[3][6],sc[3][7],sc[3][8],sc[3][9],sc[3][10],sc[3][11],sc[3][12],sc[3][13],
                         sc[3][14],sc[3][15],sc[3][16],sc[3][17],sc[3][18],sc[3][19],sc[3][20],sc[3][21],sc[3][22],sc[3][23],sc[3][24]],
                        [sc[4][0],sc[4][1],sc[4][2],sc[4][3],Q_spatial['LPOC'],sc[4][5],sc[4][6],sc[4][7],sc[4][8],sc[4][9],sc[4][10],sc[4][11],sc[4][12],sc[4][13],
                         sc[4][14],sc[4][15],sc[4][16],sc[4][17],sc[4][18],sc[4][19],sc[4][20],sc[4][21],sc[4][22],sc[4][23],sc[4][24]],
                        [sc[5][0],sc[5][1],sc[5][2],sc[5][3],sc[5][4],Q_spatial['RDOC'],sc[5][6],sc[5][7],sc[5][8],sc[5][9],sc[5][10],sc[5][11],sc[5][12],sc[5][13],
                         sc[5][14],sc[5][15],sc[5][16],sc[5][17],sc[5][18],sc[5][19],sc[5][20],sc[5][21],sc[5][22],sc[5][23],sc[5][24]],
                        [sc[6][0],sc[6][1],sc[6][2],sc[6][3],sc[6][4],sc[6][5],Q_spatial['RPOC'],sc[6][7],sc[6][8],sc[6][9],sc[6][10],sc[6][11],sc[6][12],sc[6][13],
                         sc[6][14],sc[6][15],sc[6][16],sc[6][17],sc[6][18],sc[6][19],sc[6][20],sc[6][21],sc[6][22],sc[6][23],sc[6][24]],
                        [sc[7][0],sc[7][1],sc[7][2],sc[7][3],sc[7][4],sc[7][5],sc[7][6],Q_spatial['PIN'],sc[7][8],sc[7][9],sc[7][10],sc[7][11],sc[7][12],sc[7][13],
                         sc[7][14],sc[7][15],sc[7][16],sc[7][17],sc[7][18],sc[7][19],sc[7][20],sc[7][21],sc[7][22],sc[7][23],sc[7][24]],
                        [sc[8][0],sc[8][1],sc[8][2],sc[8][3],sc[8][4],sc[8][5],sc[8][6],sc[8][7],Q_spatial['NH4'],sc[8][9],sc[8][10],sc[8][11],sc[8][12],sc[8][13],
                         sc[8][14],sc[8][15],sc[8][16],sc[8][17],sc[8][18],sc[8][19],sc[8][20],sc[8][21],sc[8][22],sc[8][23],sc[8][24]],
                        [sc[9][0],sc[9][1],sc[9][2],sc[9][3],sc[9][4],sc[9][5],sc[9][6],sc[9][7],sc[9][8],Q_spatial['NO3'],sc[9][10],sc[9][11],sc[9][12],sc[9][13],
                         sc[9][14],sc[9][15],sc[9][16],sc[9][17],sc[9][18],sc[9][19],sc[9][20],sc[9][21],sc[9][22],sc[9][23],sc[9][24]],
                        [sc[10][0],sc[10][1],sc[10][2],sc[10][3],sc[10][4],sc[10][5],sc[10][6],sc[10][7],sc[10][8],sc[10][9],Q_spatial['LDON'],sc[10][11],sc[10][12],sc[10][13],
                         sc[10][14],sc[10][15],sc[10][16],sc[10][17],sc[10][18],sc[10][19],sc[10][20],sc[10][21],sc[10][22],sc[10][23],sc[10][24]],
                        [sc[11][0],sc[11][1],sc[11][2],sc[11][3],sc[11][4],sc[11][5],sc[11][6],sc[11][7],sc[11][8],sc[11][9],sc[11][10],Q_spatial['LPON'],sc[11][12],sc[11][13],
                         sc[11][14],sc[11][15],sc[11][16],sc[11][17],sc[11][18],sc[11][19],sc[11][20],sc[11][21],sc[11][22],sc[11][23],sc[11][24]],
                        [sc[12][0],sc[12][1],sc[12][2],sc[12][3],sc[12][4],sc[12][5],sc[12][6],sc[12][7],sc[12][8],sc[12][9],sc[12][10],sc[12][11],Q_spatial['RDON'],sc[12][13],
                         sc[12][14],sc[12][15],sc[12][16],sc[12][17],sc[12][18],sc[12][19],sc[12][20],sc[12][21],sc[12][22],sc[12][23],sc[12][24]],
                        [sc[13][0],sc[13][1],sc[13][2],sc[13][3],sc[13][4],sc[13][5],sc[13][6],sc[13][7],sc[13][8],sc[13][9],sc[13][10],sc[13][11],sc[13][12],Q_spatial['RPON'],
                         sc[13][14],sc[13][15],sc[13][16],sc[13][17],sc[13][18],sc[13][19],sc[13][20],sc[13][21],sc[13][22],sc[13][23],sc[13][24]],
                        [sc[14][0],sc[14][1],sc[14][2],sc[14][3],sc[14][4],sc[14][5],sc[14][6],sc[14][7],sc[14][8],sc[14][9],sc[14][10],sc[14][11],sc[14][12],sc[14][13]
                         ,Q_spatial['PIP'],sc[14][15],sc[14][16],sc[14][17],sc[14][18],sc[14][19],sc[14][20],sc[14][21],sc[14][22],sc[14][23],sc[14][24]],
                        [sc[15][0],sc[15][1],sc[15][2],sc[15][3],sc[15][4],sc[15][5],sc[15][6],sc[15][7],sc[15][8],sc[15][9],sc[15][10],sc[15][11],sc[15][12],sc[15][13],
                         sc[15][14],Q_spatial['PO4'],sc[15][16],sc[15][17],sc[15][18],sc[15][19],sc[15][20],sc[15][21],sc[15][22],sc[15][23],sc[15][24]],
                        [sc[16][0],sc[16][1],sc[16][2],sc[16][3],sc[16][4],sc[16][5],sc[16][6],sc[16][7],sc[16][8],sc[16][9],sc[16][10],sc[16][11],sc[16][12],sc[16][13],
                         sc[16][14],sc[16][15],Q_spatial['LDOP'],sc[16][17],sc[16][18],sc[16][19],sc[16][20],sc[16][21],sc[16][22],sc[16][23],sc[16][24]],
                        [sc[17][0],sc[17][1],sc[17][2],sc[17][3],sc[17][4],sc[17][5],sc[17][6],sc[17][7],sc[17][8],sc[17][9],sc[17][10],sc[17][11],sc[17][12],sc[17][13],
                         sc[17][14],sc[17][15],sc[17][16],Q_spatial['LPOP'],sc[17][18],sc[17][19],sc[17][20],sc[17][21],sc[17][22],sc[17][23],sc[17][24]],
                        [sc[18][0],sc[18][1],sc[18][2],sc[18][3],sc[18][4],sc[18][5],sc[18][6],sc[18][7],sc[18][8],sc[18][9],sc[18][10],sc[18][11],sc[18][12],sc[18][13],
                         sc[18][14],sc[18][15],sc[18][16],sc[18][17],Q_spatial['RDOP'],sc[18][19],sc[18][20],sc[18][21],sc[18][22],sc[18][23],sc[18][24]],
                        [sc[19][0],sc[19][1],sc[19][2],sc[19][3],sc[19][4],sc[19][5],sc[19][6],sc[19][7],sc[19][8],sc[19][9],sc[19][10],sc[19][11],sc[19][12],sc[19][13],
                         sc[19][14],sc[19][15],sc[19][16],sc[19][17],sc[19][18],Q_spatial['RPOP'],sc[19][20],sc[19][21],sc[19][22],sc[19][23],sc[19][24]],
                        [sc[20][0],sc[20][1],sc[20][2],sc[20][3],sc[20][4],sc[20][5],sc[20][6],sc[20][7],sc[20][8],sc[20][9],sc[20][10],sc[20][11],sc[20][12],sc[20][13],
                         sc[20][14],sc[20][15],sc[20][16],sc[20][17],sc[20][18],sc[20][19],Q_spatial['ALG1'],sc[20][21],sc[20][22],sc[20][23],sc[20][24]],
                        [sc[21][0],sc[21][1],sc[21][2],sc[21][3],sc[21][4],sc[21][5],sc[21][6],sc[21][7],sc[21][8],sc[21][9],sc[21][10],sc[21][11],sc[21][12],sc[21][13],
                         sc[21][14],sc[21][15],sc[21][16],sc[21][17],sc[21][18],sc[21][19],sc[21][20],Q_spatial['ALG2'],sc[21][22],sc[21][23],sc[21][24]],
                        [sc[22][0],sc[22][1],sc[22][2],sc[22][3],sc[22][4],sc[22][5],sc[22][6],sc[22][7],sc[22][8],sc[22][9],sc[22][10],sc[22][11],sc[22][12],sc[22][13],
                         sc[22][14],sc[22][15],sc[22][16],sc[22][17],sc[22][18],sc[22][19],sc[22][20],sc[22][21],Q_spatial['ALG3'],sc[22][23],sc[22][24]],
                        [sc[23][0],sc[23][1],sc[23][2],sc[23][3],sc[23][4],sc[23][5],sc[23][6],sc[23][7],sc[23][8],sc[23][9],sc[23][10],sc[23][11],sc[23][12],sc[23][13],
                         sc[23][14],sc[23][15],sc[23][16],sc[23][17],sc[23][18],sc[23][19],sc[23][20],sc[23][21],sc[23][22],Q_spatial['ALG4'],sc[23][24]],
                        [sc[24][0],sc[24][1],sc[24][2],sc[24][3],sc[24][4],sc[24][5],sc[24][6],sc[24][7],sc[24][8],sc[24][9],sc[24][10],sc[24][11],sc[24][12],sc[24][13],
                         sc[24][14],sc[24][15],sc[24][16],sc[24][17],sc[24][18],sc[24][19],sc[24][20],sc[24][21],sc[24][22],sc[24][23],Q_spatial['ISS']]])
        
        Q_cov = self.N_process_sigma**2 * Q_cov
        self.Q_cov = Q_cov
                
    def KF_DA(self, dt, N_j, WQ_A_k, WQ_B_k, WQ, obs_data, T, name, Q_spatial, Q_species):
        x_hat = np.block([WQ['Temp'].c_j.T, WQ['DO'].c_j.T, WQ['DIC'].c_j.T, 
                        WQ['LDOC'].c_j.T, WQ['LPOC'].c_j.T, WQ['RDOC'].c_j.T,
                        WQ['RPOC'].c_j.T, WQ['PIN'].c_j.T, WQ['NH4'].c_j.T,
                        WQ['NO3'].c_j.T, WQ['LDON'].c_j.T, WQ['LPON'].c_j.T,
                        WQ['RDON'].c_j.T, WQ['RPON'].c_j.T, WQ['PIP'].c_j.T,
                        WQ['PO4'].c_j.T, WQ['LDOP'].c_j.T, WQ['LPOP'].c_j.T,
                        WQ['RDOP'].c_j.T, WQ['RPOP'].c_j.T, WQ['ALG1'].c_j.T,
                        WQ['ALG2'].c_j.T, WQ['ALG3'].c_j.T, WQ['ALG4'].c_j.T,
                        WQ['ISS'].c_j.T])
        z0 = np.zeros((N_j, N_j))
        TA_k = np.block([[WQ_A_k['Temp'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,WQ_A_k['DO'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,WQ_A_k['DIC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,WQ_A_k['LDOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,WQ_A_k['LPOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,WQ_A_k['RDOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,WQ_A_k['RPOC'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,WQ_A_k['PIN'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['NH4'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['NO3'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LDON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LPON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RDON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RPON'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['PIP'],z0,z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['PO4'],z0,z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LDOP'],z0,z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['LPOP'],z0,z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RDOP'],z0,z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['RPOP'],z0,z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG1'],z0,z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG2'],z0,z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG3'],z0,z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ALG4'],z0],
                        [z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,z0,WQ_A_k['ISS']]])

        TB_k = np.block([WQ_B_k['Temp'].T, WQ_B_k['DO'].T, WQ_B_k['DIC'].T, 
                        WQ_B_k['LDOC'].T, WQ_B_k['LPOC'].T, WQ_B_k['RDOC'].T,
                        WQ_B_k['RPOC'].T, WQ_B_k['PIN'].T, WQ_B_k['NH4'].T,
                        WQ_B_k['NO3'].T, WQ_B_k['LDON'].T, WQ_B_k['LPON'].T,
                        WQ_B_k['RDON'].T, WQ_B_k['RPON'].T, WQ_B_k['PIP'].T,
                        WQ_B_k['PO4'].T, WQ_B_k['LDOP'].T, WQ_B_k['LPOP'].T,
                        WQ_B_k['RDOP'].T, WQ_B_k['RPOP'].T, WQ_B_k['ALG1'].T,
                        WQ_B_k['ALG2'].T, WQ_B_k['ALG3'].T, WQ_B_k['ALG4'].T,
                        WQ_B_k['ISS'].T])
        
        T_obs = np.block([obs_data['Temp'][T], obs_data['DO'][T], obs_data['DIC'][T], 
                        obs_data['LDOC'][T], obs_data['LPOC'][T], obs_data['RDOC'][T],
                        obs_data['RPOC'][T], obs_data['PIN'][T], obs_data['NH4'][T],
                        obs_data['NO3'][T], obs_data['LDON'][T], obs_data['LPON'][T],
                        obs_data['RDON'][T], obs_data['RPON'][T], obs_data['PIP'][T],
                        obs_data['PO4'][T], obs_data['LDOP'][T], obs_data['LPOP'][T],
                        obs_data['RDOP'][T], obs_data['RPOP'][T], obs_data['ALG1'][T],
                        obs_data['ALG2'][T], obs_data['ALG3'][T], obs_data['ALG4'][T],
                        obs_data['ISS'][T]])
  
        # Make the observation matrix
        H_k = np.zeros((self.hydraulics.M*self.N_constituent,self.hydraulics.M*self.N_constituent))
        # If there is observed data at specific superjunction for specific constituent, it will be 1
        for i in range(0,self.hydraulics.M*self.N_constituent):
            if T_obs[i] < 0:
                H_k[i,i] = 0
            else:
                H_k[i,i] = 1
        
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        P_k = self.P_k
        
        # Kalman Filtering Algorithm
        x_hat_k = TA_k @ x_hat + TB_k
        P_k = TA_k @ P_k @ TA_k.T + Q_cov
        K_k = P_k @ H_k.T @ np.linalg.inv(H_k @ P_k @ H_k.T + R_cov)
        P_k = P_k - K_k @ H_k @ P_k
        x_hat = x_hat_k + K_k @ (T_obs - H_k @ x_hat_k)
        
        x_hat[x_hat < 0.0] = 10e-7
        self.P_k = P_k
        self.K_k = K_k  # Not necessary, just for checking
        self.H_k = H_k  # Not necessary, just for checking
        
        return x_hat
   
##############################################################################
#   Temperature
##############################################################################
def F_Temp(Temp, J_solar, Tair, Tdew, Fw, T_si, H_dep):
#    Temp = np.where(Temp<0,0,Temp)
    e_s = 4.596*np.exp( 17.27*Temp/(237.3 + Temp))
    e_air = 4.596*np.exp( 17.27*Tdew/(237.3 + Tdew))
    F_answer = ( 23.9*J_solar + 11.7*0.00000001*np.power(Tair + 273.15, 4)*(0.6+0.031*np.sqrt(e_air))*(1-0.03) 
                -0.97*11.7*0.00000001*np.power(Temp + 273.15, 4)
                -0.47*Fw*(Temp - Tair) -Fw*(e_s - e_air) -0.3*0.485*(Temp - T_si))/H_dep
    return F_answer

def RK4_Temp(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_Temp(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_Temp(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_Temp(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_Temp(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Dissolved Oxygen
##############################################################################
@njit
def K_L_cal(U_vel, H_dep):
    # just copy the variable's dimension
    K_L = 0*U_vel
    K_ah = 0*U_vel
    K_w =  0*U_vel
    for i in range(0,len(U_vel)):
        if H_dep[i]<0.6:
            K_ah[i] = 5.32*np.power(U_vel[i],0.67)/np.power(H_dep[i], 1.85)
        elif (H_dep[i]>0.6 and H_dep[i] > 3.45*np.power(U_vel[i],2.5)):
            K_ah[i] = 3.93*np.power(U_vel[i],0.5)/np.power(H_dep[i], 1.67)
        else:
            K_ah[i] = 5.026*U_vel[i]/np.power(H_dep[i], 1.67)
    # K_w equation should be added here
    K_L = K_ah + K_w
    return K_L

def F_DO(DO, Temp, K_L, DO_1, DO_2, DO_3):
    DO_sat = np.exp( (1.575701*1e5)/(Temp+273.15)-(6.642308*1e7)/np.power(Temp+273.15,2)
    +(1.2438*1e10)/np.power(Temp+273.15,3)-(8.621949*1e11)/np.power(Temp+273.15,4)-139.34411)
    F_answer = K_L*(DO_sat - DO) + DO_1 + DO_2 + DO_3
    return F_answer

def RK4_DO(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_DO(A, B, C, D, E, F)
    K2=(dt/86400)*F_DO(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_DO(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_DO(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Dissolved Inorganic Carbon
##############################################################################
def F_DIC(DIC, K_L, CO2_air, DIC_1, DIC_2):
    F_answer = K_L*(CO2_air - DIC) + DIC_1 + DIC_2
    return F_answer

def RK4_DIC(dt, A, B, C, D, E):
    K1=(dt/86400)*F_DIC(A, B, C, D, E)
    K2=(dt/86400)*F_DIC(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_DIC(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_DIC(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDOC: Labile Dissolved Organic Carbon
##############################################################################
def F_LDOC(LDOC, K_LDOC, K_LDOC_to_RDOC, gamma_OC, LDOC_1):
    F_answer = -K_LDOC*gamma_OC*LDOC - K_LDOC_to_RDOC*gamma_OC*LDOC + LDOC_1
    return F_answer

def RK4_LDOC(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDOC(A, B, C, D, E)
    K2=(dt/86400)*F_LDOC(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDOC(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDOC(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPOC: Labile Particulate Organic Carbon
##############################################################################
def F_LPOC(LPOC, K_LPOC, K_LPOC_to_RPOC, K_s_LPOC, H_dep, gamma_OC, LPOC_1):
    F_answer = ( -K_LPOC*gamma_OC*LPOC - K_LPOC_to_RPOC*gamma_OC*LPOC 
                 - (K_s_LPOC/H_dep)*LPOC + LPOC_1 )
    return F_answer

def RK4_LPOC(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPOC(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPOC(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPOC(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPOC(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDOC: Refractory Dissolved Organic Carbon
##############################################################################
def F_RDOC(RDOC, K_RDOC, gamma_OC, RDOC_1):
    F_answer = -K_RDOC*gamma_OC*RDOC + RDOC_1
    return F_answer

def RK4_RDOC(dt, A, B, C, D):
    K1=(dt/86400)*F_RDOC(A, B, C, D)
    K2=(dt/86400)*F_RDOC(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDOC(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDOC(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPOC: Refractory Particulate Organic Carbon
##############################################################################
def F_RPOC(RPOC, K_RPOC, gamma_OC, K_s_RPOC, H_dep, RPOC_1):
    F_answer = -K_RPOC*gamma_OC*RPOC - (K_s_RPOC/H_dep)*RPOC + RPOC_1
    return F_answer

def RK4_RPOC(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPOC(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPOC(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPOC(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPOC(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PIN: Particulate Inorganic Nitrogen
##############################################################################
def F_PIN(PIN, K_des_PIN, K_s_PIN, H_dep, PIN_1):
    F_answer = -K_des_PIN*PIN - (K_s_PIN/H_dep)*PIN + PIN_1
    return F_answer

def RK4_PIN(dt, A, B, C, D, E):
    K1=(dt/86400)*F_PIN(A, B, C, D, E)
    K2=(dt/86400)*F_PIN(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_PIN(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_PIN(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   NH4: Ammonium Nitrogen
##############################################################################
def F_NH4(NH4, K_NH4, gamma_NH4, PIN_ads, NH4_1, NH4_2):
    F_answer = -K_NH4*gamma_NH4*NH4 - PIN_ads*NH4 + NH4_1 + NH4_2
    return F_answer

def RK4_NH4(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_NH4(A, B, C, D, E, F)
    K2=(dt/86400)*F_NH4(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_NH4(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_NH4(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   NO3: Nitrate Nitrogen
##############################################################################
def F_NO3(NO3, K_NO3, gamma_NO3, K_s_NO3, H_dep, NO3_1):
    F_answer = - K_NO3*gamma_NO3*NO3 - (K_s_NO3/H_dep)*NO3 + NO3_1
    return F_answer

def RK4_NO3(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_NO3(A, B, C, D, E, F)
    K2=(dt/86400)*F_NO3(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_NO3(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_NO3(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDON: Labile Dissolved Organic Nitrogen
##############################################################################
def F_LDON(LDON, K_LDON, K_LDON_to_RDON, gamma_ON, LDON_1):
    F_answer = -K_LDON*gamma_ON *LDON - K_LDON_to_RDON*gamma_ON*LDON + LDON_1
    return F_answer

def RK4_LDON(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDON(A, B, C, D, E)
    K2=(dt/86400)*F_LDON(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDON(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDON(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPON: Labile Particulate Organic Nitrogen
##############################################################################
def F_LPON(LPON, K_LPON, K_LPON_to_RPON, K_s_LPON, H_dep, gamma_ON, LPON_1):
    F_answer = ( -K_LPON*gamma_ON*LPON - K_LPON_to_RPON*gamma_ON*LPON
                 - (K_s_LPON/H_dep)*LPON + LPON_1 )
    return F_answer

def RK4_LPON(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPON(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPON(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPON(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPON(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDON: Refractory Dissolved Organic Nitrogen
##############################################################################
def F_RDON(RDON, K_RDON, gamma_ON, RDON_1):
    F_answer = -K_RDON*gamma_ON*RDON + RDON_1
    return F_answer

def RK4_RDON(dt, A, B, C, D):
    K1=(dt/86400)*F_RDON(A, B, C, D)
    K2=(dt/86400)*F_RDON(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDON(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDON(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPON: Refractory Particulate Organic Nitrogen
##############################################################################
def F_RPON(RPON, K_RPON, gamma_ON, K_s_RPON, H_dep, RPON_1):
    F_answer = -K_RPON*gamma_ON*RPON - (K_s_RPON/H_dep)*RPON + RPON_1
    return F_answer

def RK4_RPON(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPON(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPON(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPON(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPON(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PIP: Particulate Inorganic Phosphorus
##############################################################################
def F_PIP(PIP, K_des_PIP, K_s_PIP, H_dep, PIP_1):
    F_answer = -K_des_PIP*PIP - (K_s_PIP/H_dep)*PIP + PIP_1
    return F_answer

def RK4_PIP(dt, A, B, C, D, E):
    K1=(dt/86400)*F_PIP(A, B, C, D, E)
    K2=(dt/86400)*F_PIP(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_PIP(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_PIP(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   PO4: Phosphate
##############################################################################
def F_PO4(PO4, PO4_des, PO4_1, PO4_2):
    F_answer = -PO4_des*PO4 + PO4_1 + PO4_2
    return F_answer

def RK4_PO4(dt, A, B, C, D):
    K1=(dt/86400)*F_PO4(A, B, C, D)
    K2=(dt/86400)*F_PO4(A+K1/2, B, C, D)
    K3=(dt/86400)*F_PO4(A+K2/2, B, C, D)
    K4=(dt/86400)*F_PO4(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LDOP: Labile Dissolved Organic Nitrogen
##############################################################################
def F_LDOP(LDOP, K_LDOP, K_LDOP_to_RDOP, gamma_OP, LDOP_1):
    F_answer = -K_LDOP*gamma_OP *LDOP - K_LDOP_to_RDOP*gamma_OP*LDOP + LDOP_1
    return F_answer

def RK4_LDOP(dt, A, B, C, D, E):
    K1=(dt/86400)*F_LDOP(A, B, C, D, E)
    K2=(dt/86400)*F_LDOP(A+K1/2, B, C, D, E)
    K3=(dt/86400)*F_LDOP(A+K2/2, B, C, D, E)
    K4=(dt/86400)*F_LDOP(A+K3, B, C, D, E)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   LPOP: Labile Particulate Organic Nitrogen
##############################################################################
def F_LPOP(LPOP, K_LPOP, K_LPOP_to_RPOP, K_s_LPOP, H_dep, gamma_OP, LPOP_1):
    F_answer = ( -K_LPOP*gamma_OP*LPOP - K_LPOP_to_RPOP*gamma_OP*LPOP
                 - (K_s_LPOP/H_dep)*LPOP +  + LPOP_1 )
    return F_answer

def RK4_LPOP(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_LPOP(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_LPOP(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_LPOP(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_LPOP(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RDOP: Refractory Dissolved Organic Nitrogen
##############################################################################
def F_RDOP(RDOP, K_RDOP, gamma_OP, RDOP_1):
    F_answer = -K_RDOP*gamma_OP*RDOP + RDOP_1
    return F_answer

def RK4_RDOP(dt, A, B, C, D):
    K1=(dt/86400)*F_RDOP(A, B, C, D)
    K2=(dt/86400)*F_RDOP(A+K1/2, B, C, D)
    K3=(dt/86400)*F_RDOP(A+K2/2, B, C, D)
    K4=(dt/86400)*F_RDOP(A+K3, B, C, D)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   RPOP: Refractory Particulate Organic Nitrogen
##############################################################################
def F_RPOP(RPOP, K_RPOP, gamma_OP, K_s_RPOP, H_dep, RPOP_1):
    F_answer = -K_RPOP*gamma_OP*RPOP - (K_s_RPOP/H_dep)*RPOP + RPOP_1
    return F_answer

def RK4_RPOP(dt, A, B, C, D, E, F):
    K1=(dt/86400)*F_RPOP(A, B, C, D, E, F)
    K2=(dt/86400)*F_RPOP(A+K1/2, B, C, D, E, F)
    K3=(dt/86400)*F_RPOP(A+K2/2, B, C, D, E, F)
    K4=(dt/86400)*F_RPOP(A+K3, B, C, D, E, F)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   ISS: Inorganic Suspended Solids: ISS
##############################################################################
def F_ISS(ISS, K_s_ISS, H_dep):
    F_answer = -(K_s_ISS/H_dep)*ISS
    return F_answer

def RK4_ISS(dt, A, B, C):
    K1=(dt/86400)*F_ISS(A, B, C)
    K2=(dt/86400)*F_ISS(A+K1/2, B, C)
    K3=(dt/86400)*F_ISS(A+K2/2, B, C)
    K4=(dt/86400)*F_ISS(A+K3, B, C)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Algae: Phytoplankton
##############################################################################
def F_ALG(ALG, K_ag, K_ar, K_ae, K_am, K_s_ALG, H_dep):
    F_answer = K_ag*ALG - (K_ar + K_ae + K_am)* ALG - (K_s_ALG/H_dep)*ALG
    return F_answer

def RK4_ALG(dt, A, B, C, D, E, F, G):
    K1=(dt/86400)*F_ALG(A, B, C, D, E, F, G)
    K2=(dt/86400)*F_ALG(A+K1/2, B, C, D, E, F, G)
    K3=(dt/86400)*F_ALG(A+K2/2, B, C, D, E, F, G)
    K4=(dt/86400)*F_ALG(A+K3, B, C, D, E, F, G)
    RK4_Answer = A + (K1+2*K2+2*K3+K4)/6
    return RK4_Answer

##############################################################################
#   Find minimum values
##############################################################################
@njit
def min_lambda(lambda_N, lambda_P, lambda_light):
    minimum = lambda_N
    for i in range(0, len(lambda_N)):
        if lambda_P[i] < lambda_N[i]:
            minimum[i] = lambda_P[i]
        elif lambda_light[i] < minimum[i]:
            minimum[i] = lambda_light[i]
        else:
            pass
    return minimum
