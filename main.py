from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *
import numpy as np
from opt.mc_sim import *
import pickle
from datetime import datetime
import platform

if __name__ == '__main__':
    print("#### Running Main Subroutine ####")

    sourcingEnv = SourcingEnv(
        lambda_arrival = LAMBDA, # or 10
        procurement_cost_vec = np.array([3, 1]),
        supplier_lead_times_vec = np.array([0.8, 0.5]),
        on_times = np.array([1, 1]), 
        off_times = np.array([0.3, 1]))

    s_initial = MState(stock_level = 0, 
        n_suppliers = N_SUPPLIERS, 
        n_backorders = np.array([0, 0]), 
        flag_on_off = np.array([1, 1]))
        
    state_value_dic = approx_value_iteration(sourcingEnv, s_initial)

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    write_path = 'output/msource_value_dic_{dt}.pkl'.format(dt = str(date_time)) if 'larkin' in platform.node() else 'workspace/mount/multi-sourcing-inventory/output/msource_value_dic_{dt}.pkl'.format(dt = str(date_time))
    
    with open(write_path, 'wb') as handle:
        pickle.dump(state_value_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("algorithm complete")