from env.SourcingEnv import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle as pkl
from env.HelperClasses import *
from sim.sim_functions import *
import itertools

filename = "output/msource_value_dic_07-04-2022-05-59-13.pkl"

with open(filename, 'rb') as f:
    output_obj = pkl.load(f)

    value_dic = output_obj["state_value_dic"]
    model_params = output_obj["model_params"]
    sourcingEnv = output_obj["mdp_env"]

    sourcingEnv = SourcingEnv(
        lambda_arrival = model_params['mdp_env_params']['lambda'], # or 10
        procurement_cost_vec = np.array(model_params['mdp_env_params']['procurement_cost_vec']),
        supplier_lead_times_vec = np.array(model_params['mdp_env_params']['supplier_lead_times_vec']),
        on_times = np.array([1, 1]), 
        off_times = np.array([np.Inf, np.Inf]))


state_backorders = list(itertools.product(range(MAX_INVEN - BACKORDER_MAX), range(sourcingEnv.n_suppliers)))
state_s = list(range(MAX_INVEN - BACKORDER_MAX))
state_onoff = list(itertools.product(range(2), range(sourcingEnv.n_suppliers)))

possible_state_tuples = list(itertools.product(state_s, state_backorders, state_onoff))
poss_states = [[x[0], np.array(list(x[1])), np.array(list(x[2]))] for x in possible_state_tuples]

action_space_tup = list(itertools.product(range(sourcingEnv.action_size), range(sourcingEnv.n_suppliers))) 
action_space = [np.array(list(x)) for x in action_space_tup]

m = gp.Model("MDP")
x = {}

for state in poss_states:
    for a in action_space:
        state_rep = MState(state[0], sourcingEnv.n_suppliers, state[1], state[2])
        a_rep = repr(list(a))
        cost = cost_calc_expected_di(sourcingEnv, a, custom_state = state_rep)
        x[state_rep, a_rep] = m.addVar(obj = cost, name='x-'+str(state)+"-"+str(a))
        m.addConstr(x[state_rep, a_rep] >= 0)
        

# for state in poss_states:
#     tp = sourcingEnv.compute_trans_prob()
#     m.addConstr(sum(x[state, a] for a in action_space) - sum(tp[j][a][i]*x[j,a] for j in poss_states for a in range(sourcingEnv.action_size)) == 0)

# m.addConstr(sum(x[i,a] for i in range(poss_states) for a in range(poss_states)) == 1)