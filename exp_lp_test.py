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
        order_quantity = ACTION_SIZE,
        lambda_arrival = model_params['mdp_env_params']['lambda'], # or 10
        procurement_cost_vec = np.array(model_params['mdp_env_params']['procurement_cost_vec']),
        supplier_lead_times_vec = np.array(model_params['mdp_env_params']['supplier_lead_times_vec']),
        on_times = np.array([1, 1]), 
        off_times = np.array([np.Inf, np.Inf]))


state_s = list(range(BACKORDER_MAX_LP, MAX_INVEN_LP))
state_backorders = list(itertools.product(range(MAX_INVEN_LP - BACKORDER_MAX_LP), range(sourcingEnv.n_suppliers)))
state_onoff = list(itertools.product(range(2), range(sourcingEnv.n_suppliers)))

possible_state_tuples = list(itertools.product(state_s, state_backorders, state_onoff))
poss_states = [[x[0], np.array(list(x[1])), np.array(list(x[2]))] for x in possible_state_tuples]


action_space_tup = [x for x in itertools.product(*([list(range(sourcingEnv.action_size))]*sourcingEnv.n_suppliers)) ]

# action_space_tup = list(itertools.product(range(sourcingEnv.action_size), range(sourcingEnv.n_suppliers))) 
action_space = [np.array(list(x)) for x in action_space_tup]

for s in poss_states:
    for i in range(1, len(s)):
        s[i] = list(s[i])


#############


m = gp.Model("MDP")
x = {}

for state in poss_states:
    for a in action_space:
        state_rep = MState(state[0], sourcingEnv.n_suppliers, state[1], state[2])
        state_rep_str = str(state_rep)
        a_rep = repr(list(a))
        cost = -cost_calc_expected_di(sourcingEnv, a, custom_state = state_rep)
        x[state_rep_str, a_rep] = m.addVar(obj = cost, name='x-'+str(state)+"-"+str(a))
        m.addConstr(x[state_rep_str, a_rep] >= 0)
        
# need to write a pij function
# tau = sourcingEnv.compute_event_arrival_time(a)

def add_in_additional_constr(change_i_state, a_i, x, m):
    if (str(change_i_state), repr(list(a_i))) not in x:
        cost = cost_calc_expected_di(sourcingEnv, a_i, custom_state = change_i_state)
        x[str(change_i_state), repr(list(a_i))] = m.addVar(obj = cost, name='x-'+str(state)+"-"+str(a))
        m.addConstr(x[state_rep_str, a_rep] >= 0)
    return m

poss_states_new = copy.deepcopy(poss_states)
for j_state in poss_states:
    j_state_obj = MState(j_state[0], sourcingEnv.n_suppliers, j_state[1], j_state[2])   
    poss_i_states_tuples = [] # possible prev. states
    for a_i in action_space:
        event_probs = sourcingEnv.get_event_probs(a_i)
        for k in range(sourcingEnv.n_suppliers):
            i_state_supp = copy.deepcopy(j_state[1])
            i_state_supp[k] = j_state[1][k] - a_i[k]
            change_i_state = MState(j_state[0] + 1, sourcingEnv.n_suppliers, i_state_supp, j_state[2])
            poss_i_states_tuples.append((a_i, change_i_state, event_probs[0])) # Event DEMAND_ARRIVAL
            
            poss_states_new.append(change_i_state.get_nested_list_repr())
            m = add_in_additional_constr(change_i_state, a_i, x, m)
            
            i_state_supp = copy.deepcopy(j_state[1])
            i_state_supp[k] = j_state[1][k] - a_i[k] + 1
            change_i_state = MState(j_state[0] - 1, sourcingEnv.n_suppliers, i_state_supp, j_state[2])
            index = sourcingEnv.get_event_index_from_event(Event.SUPPLY_ARRIVAL, k)
            poss_i_states_tuples.append((a_i, change_i_state, event_probs[index])) # Event SUPPLY_ARRIVAL
            
            poss_states_new.append(change_i_state.get_nested_list_repr())
            m = add_in_additional_constr(change_i_state, a_i, x, m)

            i_state_v = copy.deepcopy(j_state[2])
            if j_state[2][k] == 1:
                i_state_v[k] = 0
                change_i_state = MState(j_state[0], sourcingEnv.n_suppliers, j_state[1], i_state_v)
                index = sourcingEnv.get_event_index_from_event(Event.SUPPLIER_ON, k)
                poss_i_states_tuples.append((a_i, change_i_state, event_probs[index])) # Event SUPPLY_ARRIVAL
                # m = add_in_additional_constr(change_i_state, a_i, x, m)
            
            i_state_v = copy.deepcopy(j_state[2])
            if j_state[2][k] == 0:
                i_state_v[k] = 1
                change_i_state = MState(j_state[0], sourcingEnv.n_suppliers, j_state[1], i_state_v)
                index = sourcingEnv.get_event_index_from_event(Event.SUPPLIER_OFF, k)
                poss_i_states_tuples.append((a_i, change_i_state, event_probs[index])) # Event SUPPLY_ARRIVAL
                # m = add_in_additional_constr(change_i_state, a_i, x, m)
            
    m.addConstr(sum(x[str(j_state_obj), repr(list(a))] for a in action_space) - sum(pij*x[str(state_i), repr(list(a_i))] for (a_i, state_i, pij) in poss_i_states_tuples) == 0)

poss_states = copy.deepcopy(poss_states_new)

sa_keys = []

for state_i in poss_states:
    for a in action_space:
        sa_keys.append((str(MState(state_i[0], sourcingEnv.n_suppliers, np.array(state_i[1]), np.array(state_i[2])) ), repr(list(a))))

m.addConstr(sum(sourcingEnv.compute_event_arrival_time(a)*x[str(MState(state_i[0], sourcingEnv.n_suppliers, state_i[1], state_i[2])), repr(list(a))] for state_i in poss_states for a in action_space) == 1)

# for state in poss_states:
#     tp = sourcingEnv.compute_trans_prob()
#     m.addConstr(sum(x[state, a] for a in action_space) - sum(tp[j][a][i]*x[j,a] for j in poss_states for a in range(sourcingEnv.action_size)) == 0)

# m.addConstr(sum(x[i,a] for i in range(poss_states) for a in range(poss_states)) == 1)

m.optimize()

# Optimal Policy 
for state in poss_states_new:
    for a in action_space:
        guro_var = m.getVarByName('x-' + str(state) +"-" + str(a))
        if guro_var is not None and guro_var.X > 0:
            print(guro_var)