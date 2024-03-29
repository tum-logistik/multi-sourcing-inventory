import sys
from env.SourcingEnv import *
import numpy as np
import gurobipy as gp
import pickle as pkl
from env.HelperClasses import *
from sim.sim_functions import *
import itertools
import json
import pickle as pkl
from datetime import datetime

GRB = gp.GRB

# if len(sys.argv[1]) > 1:
#     filename = sys.argv[1]
# else:
#     filename = "output/msource_value_dic_12-22-2022-18-08-46.pkl"


def lp_solver(filename):

    print("Reading from: " + filename)

    with open(filename, 'rb') as f:
        output_obj = pkl.load(f)

        value_dic = output_obj["state_value_dic"]
        model_params = output_obj["model_params"]
        sourcingEnv = output_obj["mdp_env"]

        off_times = np.array([np.Inf, np.Inf]) if cfg['mdp_env_params']['off_times'] == "no_disrup" else np.array(cfg['mdp_env_params']['off_times'])

        # sourcingEnv = SourcingEnv(
        #     lambda_arrival = model_params['mdp_env_params']['lambda'], # or 10
        #     procurement_cost_vec = np.array(model_params['mdp_env_params']['procurement_cost_vec']),
        #     supplier_lead_times_vec = np.array(model_params['mdp_env_params']['supplier_lead_times_vec']),
        #     on_times = np.array(model_params['mdp_env_params']['on_times']), 
        #     off_times = off_times)

    state_s = list(range(BACKORDER_MAX_LP, MAX_INVEN_LP))
    state_backorders = list(itertools.product(range(MAX_INVEN_LP - BACKORDER_MAX_LP), range(sourcingEnv.n_suppliers)))
    state_onoff = list(itertools.product(range(2), range(sourcingEnv.n_suppliers)))

    possible_state_tuples = list(itertools.product(state_s, state_backorders, state_onoff))
    poss_states = [[x[0], np.array(list(x[1])), np.array(list(x[2]))] for x in possible_state_tuples]


    action_space_tup = [x for x in itertools.product(*([list(range(ACTION_SIZE_LP))]*sourcingEnv.n_suppliers)) ]

    # action_space_tup = list(itertools.product(range(sourcingEnv.action_size), range(sourcingEnv.n_suppliers))) 
    action_space = [np.array(list(x)) for x in action_space_tup]

    for s in poss_states:
        for i in range(1, len(s)):
            s[i] = list(s[i])

    poss_states_set = set([repr(v) for v in poss_states])

    ################################################################


    # need to write a pij function
    # tau = sourcingEnv.compute_event_arrival_time(a)

    m = gp.Model("MDP")
    u = {}

    def add_in_additional_var(state_obj, action):
        s_a_tuple = (state_obj.get_nested_list_repr(), repr(list(action)))
        if s_a_tuple not in u:
            tau_s_a = sourcingEnv.compute_event_arrival_time(action, state_obj = state_obj)
            state_cost = cost_calc_unit(state_obj)
            action_cost = np.sum(np.multiply(sourcingEnv.procurement_cost_vec, action))

            cost = (state_cost + action_cost) / tau_s_a # + action_cost # /tau_s_a
            u[state_obj.get_nested_list_repr(), repr(list(action))] = m.addVar(obj = cost, name='var_x..' + state_obj.get_nested_list_repr() + ".." + str(action), vtype=GRB.CONTINUOUS) # default lower bound of 0, obj = cost
            m.addConstr(u[state_obj.get_nested_list_repr(), repr(list(action))] >= 0.0)
            return True

    for state in poss_states:
        for a in action_space:
            state_rep = MState(state[0], sourcingEnv.n_suppliers, np.array(state[1]), np.array(state[2]))
            add_in_additional_var(state_rep, a)
            
    # need to write a pij function
    # tau = sourcingEnv.compute_event_arrival_time(a)

    poss_states_new = copy.deepcopy(poss_states)
    for j_state in poss_states:
        j_state_obj = MState(j_state[0], sourcingEnv.n_suppliers, np.array(j_state[1]), np.array(j_state[2]))   
        poss_i_states_tuples = [] # possible prev. states
        for a_i in action_space:
            sourcingEnv.current_state = j_state_obj
            event_probs = sourcingEnv.get_event_probs(a_i)
            all_events = [Event.DEMAND_ARRIVAL, Event.SUPPLY_ARRIVAL, Event.SUPPLIER_ON, Event.SUPPLIER_OFF]
            for k in range(sourcingEnv.n_suppliers):

                # DEMAND_ARRIVAL
                if BACKORDER_MAX_LP < j_state[0] - 1 < MAX_INVEN_LP:
                    i_state_supp = copy.deepcopy(j_state[1])
                    i_state_supp = np.clip(j_state[1] - a_i, 0, MAX_INVEN_LP)
                    j_state_s = np.clip(j_state[0] - 1, BACKORDER_MAX_LP, MAX_INVEN_LP)
                    change_i_state = MState(j_state_s, sourcingEnv.n_suppliers, np.array(i_state_supp), np.array(j_state[2]))
                    poss_i_states_tuples.append((a_i, change_i_state, event_probs[0])) # Event DEMAND_ARRIVAL
                    
                    if change_i_state.get_nested_list() not in poss_states_new:
                        poss_states_new.append(change_i_state.get_nested_list())
                    s_a_tuple = (change_i_state.get_nested_list_repr(), repr(list(a_i)))
                    if s_a_tuple not in u:
                        add_in_additional_var(change_i_state, a_i)

                # SUPPLY_ARRIVAL
                if BACKORDER_MAX_LP < j_state[0] + 1 < MAX_INVEN_LP:
                    i_state_supp = copy.deepcopy(j_state[1])
                    i_state_supp = np.clip(j_state[1] - a_i, 0, MAX_INVEN_LP)
                    i_state_supp[k] = np.clip(j_state[1][k] + 1, 0, MAX_INVEN_LP)
                    j_state_s = np.clip(j_state[0] - 1, BACKORDER_MAX_LP, MAX_INVEN_LP)
                    change_i_state = MState(j_state_s, sourcingEnv.n_suppliers, np.array(i_state_supp), np.array(j_state[2]))
                    index = sourcingEnv.get_event_index_from_event(Event.SUPPLY_ARRIVAL, k)
                    poss_i_states_tuples.append((a_i, change_i_state, event_probs[index])) # Event SUPPLY_ARRIVAL
                
                    if change_i_state.get_nested_list() not in poss_states_new:
                        poss_states_new.append(change_i_state.get_nested_list())
                    s_a_tuple = (change_i_state.get_nested_list_repr(), repr(list(a_i)))
                    if s_a_tuple not in u:
                        add_in_additional_var(change_i_state, a_i)

                # # ON OFF
                i_state_v = copy.deepcopy(j_state[2])
                if j_state[2][k] == 1:
                    i_state_v[k] = 0
                    v_event = Event.SUPPLIER_ON
                elif j_state[2][k] == 0:
                    i_state_v[k] = 1
                    v_event = Event.SUPPLIER_OFF

                change_i_state = MState(j_state[0], sourcingEnv.n_suppliers, np.array(j_state[1]), np.array(i_state_v))
                index = sourcingEnv.get_event_index_from_event(v_event, k)
                poss_i_states_tuples.append((a_i, change_i_state, event_probs[index]))

                if change_i_state.get_nested_list() not in poss_states_new:
                    poss_states_new.append(change_i_state.get_nested_list())
                s_a_tuple = (change_i_state.get_nested_list_repr(), repr(list(a_i)))
                if s_a_tuple not in u:
                    add_in_additional_var(change_i_state, a_i)
        
        m.addConstr(gp.quicksum(u[j_state_obj.get_nested_list_repr(), repr(list(a))] for a in action_space) >= VERY_SMALL_NUMBER) # Safety
        m.addConstr(gp.quicksum(pij*u[state_i.get_nested_list_repr(), repr(list(a_i2))] for (a_i2, state_i, pij) in poss_i_states_tuples) >= VERY_SMALL_NUMBER) # Safety
        
        m.addConstr(gp.quicksum(u[j_state_obj.get_nested_list_repr(), repr(list(a))] for a in action_space) - gp.quicksum(pij*u[state_i.get_nested_list_repr(), repr(list(a_i2))] for (a_i2, state_i, pij) in poss_i_states_tuples) == 0)

    # custom_state = MState(stock_level = -3, n_suppliers = N_SUPPLIERS, n_backorders = np.array([0, 0]), flag_on_off = np.array([1, 1]))
    # custom_action = np.array([1,1])
    # m.addConstr(u[custom_state.get_nested_list_repr(), repr(list(custom_action))] == 0.000004)

    # sa_keys = []
    # for state_i in poss_states:
    #     for a in action_space:
    #         sa_keys.append((str(MState(state_i[0], sourcingEnv.n_suppliers, np.array(state_i[1]), np.array(state_i[2])) ), repr(list(a))))

    poss_states = copy.deepcopy(poss_states_new)
    poss_states_objs = [MState(state[0], sourcingEnv.n_suppliers, state[1], state[2]) for state in poss_states]

    # for p in poss_states:
    #     print(p)

    for state_obj in poss_states_objs:
        # Forced Safety
        if state_obj.s < 0:
            none_action = np.array([0, 0])
            m.addConstr(u[state_obj.get_nested_list_repr(), repr(list(none_action))] == 0.0)
        for a in action_space:
            s_a_tuple = (state_obj.get_nested_list_repr(), repr(list(a)))
            if s_a_tuple not in u:
                add_in_additional_var(s, a)
            

    m.addConstr(gp.quicksum(sourcingEnv.compute_event_arrival_time(a, state_obj = state_i)*u[state_i.get_nested_list_repr(), repr(list(a))] for state_i in poss_states_objs for a in action_space) == 1)

    ################################################################

    # m.setObjective(GRB.MINIMIZE)

    m.optimize()
    m.write("output_lp/model_lp_2source.sol")
    m.printStats()
    m.printAttr('x')
    # print("all variables")
    # Optimal Policy 
    # for state in poss_states:
    #     for a in action_space:
    #         guro_var = m.getVarByName('var_x..' + repr(state) + ".." + str(a))
    #         # if guro_var is not None and guro_var.X > 0:
    #         print(guro_var)

    data = json.loads(m.getJSONSolution())
    nz_sol = [(x['VarName'], x['X']) for x in data['Vars']]

    # now = datetime.now()
    # date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    filename_lp = filename.split("output/")
    write_path = 'output/lp_sol_{str_name}'.format(str_name = str(filename_lp[1]))

    with open(write_path, 'wb') as handle:
        pkl.dump(nz_sol, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    return nz_sol, write_path


