import numpy as np
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *

def mc_episode_with_ss_policy(sourcingEnv, 
    h_cost = H_COST, 
    b_penalty = B_PENALTY, 
    small_s = SMALL_S, 
    big_s = BIG_S, 
    periods = PERIODS, 
    ss_policy = ss_policy_fastest_supp_backlog):

    cost = cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    total_costs = [cost]
    for i in range(periods):
        
        policy_action = ss_policy(sourcingEnv, small_s = small_s, big_s = big_s)
        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)
        cost = cost_calc(next_state, h_cost = h_cost, b_penalty = b_penalty)
        total_costs.append(cost)

    return np.sum(total_costs), np.sum(total_costs)/periods

def mc_with_ss_policy(sourcingEnv, 
    start_state = False,
    h_cost = H_COST, 
    b_penalty = B_PENALTY, 
    small_s = SMALL_S, 
    big_s = BIG_S, 
    periods = PERIODS, 
    ss_policy = ss_policy_fastest_supp_backlog, 
    nested_mc_iters = NESTED_MC_ITERS):
    
    mc_avg_costs = []

    for i in range(nested_mc_iters):
        if start_state != False:
            sourcingEnv.current_state = start_state

        start_time = time.time()
        _, avg_cost = mc_episode_with_ss_policy(sourcingEnv, h_cost = h_cost, b_penalty = b_penalty, small_s = small_s, big_s = big_s, periods = periods, ss_policy = ss_policy)
        mc_avg_costs.append(avg_cost)
        run_time = time.time() - start_time
        if i % 100 == 0:
            print("time per 100 iter: " + str(run_time))
    
    return mc_avg_costs

def approx_value_iteration(sourcingEnv, initial_state):
    # initialize random values.array
    # simulate 5x as a first guess, and use a uniform range
    
    mc_avg_costs = mc_with_ss_policy(sourcingEnv, start_state = initial_state)
    
    mean_cost = np.mean(mc_avg_costs)
    std = np.std(mc_avg_costs)

    value_ini_ub = mean_cost + std
    value_ini_lb = mean_cost - std

    # initialize states
    # dual sourcing 40k 
    # 3x sourcing 800k states
    
    on_off_flags_combos = np.array(np.meshgrid(*[[0, 1] for x in range(sourcingEnv.n_suppliers)])).T.reshape(-1,sourcingEnv.n_suppliers)
    back_log_combos = np.array(np.meshgrid(*[range(0, sourcingEnv.action_size) for x in range(sourcingEnv.n_suppliers)])).T.reshape(-1,sourcingEnv.n_suppliers)

    state_dic = {}
    i = 0
    for stock in range(-STOCK_BOUND, STOCK_BOUND):
        for b_combo in back_log_combos:
            for on_off_combo in on_off_flags_combos:
                state_add = MState(stock_level = stock, 
                    n_suppliers = sourcingEnv.n_suppliers, 
                    n_backorders = b_combo, 
                    flag_on_off = on_off_combo)
                
                state_dic[state_add.get_repr_key()] = np.random.uniform(value_ini_lb, value_ini_ub,1)[0]
                i += 1
                if i % 100 == 0:
                    print("state added: " + str(state_add)) 
    num_states = len(state_dic)




    return state_dic