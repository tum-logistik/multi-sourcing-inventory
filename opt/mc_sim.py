import numpy as np
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *
from datetime import datetime
import pickle


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
        # if i % 100 == 0:
        #     print("time per 100 iter: " + str(run_time))
    
    return mc_avg_costs

def get_best_action():
    return 1

def approx_value_iteration(sourcingEnv, initial_state, 
    max_steps = MAX_STEPS, 
    num_episodes = MC_EPISODES,
    max_stock = MAX_INVEN,
    discount_fac = DISCOUNT_FAC,
    explore_eps = EXPLORE_EPS,
    backorder_max = BACKORDER_MAX,
    max_inven = MAX_INVEN,
    model_args_dic = MODEL_ARGS_DIC):
    # initialize random values.array
    # simulate 5x as a first guess, and use a uniform range
    
    mc_avg_costs = mc_with_ss_policy(sourcingEnv, start_state = initial_state)
    
    mean_cost = np.mean(mc_avg_costs)
    std = np.std(mc_avg_costs)

    value_ini_ub = -mean_cost + std
    value_ini_lb = -mean_cost - std

    # initialize states
    # dual sourcing 40k 
    # 3x sourcing 800k states
    sourcingEnv.reset()
     
    on_off_flags_combos = get_combo(2, sourcingEnv.n_suppliers)
    back_log_combos = get_combo(max_stock - backorder_max + 1, sourcingEnv.n_suppliers)

    state_value_dic = {}
    i = 0
    for stock in range(-backorder_max, max_inven+1):
        for b_combo in back_log_combos:
            for on_off_combo in on_off_flags_combos:
                state_add = MState(stock_level = stock, 
                    n_suppliers = sourcingEnv.n_suppliers, 
                    n_backorders = b_combo, 
                    flag_on_off = on_off_combo)
                
                state_value_dic[state_add.get_repr_key()] = np.random.uniform(value_ini_lb, value_ini_ub,1)[0]
                i += 1
                
    num_states = len(state_value_dic)

    # Iterate all episodes, do periodic MC update.
    now = datetime.now()
    model_start_date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    for e in range(num_episodes):
        episode_start_time = time.time()
        sourcingEnv.reset()
        for m in range(max_steps):
            step_start_time = time.time()
            # careful about backlog order
            possible_joint_actions = get_combo(int(max_stock - sourcingEnv.current_state.s), sourcingEnv.n_suppliers)
            # We know the transition probabilities
            value_array = np.zeros(len(possible_joint_actions))
            for pa in range(len(possible_joint_actions)):
                value = 0
                event_probs = sourcingEnv.get_event_probs(possible_joint_actions[pa])
                for i in range(len(event_probs)):
                    if event_probs[i] > 0:
                        sourcingEnvCopy = copy.deepcopy(sourcingEnv)
                        event_tuple = sourcingEnvCopy.get_event_tuple_from_index(i)
                        potential_state, _,_,_,_ = sourcingEnvCopy.step(possible_joint_actions[pa], event_tuple)
                        reward_contribution = - event_probs[i] * discount_fac * cost_calc(potential_state)
                        state_key = potential_state.get_repr_key()
                        if np.random.uniform(0, 1, 1)[0] < explore_eps:
                            value_estimates = mc_with_ss_policy(sourcingEnvCopy, potential_state)
                            avg_value_estimate = -np.mean(value_estimates)
                            state_value_dic[state_key] = avg_value_estimate
                            print("episode: {ep}  | step: {st} | potential_state: {ps}| pos ac: {pos_ac}".format(ep = str(e), st = str(m), pos_ac = str(pa), ps = str(potential_state)))
                        else:
                            if state_key in state_value_dic:
                                avg_value_estimate = state_value_dic[state_key]
                            else:
                                avg_value_estimate = np.mean(list(state_value_dic.values()))
                            
                        value += reward_contribution + avg_value_estimate
                value_array[pa] = value
            
            # decide transition
            if len(possible_joint_actions) > 0:
                if np.random.uniform(0, 1, 1)[0] < explore_eps:
                    action_index = np.random.randint(0, len(possible_joint_actions))
                    print("eps explore")
                else:
                    if len(value_array) > 0:
                        action_index = np.argmax(value_array[np.nonzero(value_array)])  
                    else: 
                        action_index = np.random.randint(0, len(possible_joint_actions)) if len(possible_joint_actions) > 1 else None
                print("step max V")
            else:
                action_index = None

            if action_index != None and sourcingEnv.current_state.s <= max_inven:
                sourcingEnv.step(possible_joint_actions[action_index])
            else:
                # otherwise do nothing
                sourcingEnv.step(np.array([0, 0]))
            
            step_time = time.time() - step_start_time
            print("############ [STEP TIME] episode: {ep} | step: {st}| elapsed time: {time}".format(ep = str(e), time = str(step_time), st = str(m) ))
        
        # model save every save_interval intervals
        write_path = 'output/msource_value_dic_{dt}_interval.pkl'.format(dt = str(model_start_date_time)) if 'larkin' in platform.node() else 'workspace/mount/multi-sourcing-inventory/output/msource_value_dic_{dt}.pkl'.format(dt = str(model_start_date_time))
        output_obj = {"state_value_dic": state_value_dic, "model_params": model_args_dic}

        with open(write_path, 'wb') as handle:
            pickle.dump(output_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        episode_run_time = time.time() - episode_start_time
        print("############ episode: {ep} | elapsed time: {time}".format(ep = str(e), time = str(episode_run_time) ))
    
    return {"state_value_dic": state_value_dic, "model_params": model_args_dic}