import numpy as np
from env.SourcingEnv import *
from sim.policies import *
from sim.sim_functions import *
import time
from common.variables import *
from datetime import datetime
import pickle
from tqdm import tqdm


def mc_episode_with_policy(sourcingEnv, 
    policy = ss_policy_fastest_supp_backlog, 
    **kwargs):

    b_penalty = B_PENALTY if "b_penalty" not in kwargs else kwargs["b_penalty"]
    h_cost = H_COST if "h_cost" not in kwargs else kwargs["h_cost"]
    periods = PERIODS if "periods" not in kwargs else kwargs["periods"]
    
    sourcingEnv.reset()

    cost = cost_calc(sourcingEnv.current_state, h_cost = h_cost, b_penalty = b_penalty)
    total_costs = [cost]
    for i in range(periods):
        
        policy_action = policy(sourcingEnv, **kwargs)
        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)
        cost = cost_calc(next_state, h_cost = h_cost, b_penalty = b_penalty)
        
        total_procurement_cost = np.sum(np.multiply(policy_action, sourcingEnv.procurement_cost_vec))
        total_cost = cost + total_procurement_cost
        total_costs.append(total_cost)

    return np.sum(total_costs), np.sum(total_costs)/periods

def mc_with_policy(sourcingEnv, 
    start_state = False,
    policy_callback = ss_policy_fastest_supp_backlog, 
    nested_mc_iters = NESTED_MC_ITERS,
    use_tqdm = False,
    **kwargs):
    
    mc_avg_costs = []

    for i in tqdm(range(nested_mc_iters)) if use_tqdm else range(nested_mc_iters):
        if start_state != False:
            sourcingEnv.current_state = start_state

        start_time = time.time()
        _, avg_cost = mc_episode_with_policy(sourcingEnv, policy = policy_callback, **kwargs)
        mc_avg_costs.append(avg_cost)
        run_time = time.time() - start_time
        # if i % 100 == 0:
        #     print("time per 100 iter: " + str(run_time))
    
    return mc_avg_costs


def approx_value_iteration(sourcingEnv, initial_state, 
    max_steps = MAX_STEPS, 
    num_episodes = MC_EPISODES,
    max_stock = MAX_INVEN,
    discount_fac = DISCOUNT_FAC,
    explore_eps = EXPLORE_EPS,
    backorder_max = BACKORDER_MAX,
    max_inven = MAX_INVEN,
    model_args_dic = MODEL_ARGS_DIC,
    debug_bool = DEBUG_BOOL,
    learn_rate = FIXED_LEARN_RATE,
    small_s = SMALL_S, 
    big_s = BIG_S, ):
    # initialize random values.array
    # simulate 5x as a first guess, and use a uniform range
    
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    debug_write_path = 'output/debug_output_{dt}.txt'.format(dt = str(date_time)) if 'larkin' in platform.node() else 'output/debug_output_{dt}.txt'.format(dt = str(date_time))
    with open(debug_write_path, 'a') as f:
        f.write("####### DEBUG OUTPUT ####### \n")
        f.close()
        
    # initialize states, ex. dual sourcing 40k, 3x sourcing 800k states
    state_value_dic = {}
    sourcingEnv.reset()

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
                future_value = 0
                event_probs = sourcingEnv.get_event_probs(possible_joint_actions[pa])
                for i in range(len(event_probs)):
                    if event_probs[i] > 0:
                        sourcingEnvCopy = copy.deepcopy(sourcingEnv)
                        event_tuple = sourcingEnvCopy.get_event_tuple_from_index(i)
                        potential_state, _,_,_,_ = sourcingEnvCopy.step(possible_joint_actions[pa], event_tuple)
                        reward_contribution = - event_probs[i] * discount_fac * cost_calc(potential_state)
                        state_key = potential_state.get_repr_key()

                        # state_value_dic is a tuple (value, n_visits)
                        if state_key in state_value_dic and np.random.uniform(0, 1, 1)[0] > explore_eps:
                            avg_value_estimate = state_value_dic[state_key][0]
                            # Update state visit
                            state_value_dic[state_key]= (avg_value_estimate, state_value_dic[state_key][1] + 1)
                        else:
                            # there is a explore_eps chance of state-value re-estimation, and value update
                            value_estimates = mc_with_policy(sourcingEnvCopy, potential_state, big_s = big_s, small_s = small_s)
                            avg_value_estimate = -np.mean(value_estimates)

                            # value update on the MC explored states
                            if state_key not in state_value_dic:
                                state_value_dic[state_key] = (avg_value_estimate, 1)
                            else:
                                old_value = state_value_dic[state_key][0]
                                new_value_adap = (1 - learn_rate)*old_value + learn_rate*avg_value_estimate
                                state_value_dic[state_key] = (new_value_adap, state_value_dic[state_key][1] + 1)
                            
                            if debug_bool:
                                print("episode: {ep}  | step: {st} | potential_state: {ps}| vdic size: {vdic}".format(ep = str(e), st = str(m), ps = str(potential_state), vdic = str(len(state_value_dic))))
                            
                        # if np.random.uniform(0, 1, 1)[0] < explore_eps:
                        # else:
                        #     avg_value_estimate = np.mean(list(state_value_dic.values()))

                        future_value += reward_contribution + event_probs[i] * avg_value_estimate
                    action_reward = -np.sum(np.multiply(sourcingEnv.procurement_cost_vec, pa))

                value_array[pa] = action_reward + future_value
                
            # decide transition
            trans_ac_type = "step max V"
            if len(possible_joint_actions) > 0:
                if np.random.uniform(0, 1, 1)[0] < explore_eps:
                    action_index = np.random.randint(0, len(possible_joint_actions))
                    trans_ac_type = "eps explore"
                else:
                    if len(value_array) > 0:
                        action_index = np.argmax(value_array[np.nonzero(value_array)])  
                    else: 
                        action_index = np.random.randint(0, len(possible_joint_actions)) if len(possible_joint_actions) > 1 else None
            else:
                action_index = None
                trans_ac_type = "no selection of action"

            print(trans_ac_type)
            
            state_add = sourcingEnv.current_state.get_repr_key()

            # Value update on the current state
            if state_add not in state_value_dic and action_index is not None:
                state_value_dic[state_add] = (value_array[action_index], 1)
            elif state_add in state_value_dic and action_index is not None:
                old_value = state_value_dic[state_add][0]
                new_value_adap = (1 - learn_rate)*old_value + learn_rate*value_array[action_index]
                state_value_dic[state_add] = (new_value_adap, state_value_dic[state_add][1] + 1)

            if action_index != None and sourcingEnv.current_state.s <= max_inven:
                selected_action = possible_joint_actions[action_index]
            else:
                # otherwise do nothing
                selected_action = np.array([0, 0])
            
            next_state, event, _, _, supplier_index = sourcingEnv.step(selected_action)
            
            step_time = time.time() - step_start_time
            
            debug_trans_msg = "############ [STATE INFO] next_state: {ns} | event: {ev}| sel. act: {sa}| sup index: {sind}".format(ns= str(next_state), ev = str(event), sa = str(selected_action), sind = str(supplier_index))
            debug_count_msg = "############ [STEP TIME] episode: {ep} | step: {st}| elapsed time: {time}".format(ep = str(e), time = str(step_time), st = str(m))
            
            with open(debug_write_path, 'a') as f:
                f.write(trans_ac_type + "\n")
                f.write(debug_trans_msg + "\n")
                f.write(debug_count_msg + "\n")
                f.close()

            print(debug_trans_msg)
            print(debug_count_msg)

        # model save every save_interval intervals
        write_path = 'output/msource_value_dic_{dt}_interval.pkl'.format(dt = str(model_start_date_time)) if 'larkin' in platform.node() else 'output/msource_value_dic_{dt}.pkl'.format(dt = str(model_start_date_time))
        output_obj = {"state_value_dic": state_value_dic, "model_params": model_args_dic, "mdp_env": sourcingEnv}

        with open(write_path, 'wb') as handle:
            pickle.dump(output_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        episode_run_time = time.time() - episode_start_time
        print("############ episode: {ep} | elapsed time: {time}".format(ep = str(e), time = str(episode_run_time) ))
    
    return {"state_value_dic": state_value_dic, "model_params": model_args_dic, "mdp_env": sourcingEnv}


def find_opt_ss_policy_via_mc(sourcingEnv, 
    periods = PERIODS,
    nested_mc_iters = NESTED_MC_ITERS,
    h_cost = H_COST,
    b_penalty = B_PENALTY,
    max_S = MAX_INVEN):

    best_val = -np.Inf
    best_small_s = 0
    best_big_s = 1

    for small_s in range(0, max_S):
        for big_s in range(small_s+1, max_S):
            mc_avg_costs = mc_with_policy(sourcingEnv, 
                periods = periods,
                nested_mc_iters = nested_mc_iters,
                big_s = big_s,
                small_s = small_s,
                h_cost = h_cost,
                b_penalty = b_penalty)
            value = -np.mean(mc_avg_costs)
            if value > best_val:
                best_val = value
                best_big_s = big_s
                best_small_s = small_s

                print("new best value: " + str((best_small_s, best_big_s, best_val)))
    
    return best_small_s, best_big_s, best_val
