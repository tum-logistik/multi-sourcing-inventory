from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *
import numpy as np
from opt.mc_sim import *
from opt.eval_policy import *
import pickle as pkl


if __name__ == '__main__':
    print("#### Running Main Simulation ####")

    # sourcingEnv = SourcingEnv(
    #     lambda_arrival = 8, # or 10
    #     procurement_cost_vec = np.array([3, 1, 2]),
    #     supplier_lead_times_vec = np.array([0.8, 0.5, 1.0]),
    #     on_times = np.array([1, 1, 2]), 
    #     off_times = np.array([0.3, 1, 0.2]))
    
    sourcingEnv = SourcingEnv()
    
    pp = PROCUREMENT_COST_VEC

    filename = "output/msource_value_dic_07-02-2022-06-43-07.pkl"

    with open(filename, 'rb') as f:
        output_obj = pkl.load(f)

    value_dic = output_obj["state_value_dic"]
    model_params = output_obj["model_params"]

    mc_avg_costs = mc_with_policy(sourcingEnv, 
        periods = 30,
        nested_mc_iters = 100,
        big_s = model_params['policy_params']['big_s'],
        small_s = model_params['policy_params']['small_s'],
        h_cost = model_params['policy_params']['h_cost'],
        b_penalty = model_params['policy_params']['b_penalty'],
        use_tqdm = True,
        policy_callback = single_source_orderupto_policy)

    
    eval_steps = 50
    mc_eval_iter = 3
    mc_eval_policy_perf(sourcingEnv, value_dic, 
        max_steps = eval_steps, mc_iters = mc_eval_iter)
    
    history = []

    cost = cost_calc(sourcingEnv.current_state, h_cost = 4, b_penalty = 6)
    print(str(sourcingEnv.current_state) + " cost: " + str(cost))

    total_costs = [cost]

    periods = PERIODS
    for i in range(periods):
        
        # totally random order amounts
        random_action = np.array([np.random.randint(0, 2) for x in range(sourcingEnv.n_suppliers)])

        # s S policy with aribtrary s S, and randomly selected vendor
        policy_action = ss_policy_rand_supp(sourcingEnv, small_s = 3, big_s = 15)
        kwargs = {"small_s": 3, "big_s": 10}
        policy_action = ss_policy_fastest_supp_backlog(sourcingEnv, **kwargs)
        
        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(policy_action)

        cost = cost_calc(next_state, h_cost = 4, b_penalty = 6)
        print(str(next_state) + " cost: " + str(cost) + " event: " + str(event) + " action:" + str(policy_action))
        
        total_costs.append(cost)
        history.append([next_state, event, i, probs])

        
    print("#### Total Cost: " + str(np.sum(total_costs)) + " / Avg. Cost: " +  str(np.sum(total_costs)/periods) )
