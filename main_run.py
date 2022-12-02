from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *
import numpy as np
from opt.mc_sim import *
from opt.eval_policy import *
import pickle as pkl
from opt.eval_policy import *

if __name__ == '__main__':

    print("#### Running Debug Scenario #####")
    filename = "output/msource_value_dic_12-01-2022-13-35-10.pkl"

    with open(filename, 'rb') as f:
        output_obj = pkl.load(f)

    value_dic = output_obj["state_value_dic"]
    model_params = output_obj["model_params"]
    sourcingEnv = output_obj["mdp_env"]

    sourcingEnv2 = SourcingEnv(
        lambda_arrival = model_params['mdp_env_params']['lambda'], # or 10
        procurement_cost_vec = np.array(model_params['mdp_env_params']['procurement_cost_vec']),
        supplier_lead_times_vec = np.array(model_params['mdp_env_params']['supplier_lead_times_vec']),
        fixed_costs = np.array(model_params['mdp_env_params']['fixed_costs']) if 'fixed_costs' in model_params['mdp_env_params'] else FIXED_COST_VEC,
        on_times = np.array([1, 1]), 
        off_times = np.array([np.Inf, np.Inf]))
    
    cost = cost_calc(sourcingEnv2.current_state, h_cost = 4, b_penalty = 6)
    print(str(sourcingEnv2.current_state) + " cost: " + str(cost))

    s_custom = MState(stock_level = 0, 
        n_suppliers = N_SUPPLIERS, 
        n_backorders = np.array([0, 0]), 
        flag_on_off = np.array([1, 1]))

    kwargs = {
        "value_dic": value_dic, 
        "periods": 100, 
        "periods_val_it": 30,
        "nested_mc_iters": 4,
        "max_stock": 2,
        "discount_fac": DISCOUNT_FAC,
        "h_cost": model_params['policy_params']['h_cost'],
        "b_penalty": model_params['policy_params']['b_penalty'],
        "n_visit_lim": N_VISIT_LIM,
        "default_ss_policy": ss_policy_fastest_supp_backlog,
        "safe_factor": 1.1, #SAFE_FACTOR,
        "sub_eval_periods": SUB_EVAL_PERIODS,
        "sub_nested_mc_iter": SUB_NESTED_MC_ITER,
        "approx_eval": True
    }

    sourcingEnv2.lambda_arrival = 50
    # sourcingEnv2.mu_lt_rate = np.array([12.5, 5.5])
    mc_avg_costs = mc_with_policy(sourcingEnv2, 
        start_state = s_custom, 
        use_tqdm = False,
        policy_callback = dummy_explore_policy,
        **kwargs
    )

    mc_avg_costs = mc_with_policy(sourcingEnv2, 
        start_state = s_custom, 
        use_tqdm = False,
        policy_callback = eval_policy_from_value_dic,
        **kwargs
    )

    # single_supplier_mean_costs = []
    # for s in range(sourcingEnv2.n_suppliers):

    #     kwargs = {"periods" : 30,
    #         "nested_mc_iters" : 30,
    #         "h_cost": model_params['policy_params']['h_cost'],
    #         "b_penalty" : model_params['policy_params']['b_penalty'],
    #         "supplier_index": s
    #     }

    #     single_supplier_costs = mc_with_policy(sourcingEnv2, start_state = s_custom, 
    #         use_tqdm = True,
    #         policy_callback = single_source_orderupto_policy,
    #         **kwargs)
        
    #     single_supplier_mean_costs.append(np.mean(single_supplier_costs))

    # print(single_supplier_mean_costs)
    # print(np.min(single_supplier_mean_costs))

    # sourcingEnv2.lambda_arrival = 50
    # sourcingEnv2.supplier_lead_times_vec = np.array([0.008, 0.04])
    # sourcingEnv2.mu_lt_rate = np.array([1/0.008, 1/0.04])

    mc_avg_costs = mc_with_policy(sourcingEnv2, 
        start_state = s_custom, 
        use_tqdm = True,
        policy_callback = eval_policy_from_value_dic,
        **kwargs)
    
    mc_avg_costs = mc_with_policy(sourcingEnv2, 
        start_state = s_custom, 
        use_tqdm = True,
        policy_callback = myopic2_policy,
        **kwargs)

    mc_avg_costs_di = mc_with_policy(sourcingEnv2, start_state = s_custom, 
        periods = 20,
        nested_mc_iters = 20,
        big_s = model_params['policy_params']['big_s'],
        small_s = model_params['policy_params']['small_s'],
        h_cost = model_params['policy_params']['h_cost'],
        b_penalty = model_params['policy_params']['b_penalty'],
        policy_callback=dual_index_policy,
        use_tqdm = True)
    

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
        
    mc_with_policy(sourcingEnv, value_dic, 
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

