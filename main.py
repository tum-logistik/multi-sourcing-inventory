from env.SourcingEnv import *
from sim.sim_functions import *
from sim.policies import *
import numpy as np
from opt.mc_sim import *
import pickle
from datetime import datetime
import platform
import importlib 
from opt.eval_policy import *


if __name__ == '__main__':

    print("#### Running Main Subroutine ####")

    # optional import for git users
    if importlib.util.find_spec("git") is not None: 
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = str(repo.head.object.hexsha)
        branch_name = str(repo.active_branch)
    else:
        sha = "no_git"
        branch_name = "no_git"

    print("git branch: " + branch_name)
    print("git hash code: " + sha)

    sourcingEnv = SourcingEnv()

    s_initial = MState(stock_level = 0, 
        n_suppliers = N_SUPPLIERS, 
        n_backorders = np.array([0, 0]), 
        flag_on_off = np.array([1, 1]))

    # finding sS* policy for approx

    print ("Finding sS* policy")
    best_small_s, best_big_s, best_val = find_opt_ss_policy_via_mc(sourcingEnv,
        periods = 30,
        nested_mc_iters = 100,
        h_cost = H_COST,
        b_penalty = B_PENALTY,
        max_S = BIG_S)

    print("sS* policy: " + str((best_small_s, best_big_s, best_val)))
    output_dic = approx_value_iteration(sourcingEnv, s_initial, big_s = best_big_s, small_s = best_small_s)

    output_dic['model_params']['git_commit'] = sha
    output_dic['model_params']['branch_name'] = branch_name

    output_dic['model_params']['policy_params']['big_s'] = best_big_s
    output_dic['model_params']['policy_params']['small_s'] = best_small_s

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    write_path = 'output/msource_value_dic_{dt}.pkl'.format(dt = str(date_time)) if 'larkin' in platform.node() else 'output/msource_value_dic_{dt}.pkl'.format(dt = str(date_time))
    
    # Evaluate model
    mc_avg_costs = mc_with_policy(sourcingEnv, 
        periods = output_dic['model_params']['algo_params']['periods'],
        nested_mc_iters = output_dic['model_params']['algo_params']['nested_mc_iters'],
        big_s = output_dic['model_params']['policy_params']['big_s'],
        small_s = output_dic['model_params']['policy_params']['small_s'],
        h_cost = output_dic['model_params']['policy_params']['h_cost'],
        b_penalty = output_dic['model_params']['policy_params']['b_penalty'],
        policy_callback=dual_index_policy,
        use_tqdm = True)
    
    output_dic['approx_di_cost'] = np.mean(np.array(mc_avg_costs))

    eval_costs = mc_with_policy(sourcingEnv, 
        periods = output_dic['model_params']['algo_params']['periods'],
        nested_mc_iters = output_dic['model_params']['algo_params']['nested_mc_iters'],
        big_s = output_dic['model_params']['policy_params']['big_s'],
        small_s = output_dic['model_params']['policy_params']['small_s'],
        h_cost = output_dic['model_params']['policy_params']['h_cost'],
        b_penalty = output_dic['model_params']['policy_params']['b_penalty'],
        policy_callback=dual_index_policy,
        use_tqdm = True)
    
    kwargs = {
        "value_dic": output_dic["state_value_dic"], 
        "periods": 10, 
        "periods_val_it": 1,
        "nested_mc_iters": 30,
        "max_stock": BIG_S,
        "discount_fac": DISCOUNT_FAC,
        "h_cost": output_dic['model_params']['policy_params']['h_cost'],
        "b_penalty": output_dic['model_params']['policy_params']['b_penalty'],
        "n_visit_lim": N_VISIT_LIM,
        "default_ss_policy": ss_policy_fastest_supp_backlog,
        "safe_factor": SAFE_FACTOR,
        "sub_eval_periods": SUB_EVAL_PERIODS,
        "sub_nested_mc_iter": SUB_NESTED_MC_ITER,
        "max_stock": 2,
        "approx_eval": True
    }

    mc_avg_costs = mc_with_policy(sourcingEnv, 
        use_tqdm = True,
        policy_callback = eval_policy_from_value_dic,
        **kwargs)

    eval_costs_scaled = np.mean(mc_avg_costs)
    output_dic['adp_cost'] = np.mean(eval_costs_scaled)

    mc_avg_costs_ss_prime = mc_with_policy(sourcingEnv, 
        policy_callback = ss_policy_fastest_supp_backlog, 
        big_s = best_big_s,
        small_s = best_small_s,
        use_tqdm = False)
    
    output_dic['ss_cost'] = np.mean(np.array(mc_avg_costs_ss_prime))

    with open(write_path, 'wb') as handle:
        pickle.dump(output_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("algorithm complete, wrote file to: " + write_path)