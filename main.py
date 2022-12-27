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
from env.email_functions import *
import json

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

    output_dic = approx_value_iteration(sourcingEnv, s_initial, 
        big_s = best_big_s, 
        small_s = best_small_s)

    output_dic['model_params']['git_commit'] = sha
    output_dic['model_params']['branch_name'] = branch_name

    output_dic['model_params']['policy_params']['big_s'] = best_big_s
    output_dic['model_params']['policy_params']['small_s'] = best_small_s
    node_name = platform.node()
    output_dic['model_params']['docker_id'] = node_name

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    
    output_obj_path = 'msource_value_dic_{dt}.pkl'.format(dt = str(date_time)) if 'larkin' in platform.node() else 'msource_value_dic_{dt}.pkl'.format(dt = str(date_time))
    write_path = "output/" + output_obj_path

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
    
    output_dic['di_cost'] = np.mean(np.array(mc_avg_costs))

    mc_avg_costs_ss_prime = mc_with_policy(sourcingEnv, 
        policy_callback = ss_policy_fastest_supp_backlog, 
        big_s = best_big_s,
        small_s = best_small_s,
        use_tqdm = False)
    
    output_dic['ss_cost'] = np.mean(np.array(mc_avg_costs_ss_prime))

    mc_avg_costs_ssn = mc_with_policy(sourcingEnv, 
        policy_callback = single_source_orderupto_policy, 
        big_s = best_big_s,
        small_s = best_small_s,
        use_tqdm = False)
    
    output_dic['ssn_cost'] = np.mean(np.array(mc_avg_costs_ssn))

    myopic_cost = mc_with_policy(sourcingEnv, 
        periods = EVAL_PERIODS,
        nested_mc_iters = 5,
        big_s = best_small_s,
        small_s = best_big_s,
        max_order = BIG_S,
        policy_callback=myopic2_policy,
    use_tqdm = True)

    print(np.mean(myopic_cost))
    print(np.median(np.array(myopic_cost)))
    print(np.std(np.array(myopic_cost)))

    output_dic['myopic_cost'] = np.mean(np.array(myopic_cost))

    print("Running ADP eval: writing temp file to : " + write_path)
    with open(write_path, 'wb') as handle:
        pickle.dump(output_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    kwargs = {
        "value_dic": output_dic["state_value_dic"], 
        "periods": EVAL_PERIODS, 
        "periods_val_it": 1,
        "nested_mc_iters": NESTED_MC_ITERS,
        "discount_fac": DISCOUNT_FAC,
        "h_cost": output_dic['model_params']['policy_params']['h_cost'],
        "b_penalty": output_dic['model_params']['policy_params']['b_penalty'],
        "n_visit_lim": N_VISIT_LIM,
        "default_ss_policy": ss_policy_fastest_supp_backlog,
        "safe_factor": SAFE_FACTOR,
        "sub_eval_periods": SUB_EVAL_PERIODS,
        "sub_nested_mc_iter": SUB_NESTED_MC_ITER,
        "max_stock": BIG_S,
        "approx_eval": True,
        "pol_dic": output_dic["pol_dic"]
    }

    mc_avg_costs = mc_with_policy(sourcingEnv, 
        use_tqdm = True,
        policy_callback = eval_policy_from_policy_dic,
        **kwargs)

    eval_costs_scaled = np.mean(mc_avg_costs)
    output_dic['adp_cost'] = np.mean(eval_costs_scaled)

    with open(write_path, 'wb') as handle:
        pickle.dump(output_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("algorithm complete, wrote file to: " + write_path)

    print("Sending emailend email")
    highligh_dic_keys = ['ssn_cost', 'ss_cost', 'myopic_cost', 'di_cost', 'adp_cost']
    id_dic_keys = ['model_params']
    
    results_dic = {key: output_dic[key] for key in highligh_dic_keys}
    id_info_dic = {key: output_dic[key] for key in id_dic_keys}

    string_content = str(results_dic) + "\n\n" + json.dumps(id_info_dic, indent = 2)
    
    print(string_content)

    try:
        send_email(file_id = output_obj_path, mail_content = string_content)
    except:
        print("Email filed to send!")

    # execute LP solver


    