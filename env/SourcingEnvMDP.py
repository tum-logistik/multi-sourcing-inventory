import numpy as np
from env.HelperClasses import *
from common.variables import *
import copy
from scipy.stats import poisson
from sim.sim_functions import *

class SourcingEnvMDP():

    # Switch to np.array for speed up
    def __init__(self, 
        order_quantity = ACTION_SIZE,
        lambda_arrival = LAMBDA,
        procurement_cost_vec = PROCUREMENT_COST_VEC, 
        supplier_lead_times_vec = SUPPLIER_LEAD_TIMES_VEC, 
        on_times = ON_TIMES, 
        off_times = OFF_TIMES,
        fixed_costs = FIXED_COST_VEC,
        max_demand = MAX_DEMAND,
        max_supply_cap = MAX_SUPPLY_CAP,
        backorder_max = BACKORDER_MAX,
        inven_limit = INVEN_LIMIT,
        tracking_flag = True):
        
        invert_np = lambda x: 1/x

        self.action_size = order_quantity

        self.lambda_arrival = lambda_arrival
        self.on_times = on_times
        self.off_times = off_times
        self.procurement_cost_vec = procurement_cost_vec
        self.supplier_lead_times_vec = supplier_lead_times_vec
        self.n_suppliers = len(procurement_cost_vec)
        self.fixed_costs = fixed_costs
        self.max_demand = max_demand
        self.demand_overage_prob = 1 - poisson.cdf(self.max_demand, self.lambda_arrival)
        self.max_supply_cap = max_supply_cap
        self.backorder_max = backorder_max
        self.inven_limit = inven_limit
        _ = self.reset()

        self.tracking_flag = tracking_flag # obselete for MDP version

        self.mu_lt_rate = invert_np(self.supplier_lead_times_vec)
        self.mu_on_times = invert_np(self.on_times)
        self.mu_off_times = invert_np(self.off_times)
        self.event_space = [Event.DEMAND_ARRIVAL, Event.SUPPLY_ARRIVAL, Event.SUPPLIER_ON, Event.SUPPLIER_OFF, Event.NO_EVENT]

        self.availibilities = np.array([0.5]*self.n_suppliers)
        for n in range(self.n_suppliers):
            self.availibilities[n] = self.mu_off_times[n] / (self.mu_off_times[n] + self.mu_on_times[n])
        
        # initialize marginal probability matrix
        self.all_trans_event_array = get_combo_states_grid(list(range(self.max_demand)), list(range(self.max_supply_cap)), 2)

        self.trans_prob_dic = generate_trans_dic(self.all_trans_event_array, self.lambda_arrival, self.mu_lt_rate, self.availibilities, 
            demand_overage_prob = self.demand_overage_prob, n_suppliers = self.n_suppliers)

        print("init ready")

    # state is defined as inventories of each agent + 
    def reset(self):
        initial_state = MState(n_suppliers = self.n_suppliers)
        self.current_state = initial_state
        return initial_state

    def step(self, order_quantity_vec, force_event_tuple = None):

        assert order_quantity_vec.all() >= 0, "Assertion Failed: Negative order quantity!"
        
        # Check force_event_tuple
        # if forced transition
        
        # if natural transition
        # do action
        # tuple format is (next_state_obj, ?)

        if force_event_tuple is not None:
            # event = force_event_tuple[0]
            next_state_obj = force_event_tuple[0]
            inventory_diff = next_state_obj.s - self.current_state.s
            
            all_trans_event_array_demand_neg = self.all_trans_event_array.copy()
            all_trans_event_array_demand_neg[:,0] = -1*all_trans_event_array_demand_neg[:,0]

            inds = np.argwhere(np.sum(all_trans_event_array_demand_neg[:,0:self.n_suppliers+1], axis=1) == inventory_diff)
            # arrival_events_next_state = self.all_trans_event_array[inds, :]

            # probability of each event occuring,
            # complex stuff...

            next_inven_level = next_state_obj.s
            # marg_event_prob = poisson.cdf(next_inven_level, self.lambda_arrival) - poisson.cdf(next_inven_level-1, self.lambda_arrival) - self.demand_overage_prob
        
        else:
            next_state = copy.deepcopy(self.current_state)
            demand_arrivals = np.random.poisson(self.lambda_arrival, 1)[0]
            
            self.current_state.n_backorders += order_quantity_vec

            order_arrivals = [0] * self.n_suppliers
            for n in range(self.n_suppliers):
                order_arrivals_no_clip = np.random.poisson(self.mu_lt_rate[n], 1)[0]
                order_arrivals[n] = np.clip(order_arrivals_no_clip, 0, self.current_state.n_backorders[n])
            
            current_on_off_status = self.current_state.flag_on_off
            for n in range(len(current_on_off_status)):
                if current_on_off_status[n] == 1:
                    p_switch = 1 - self.availibilities[n]
                    if np.random.binomial(1, p_switch, 1)[0] > 0:
                        current_on_off_status[n] = 0
                elif current_on_off_status[n] == 0:
                    p_switch = self.availibilities[n]
                    if np.random.binomial(1, p_switch, 1)[0] > 0:
                        current_on_off_status[n] = 1
                else:
                    pass
            
            next_state.s = self.current_state.s - demand_arrivals + np.sum(order_arrivals)
            next_state.n_backorders = self.current_state.n_backorders - order_arrivals
            next_state.flag_on_off = current_on_off_status

            # next_state = copy.deepcopy(self.current_state)
        self.current_state = copy.deepcopy(next_state)
        return self.current_state, demand_arrivals, order_arrivals, self.current_state.flag_on_off  #, event, i, event_probs, supplier_index