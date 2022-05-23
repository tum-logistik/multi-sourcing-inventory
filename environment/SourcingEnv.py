import numpy as np
from environment.Event import *
import copy

class SourcingEnv():

    # Switch to np.array for speed up
    def __init__(self, 
        order_quantity = 30,
        lambda_arrival = 0.1,
        procurement_cost_vec = np.array([2, 1.7]), 
        supplier_lead_times_vec = np.array([0.5, 0.75]), 
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1])):
        
        invert_np = lambda x: 1/x

        self.action_size = order_quantity
        self.lambda_arrival = lambda_arrival
        self.on_times = on_times
        self.off_times = off_times
        self.procurement_cost_vec = procurement_cost_vec
        self.supplier_lead_times_vec = supplier_lead_times_vec
        self.n_suppliers = len(procurement_cost_vec)
        self.current_state = self.reset()

        self.mu_lead_time = invert_np(self.supplier_lead_times_vec)
        self.mu_on_times = invert_np(self.on_times)
        self.mu_off_times = invert_np(self.off_times)
        self.event_space = [Event.DEMAND_ARRIVAL, Event.SUPPLY_ARRIVAL, Event.SUPPLIER_ON, Event.SUPPLIER_OFF, Event.NO_EVENT]

    # state is defined as inventories of each agent + 
    def reset(self):
        return np.array([0] + [0] * self.n_suppliers + [1] * self.n_suppliers)

    # order_quantity_vec is action (tau)
    def compute_event_rate(self, order_quantity_vec, 
        state_vec = None, 
        lambda_arrival = None, 
        mu_lead_time = None, 
        mu_on_times = None, 
        mu_off_times = None):
        
        if lambda_arrival == None:
            lambda_arrival = self.lambda_arrival
        if mu_lead_time == None:
            mu_lead_time = self.mu_lead_time
        if mu_on_times == None:
            mu_on_times = self.mu_on_times
        if mu_off_times == None:
            mu_off_times = self.mu_off_times
        if state_vec == None:
            state_vec = self.current_state
        
        outstd_orders = state_vec[1:1+self.n_suppliers]
        onoff_status = state_vec[1+self.n_suppliers:]

        lead_time_comps = np.multiply(order_quantity_vec + outstd_orders, mu_lead_time)
        lead_time_comp = np.sum(lead_time_comps)
        mu_off_comp = np.sum(np.multiply(1 - onoff_status, mu_off_times))
        mu_on_comp = np.sum(np.multiply(onoff_status, mu_on_times))

        expected_time = lambda_arrival + lead_time_comp + mu_off_comp + mu_on_comp

        return 1 / expected_time
    
    # index for each event
    # k is supplier index (pij)
    def compute_trans_prob(self, order_quantity_vec, event_type, k = None, state_vec = None, event_rate = None):
        
        if state_vec == None:
            state_vec = self.current_state
        if event_rate == None:
            event_rate = self.compute_event_rate(order_quantity_vec)
        
        if event_type == Event.DEMAND_ARRIVAL:
            return self.lambda_arrival * event_rate
        else:
            assert k is not None
            if event_type == Event.SUPPLY_ARRIVAL:
                outstd_orders = state_vec[1+k]
                return (outstd_orders + order_quantity_vec[k]) * self.mu_lead_time[k] * event_rate
            
            if event_type == Event.SUPPLIER_ON:
                onoff_status = state_vec[1+self.n_suppliers+k]
                if 1 - onoff_status < 0:
                    print("here")
                return (1 - onoff_status) * self.mu_off_times[k] * event_rate
            
            if event_type == Event.SUPPLIER_OFF:
                onoff_status = state_vec[1+self.n_suppliers+k]
                return onoff_status * self.mu_on_times[k] * event_rate
        
        return 0
    
    
    # action: order_quantity_vec
    def get_event_probs(self, order_quantity_vec):
        # use np.random.choice
        # vi = np.random.choice(agent_ids, 1, p=win_probs, replace=False)
        demand_arrival_prob = self.compute_trans_prob(order_quantity_vec, Event.DEMAND_ARRIVAL)
        
        supply_arrival_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLY_ARRIVAL, k = i) for i in range(self.n_suppliers)]
        supply_on_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLIER_ON, k = i) for i in range(self.n_suppliers)]
        supply_off_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLIER_OFF, k = i) for i in range(self.n_suppliers)]

        event_probs = np.array([demand_arrival_prob] + supply_arrival_probs + supply_on_probs + supply_off_probs)
        
        return event_probs
    
    # .step(action)
    def step(self, order_quantity_vec):
        event_probs = self.get_event_probs(order_quantity_vec)
        event_indexes = np.array(range(len(event_probs)))

        # for i in event_indexes:
        #     b = np.random.choice([0, 1], 1, p=[1-event_probs[i], event_probs[i]] )[0]
        #     if b == 1:
        #         break
        
        i = np.random.choice(event_indexes, 1, p = event_probs)[0]
        
        next_state = copy.deepcopy(self.current_state) 
        if i == 0:
            event = Event.DEMAND_ARRIVAL
            next_state[0] = next_state[0] - 1
            for k in range(self.n_suppliers):
                if self.current_state[1+self.n_suppliers+k] == 1:
                    next_state[1+k] += order_quantity_vec[i-1]
        elif 0 < i < 1 + self.n_suppliers:
            event = Event.SUPPLY_ARRIVAL
            next_state[0] = next_state[0] + 1
            next_state[i] = next_state[i] + order_quantity_vec[i-1] - 1 # match index with k
        elif 1 + self.n_suppliers - 1 < i < 1 + 2*self.n_suppliers:
            event = Event.SUPPLIER_ON
            next_state[i] = next_state[i] + 1 # coincidentally same index
        elif 1 + 2*self.n_suppliers - 1 < i:
            event = Event.SUPPLIER_OFF
            index = i - self.n_suppliers
            next_state[index] = next_state[index] - 1
        else:
            event = Event.NO_EVENT # No state transition

        self.current_state = next_state

        return next_state, event, i, event_probs 

    