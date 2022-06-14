import numpy as np
from env.HelperClasses import *
from common.variables import *
import copy

class SourcingEnv():

    # Switch to np.array for speed up
    def __init__(self, 
        order_quantity = ACTION_SIZE,
        lambda_arrival = LAMBDA,
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
        _ = self.reset()

        self.mu_lt_rate = invert_np(self.supplier_lead_times_vec)
        self.mu_on_times = invert_np(self.on_times)
        self.mu_off_times = invert_np(self.off_times)
        self.event_space = [Event.DEMAND_ARRIVAL, Event.SUPPLY_ARRIVAL, Event.SUPPLIER_ON, Event.SUPPLIER_OFF, Event.NO_EVENT]

        assert len(self.on_times) == self.n_suppliers, "Assertion Failed: Mismatch length - on_times"
        assert len(self.off_times) == self.n_suppliers, "Assertion Failed: Mismatch length - off_times"
        assert len(self.procurement_cost_vec) == self.n_suppliers, "Assertion Failed: Mismatch length - procurement_cost_vec"
        assert len(self.supplier_lead_times_vec) == self.n_suppliers, "Assertion Failed: Mismatch length - supplier_lead_times_vec"

    # state is defined as inventories of each agent + 
    def reset(self):
        initial_state = MState(n_suppliers = self.n_suppliers)
        self.current_state = initial_state
        return initial_state

    # order_quantity_vec is action (tau)
    def compute_event_arrival_time(self, order_quantity_vec, 
        state_obj = None, 
        lambda_arrival = None, 
        mu_lt_rate = None, 
        mu_on_times = None, 
        mu_off_times = None):
        
        if lambda_arrival == None:
            lambda_arrival = self.lambda_arrival
        if mu_lt_rate == None:
            mu_lt_rate = self.mu_lt_rate
        if mu_on_times == None:
            mu_on_times = self.mu_on_times
        if mu_off_times == None:
            mu_off_times = self.mu_off_times
        if state_obj == None:
            state_obj = self.current_state
        
        outstd_orders = state_obj.n_backorders        
        onoff_status = state_obj.flag_on_off

        lead_time_comps = np.multiply(order_quantity_vec + outstd_orders, mu_lt_rate)
        lead_time_comp = np.sum(lead_time_comps)

        mu_off_comp = np.sum(np.multiply(1 - onoff_status, mu_off_times))
        mu_on_comp = np.sum(np.multiply(onoff_status, mu_on_times))

        prob_demand_arrival = lambda_arrival + lead_time_comp + mu_off_comp + mu_on_comp

        return 1 / prob_demand_arrival
    
    # index for each event, k is supplier index (pij)
    def compute_trans_prob(self, 
        order_quantity_vec, 
        event_type, 
        k = None, 
        state_obj = None, 
        tau_event = None):
        
        if state_obj == None:
            state_obj = self.current_state
        if tau_event == None:
            tau_event = self.compute_event_arrival_time(order_quantity_vec)
        
        if event_type == Event.DEMAND_ARRIVAL:
            return self.lambda_arrival * tau_event
        else:
            assert k is not None
            if event_type == Event.SUPPLY_ARRIVAL:
                outstd_orders = state_obj.n_backorders
                return (outstd_orders[k] + order_quantity_vec[k]) * self.mu_lt_rate[k] * tau_event
            
            onoff_status = state_obj.flag_on_off[k]
            assert onoff_status > -1, "Assertion Failed: On / off flag outside bound (0, 1)"
            if event_type == Event.SUPPLIER_ON:
                return (1 - onoff_status) * self.mu_off_times[k] * tau_event
            
            if event_type == Event.SUPPLIER_OFF:
                return onoff_status * self.mu_on_times[k] * tau_event
        
        return 0
    
    # action: order_quantity_vec
    def get_event_probs(self, order_quantity_vec):

        demand_arrival_prob = self.compute_trans_prob(order_quantity_vec, Event.DEMAND_ARRIVAL)
        supply_arrival_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLY_ARRIVAL, k = i) for i in range(self.n_suppliers)]
        supply_on_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLIER_ON, k = i) for i in range(self.n_suppliers)]
        supply_off_probs = [self.compute_trans_prob(order_quantity_vec, Event.SUPPLIER_OFF, k = i) for i in range(self.n_suppliers)]

        event_probs = np.array([demand_arrival_prob] + supply_arrival_probs + supply_on_probs + supply_off_probs)
        sum_check = np.sum(event_probs)

        assert np.abs(1 - sum_check) < PROB_EPSILON, "Assertion Failed: Probability of events do not sum to 1!"

        return event_probs
    
    def get_event_index_from_event(self, event, supplier_index):
        if event == Event.DEMAND_ARRIVAL:
            i = 0
        elif event == Event.SUPPLY_ARRIVAL: # tuple includes (state, supplier)
            i = 1 + supplier_index
        elif event == Event.SUPPLIER_ON: 
            i = 1 + self.n_suppliers + supplier_index
        elif event == Event.SUPPLIER_OFF:
            i = 1 + 2*self.n_suppliers + supplier_index
        return i
    
    def get_event_tuple_from_index(self, i):
        if i == 0:
            event = Event.DEMAND_ARRIVAL
            supplier_index = None
        elif 0 < i <= 1 + self.n_suppliers:
            event = Event.SUPPLY_ARRIVAL # tuple includes (state, supplier)
            supplier_index = i - 1
        elif 1 + self.n_suppliers < i <= 1 + 2*self.n_suppliers:
            event = Event.SUPPLIER_ON
            supplier_index = i - 1 - self.n_suppliers
        elif 1 + 2*self.n_suppliers < i <= 1 + 3*self.n_suppliers:
            event = Event.SUPPLIER_OFF
            supplier_index = i - 1 - 2*self.n_suppliers
        return event, supplier_index
    
    # .step(action), defaults to backorder model
    def step(self, order_quantity_vec, force_event_tuple = None):

        assert order_quantity_vec.all() >= 0, "Assertion Failed: Negative order quantity!"
        
        if not force_event_tuple: 
            event_probs = self.get_event_probs(order_quantity_vec)
            event_indexes = np.array(range(len(event_probs)))            
            i = np.random.choice(event_indexes, 1, p = event_probs)[0]
        else:
            event_probs = np.zeros(1 + 3*self.n_suppliers)
            i = self.get_event_index_from_event(force_event_tuple[0], force_event_tuple[1]) # (event, supplier index)
            event_probs[i] = 1
        
        next_state = copy.deepcopy(self.current_state)
        
        if i == 0:
            event = Event.DEMAND_ARRIVAL
            next_state.s = next_state.s - 1
            supplier_index = None
        elif 0 < i < 1 + self.n_suppliers:
            event = Event.SUPPLY_ARRIVAL
            supplier_index = i-1
            next_state.s = next_state.s + next_state.n_backorders[supplier_index]
            next_state.n_backorders[supplier_index] = 0
        elif 1 + self.n_suppliers - 1 < i < 1 + 2*self.n_suppliers:
            event = Event.SUPPLIER_ON
            supplier_index = i - 1 - self.n_suppliers
            next_state.flag_on_off[supplier_index] = np.clip(next_state.flag_on_off[supplier_index] + 1, 0 ,1)
            assert -1 < next_state.flag_on_off[supplier_index] < 2, "Assertion Failed: Supplier ON Index over 1 or under 0"
        elif 1 + 2*self.n_suppliers - 1 < i:
            event = Event.SUPPLIER_OFF
            supplier_index = i - 1 - 2*self.n_suppliers
            next_state.flag_on_off[supplier_index] = np.clip(next_state.flag_on_off[supplier_index] - 1, 0 ,1)
            assert -1 < next_state.flag_on_off[supplier_index] < 2, "Assertion Failed: Supplier OFF Index over 1 or under 0"
        else:
            event = Event.NO_EVENT # No state transition
            supplier_index = None
        
        # Assume when supplier is unavailable, no addition to backlog.
        for k in range(self.n_suppliers):
            if self.current_state.flag_on_off[k] == 1:
                next_state.n_backorders[k] += order_quantity_vec[k]

        if supplier_index != None:
            assert supplier_index > -1, "Assertion Failed: supplier_index < 0"
        else:
            assert event in [Event.DEMAND_ARRIVAL, Event.NO_EVENT], "AssertAssertion Failed: Unknown event."

        self.current_state = next_state
        
        return next_state, event, i, event_probs, supplier_index

    