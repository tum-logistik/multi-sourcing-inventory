import numpy as np
from common.variables import *

# simulation functions

def cost_calc(state, h_cost = H_COST, b_penalty = B_PENALTY):
    cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return cost


