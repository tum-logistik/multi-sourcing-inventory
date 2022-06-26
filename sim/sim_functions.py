import numpy as np
from common.variables import *

# simulation functions

def cost_calc(state, h_cost = H_COST, b_penalty = B_PENALTY):
    cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return cost

# TODO: Create an expected cost function


def get_combo(y, n):
    return np.array(np.meshgrid(*[range(0, y) for x in range(n)])).T.reshape(-1, n)