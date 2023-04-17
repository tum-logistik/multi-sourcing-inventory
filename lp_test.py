import gurobipy as gp

# Create a new model
model = gp.Model("simple_lp")

# Define the decision variables
x1 = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="x1")
x2 = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="x2")

# Set the objective function
model.setObjective(3*x1 + 2*x2, gp.GRB.MAXIMIZE)

# Add the constraints
model.addConstr(2*x1 + x2 <= 4, name="c1")
model.addConstr(x1 + 2*x2 <= 3, name="c2")

# Optimize the model
model.optimize()

# Print the optimal solution
print(f"Optimal objective value: {model.objVal:.2f}")
for var in model.getVars():
    print(f"{var.varName} = {var.x:.2f}")
