import gurobipy as gp
import numpy as np

m = gp.Model()

x = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = -1, ub = 1)
y = m.addVar(vtype = gp.GRB.CONTINUOUS, lb = -1, ub = 1)

m.addConstr(x == 5)

m.setObjective(5*x, gp.GRB.MINIMIZE)
m.optimize()
print(m)

# debug

m.computeIIS()
m.write("model1.ilp")



