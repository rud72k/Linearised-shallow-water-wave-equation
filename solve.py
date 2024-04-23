#%% 

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import list_initial_conditions as ic
import list_ibvp_solution as ibvp
import list_swwe_function as swe
from time import time
from flowtype import flowtype_function

# def main():
#     """Main function to solve the linearised shallow water wave equations.
#     This main program solving Initial Boundary Value Problem (IBVP) for the linearised shallow water wave equations.
#     The IBVP is solved using the Runge Kutta 4th order method.
#     The results are not with the manufactured solution in this script. 

#     Parameters:
#     constants = x,H,U,g,c,alpha,flowtype,Q,A,P_inv, 
#                 RHS_function, SAT_function, 
#                 manufactured_solution, analytical_solution,
#     x: float
#     H: float
#     U: float
#     g: float
#     c: float
#     alpha: float
#     flowtype: str
#     Q: np.ndarray
#     A: np.ndarray
#     P_inv: np.ndarray
#     RHS_function: callable
#     SAT_function: callable
#     manufactured_solution: callable, False
#     analytical_solution: callable, None
#     """

start=time()
# set up nature of the simulation, with or without manufactured solution:
# 1. manufactured_solution = ibvp.linearSWWE_mms
# 2. manufactured_solution = False
mms = False

SAT_function = swe.linearised_SWWE_SAT_terms
RHS = swe.linearswwe_RHS

# set 'generated_wave = False' if manufactured_solution is not False
generated_wave = ibvp.zero_wave


# set up parameters
cfl = 0.25
H = 1.0
g = 9.81
c = np.sqrt(g*H)

# coefficient may vary 
U_coeff = 0.5
alpha_coeff = 0

U = U_coeff*c
alpha = alpha_coeff*(np.abs(U)+c)

flowtype = flowtype_function(U,c)

# set up the domain 
# x0, xN = 0, 4
x0, xN = 0,(np.abs(U)+np.sqrt(g*H))*5
N = 2**3
dx = (xN-x0)/(N-1)  
x = np.linspace(x0,xN,N)

# set up time parameters
dt = cfl*dx/(np.abs(U) + np.sqrt(g*H))      # (cfl * dx)/max speed
sim_time = 40*dt

# set up supportive matrix 
Q, A, P_inv = swe.linearised_SWWE_matrix_supportive(N, dx)

# list constants
constants = x,H,U,g,c,alpha,flowtype,Q,A,P_inv, \
        RHS, SAT_function, \
        False, generated_wave 

# x,H,U,g,c,alpha,flowtype,Q,A,P_inv,RHS_function, SAT,manufactured_solution, analytical_solution = constants

print("constants")
print(H,U,g,c,alpha,flowtype, dt)
# solving the IBVP 

# set up initial conditions
q = ic.initial_condition(x, H, U, 0)
q_numerical = copy(q)
q_analytical = copy(q)

q_numerical = swe.numerical_solution(q_numerical,sim_time,dt,constants)
# q_analytical = swe.analytical_solution(x,sim_time,dt,constants)

end = time()
print("Time taken: ", end-start)

plt.plot(x,q[0], label=0)
for i in range (int(sim_time//dt)):
    q0 = q_numerical[i+1]
    h, u = q0 
    plt.plot(x,h,label=i)
    plt.legend()
plt.show()
# %%
 