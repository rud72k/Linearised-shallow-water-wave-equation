import numpy as np
from scipy.sparse import diags
from scipy.sparse import identity
import list_initial_conditions as lic
import list_ibvp_solution as lisol
import list_swwe_function as lsf
from copy import copy


# constants = x,H,U,g,c,alpha,flowtype,Q,A,P_inv, 
#             RHS_function, SAT_function, 
#             manufactured_solution, analytical_solution 
# x: float
# H: float
# U: float
# g: float
# c: float
# alpha: float
# flowtype: str
# Q: np.ndarray
# A: np.ndarray
# P_inv: np.ndarray
# RHS_function: callable
# SAT_function: callable
# manufactured_solution: callable, None
# analytical_solution: callable, None


def linearswwe_RHS(quantity, time_local, constants):
    """The right hand side of the linearised shallow water wave equations."""
    
    # Unpack the constants
    x,H,U,g,_,alpha,_,Q,A,P_inv,_,_, SAT,manufactured_solution, _ = constants
    h, u = quantity
    
    # Set the manufactured solution 
    
    if manufactured_solution is False:
        mms_list = [[0,0],[0,0]]
        f1, f2 = 0, 0
    else:
        mms_list = manufactured_solution(x, time_local, H, U)
        _, F = mms_list
        f1, f2 = F

    # calling SAT terms 
    SAT__ = SAT(quantity,time_local,constants,mms_list)

    SAT_terms_1 = SAT__[0]
    SAT_terms_2 = SAT__[1] 

    # Right hand side of the equation
    # based on equation (27)

    RHS_h = -(U*(Q)@h + H*(Q)@u - (alpha/2)*A@h ) + SAT_terms_1
    RHS_u = -(g*(Q)@h + U*(Q)@u - (alpha/2)*A@u ) + SAT_terms_2

    # moving out to the right hand side of the equation
    # and multiply with P_inverse to let the PDE ready to solve with
    # Runge Kutta (rk4 function in this code)
    RHS_h = P_inv@(RHS_h) + f1
    RHS_u = P_inv@(RHS_u) + f2
    
    RHS = np.array([RHS_h, RHS_u])
    return RHS

def linearised_SWWE_SAT_terms(quantity,time_local,constants, mms_list):
    '''
    This function is to calculate the SAT terms for the linearised SWWE
    The boundary condition for this SAT function is transmissive boundary condition.
    The input of the quantity h and u
    '''

    # unpacking constants
    x,H,U,_,c,_,flowtype,_,_,_,_,_,_,_, generated_wave_data = constants

    #unpacking q
    h, u = quantity
    n = len(x)

    # unpacking analytical solution from mms_list
    # in this point, we not calculate anything, just unpacking the values
    q_mms, _ = mms_list
    h_analytic_mms, u_analytic_mms = q_mms

    # creating unit vectors
    e0 = np.hstack([1, np.zeros(n-1)])
    eN = np.hstack([np.zeros(n-1), 1])
    

    # Some variables based on equation (13)
    lambda1 = U + c
    lambda2 = U - c

    S = (1/np.sqrt(2))*np.array([[1,1],
                                 [1,-1]]) 

    # based on equation (14)
    # turn h and u into dimensionless h/H and u/c

    # ver1. the original without add the average height
    # w_1 = ((h-h_analytic_mms)/H + (u-u_analytic_mms)/c)/np.sqrt(2)
    # w_2 = ((h-h_analytic_mms)/H - (u-u_analytic_mms)/c)/np.sqrt(2)

    # ver2. add the H to the perturbation height
    w_1 = ((h-h_analytic_mms-H)/H + (u-u_analytic_mms)/c)/np.sqrt(2)
    w_2 = ((h-h_analytic_mms-H)/H - (u-u_analytic_mms)/c)/np.sqrt(2)

    # based on proof of theorem 9,10,11
    W_inv = np.diag([H,c])

    W_inv_S = W_inv@S 
 
    g_1 = generated_wave_data(time_local) if generated_wave_data is not False else 0
    g_2 = 0                                 # can be change into some function if needed

    # to-do: calculate g_1 again if there is a generated wave
    g_1 = g_1 - H 

    if flowtype == 'subcritical':

        # based on Lemma 5 for subcritical flow
        gamma0, gammaN = 0, 0

        # based on theorem 9 for subcritical flow
        tau_01 = lambda1
        tau_02 = gamma0*lambda1
        tau_N1 = -gammaN*lambda2 
        tau_N2 = -lambda2 

        # based on equation 28 (without (-1/2)W_inv_S kron I)
        BC_0 = (w_1[0] - gamma0*w_2[0])  - g_1*np.sqrt(2)/H
        BC_N = (w_2[-1]- gammaN*w_1[-1]) - g_2*np.sqrt(2)/H

        SAT__ =  np.array([(tau_01*e0*BC_0 + tau_N1*eN*BC_N),\
                           (tau_02*e0*BC_0 + tau_N2*eN*BC_N)])

    elif flowtype == 'critical':

        # based on theorem 10 for critical flow
        tau_01 = lambda1
        tau_02 = 0
        tau_N1 = 0 
        tau_N2 = -lambda2 
        
        # based on equation 29a and 29b (without (-1/2)W_inv_S kron I)
        BC__1 = w_1 - g_1*np.sqrt(2)/H
        BC__2 = w_2 - g_2*np.sqrt(2)/H

        if U > 0:
            SAT__ =  np.array([tau_01*e0*(BC__1),\
                               tau_02*e0*(BC__2)])
        else:
            SAT__ =  np.array([tau_01*eN*(BC__1),
                               tau_02*eN*(BC__2)])

    elif flowtype == 'supercritical':

        # based on theorem 11 for supercritical flow
        tau_01 = lambda1
        tau_02 = lambda2
        tau_N1 = -lambda1 
        tau_N2 = -lambda2 

        # based on equation 29a and 29b (without (-1/2)W_inv_S kron I)
        BC__1 = w_1 - g_1*np.sqrt(2)/H
        BC__2 = w_2 - g_2*np.sqrt(2)/H

 
        if U > 0:
            SAT__ =  np.array([tau_01*e0*(BC__1),\
                               tau_02*e0*(BC__2)])
        else:
            SAT__ =  np.array([tau_01*eN*(BC__1),
                               tau_02*eN*(BC__2)])
            
    # multiply SAT__ with (-1/2)W_inv_S kron I to get the proper terms for SAT

    SAT = -(1/2)*np.array([W_inv_S[0,0]*SAT__[ 0] +  W_inv_S[0,1]*SAT__[ 1],
                           W_inv_S[1,0]*SAT__[ 0] +  W_inv_S[1,1]*SAT__[ 1]])

    return SAT

def SWE_SAT_terms_no_SAT(quantity,time_local,constants, mms_list):
    x,_,_,_,_,_,_,_,_,_,_,_,_,_, _ = constants
    n = len(x)
    return np.zeros((2,n))

def linearised_SWWE_matrix_supportive(n: int,delta_x: float):
    """
    Create the matrices A, Q, and P_inv for the PDE.

    Parameters:
    n (integer): The number of grid cells.
    delta_x (float): The difference in the x-coordinate. Can be expanded into a vector if necessary.

    Returns:
    A, Q, P_inv: sparse matrix  = refer to the equation (24)
    """

    A_diag = np.hstack([-1, -2*np.ones(n-2), -1])
    A = diags([np.ones(n-1), A_diag, np.ones(n-1)], [-1, 0, 1])

    Q_diag = np.hstack([-1/2, -2*np.zeros(n-2), 1/2])
    Q = diags([-np.ones(n-1)/2, Q_diag, np.ones(n-1)/2], [-1, 0, 1])

    P_inv = diags(np.hstack([2, np.ones(n-2), 2])/delta_x)

    I_N = identity(n)
    pde_matrices = Q, A, P_inv, I_N
    return pde_matrices
    # ---- end setup ---- # 


def numerical_solution_linear(initial_quantity:np.ndarray, simulation_time:float, time_step:float, constants: tuple):
    """Solve the PDE using the Runge Kutta 4 method.
    Input and output would be in the form of the perturbation of the height and velocity.
    """
    _,_,_,_,_,_,_,_,_,_,_,RHS_function,_,_,_ = constants
    
    # initiate solution at time = 0
    local_time = 0
    quantity__ = copy(initial_quantity)

    # solution = np.array([quantity])
    solution = [quantity__]
    while local_time < simulation_time: 
        # calculate the new quantity with Runge-Kutta4 method
        k1 = RHS_function(quantity__,                        local_time,                  constants)
        k2 = RHS_function(quantity__ + 0.5 * k1 * time_step, local_time + 0.5* time_step, constants)
        k3 = RHS_function(quantity__ + 0.5 * k2 * time_step, local_time + 0.5* time_step, constants)
        k4 = RHS_function(quantity__ +       k3 * time_step, local_time +      time_step, constants)

        quantity__ += (k1 + 2*k2 + 2*k3 + k4)*time_step/6
        # solution.append(quantity)
        solution = np.vstack((solution,[quantity__]))
        local_time += time_step
    return solution

def analytical_solution_mms(x_array:np.ndarray, simulation_time:float, time_step:float, constants):
    """Calculate the analytical solution to the PDE."""
    x_array,H,U,g,_,_,_,_,_,_,_,_, _,manufactured_solution, _ = constants
    # calculate the analytical solution
    local_time = 0
    q__, _ = manufactured_solution(x_array,0, H,U)
    solution = [q__]
    while local_time < simulation_time:
        local_time += time_step
        solution_hu, _ = manufactured_solution(x_array,local_time, H,U,g)
        solution.append(solution_hu)

    # adding the difference of time_local with the analytical
    
    return solution

def analytical_solution_routine(x_array:np.ndarray, simulation_time, time_step,constants):
    """Calculate the analytical solution to the PDE."""
    x_array,H,U,g,_,_,_,_,_,_,_, _,_,_, analytical_solution = constants
    # calculate the analytical solution
    local_time = 0
    solution = []
    while time_step < simulation_time:
        solution1 = analytical_solution(x_array,time_step, H,U,g)
        solution.append(solution1)
        local_time += time_step

    return solution


# Non-linear SWWE 

def nonlinear_swwe_RHS(quantity, time_local, constants):
    """The right hand side of the nonlinear shallow water wave equations."""
    # Unpack the constants
    
    x,H,U,g,c,alpha,flowtype,Q,A,P_inv,I_N, RHS_function, SAT, manufactured_solution, generated_wave  = constants
    
    h, uh = quantity
    u = uh/h
    # Set the manufactured solution 
    if manufactured_solution is False:
        mms_list = [[0,0],[0,0]]
        f1, f2 = 0, 0
    else:
        mms_list = manufactured_solution(x, time_local, H, U)
        _, F = mms_list
        f1, f2 = F
        
    # perturbation of the height and velocity to be used in the SAT terms
    # quantity_nonconservative = np.array([h-H,u-U]) # perturbation h, u 
    quantity_nonconservative = np.array([h,u]) # perturbation h, u 
    
    # calling SAT terms 
    
    constants_for_SAT = x,H,U,g,np.sqrt(g*H),  \
        alpha,flowtype,Q,A,P_inv,I_N, RHS_function, SAT, manufactured_solution, generated_wave
    

    SAT__linear = SAT(quantity_nonconservative,time_local,constants_for_SAT,mms_list)


    SAT_terms_1 = SAT__linear[0]    # h
    SAT_terms_2 = SAT__linear[1]    # u

    # Right hand side of the equation
    # based on equation (27)

    # ---------------------- #
    # # ver1.2 use fixed A matrix 
    # ---------------------- #

    # ---------------------- #
    # ver1.3 use variable A matrix alpha = max(|u| + sqrt(g*h))
    
    alphas = np.maximum(np.abs(u - np.sqrt(g*h)),np.abs(u + np.sqrt(g*h)))

    A_diag_up = alphas[1:]
    A_diag_up[0] = alphas[0]

    A_diag_down = alphas[:-1]
    A_diag_down[-1] = alphas[-1]

    A_diag_0  = -2*alphas
    A_diag_0[0] = -alphas[0]
    A_diag_0[-1] = -alphas[-1]    

    A = diags([A_diag_up, A_diag_0, A_diag_down], [1, 0, -1])
    # ---------------------- #

    RHS_h  = - I_N*(Q)@(uh)                 + (1/2)*A@h    
    RHS_uh = - I_N*(Q)@(uh**2/h + g*h**2/2) + (1/2)*A@uh   

    RHS_h += SAT_terms_1
    # RHS_uh += SAT_terms_1 * u + SAT_terms_2 * h 
    RHS_uh += SAT_terms_1 * U + SAT_terms_2 * H 

    # print(f"h[0]: {h[0]}")
    # print(f"h[-1]: {h[-1]}")
    # print(f"SAT_terms_1[0]: {SAT_terms_1[0]}")
    # print(f"SAT_terms_1[-1]: {SAT_terms_1[-1]}")

    # print(f"u[0] = {u[0]}")
    # print(f"u[-1] = {u[-1]}")
    # print(f"SAT_terms_2[0]: {SAT_terms_2[0]}")
    # print(f"SAT_terms_2[-1]: {SAT_terms_2[-1]}")



    # moving out to the right hand side of the equation
    # and multiply with P_inverse to let the PDE ready to solve with
    # Runge Kutta (rk4 function in this code)
    RHS_h = P_inv@(RHS_h) + f1
    RHS_uh = P_inv@(RHS_uh) + f2

    RHS = np.array([RHS_h, RHS_uh])
    return RHS

def numerical_solution_nonlinear(initial_quantity:np.ndarray, simulation_time:float, time_step:float, constants: tuple):
    """Solve the PDE using the Runge Kutta 4 method."""
    # _,_,_,_,_,_,_,_,_,_,_,RHS_function,_,_,_ = constants
    x,H,U,g,c,alpha,flowtype,Q,A,P_inv,I_N, RHS_function, SAT_function, mms_flag, generated_wave  = constants
    
    # initiate solution at time = 0
    local_time = 0

    # solution = np.array([quantity])
    quantity__ = copy(initial_quantity)
    solution = [quantity__]
    while local_time < simulation_time: 
        # print(f"local_time: {local_time}")
        # calculate the new quantity with Runge-Kutta4 method
        k1 = RHS_function(quantity__,                        local_time,                  constants)
        k2 = RHS_function(quantity__ + 0.5 * k1 * time_step, local_time + 0.5* time_step, constants)
        k3 = RHS_function(quantity__ + 0.5 * k2 * time_step, local_time + 0.5* time_step, constants)
        k4 = RHS_function(quantity__ +       k3 * time_step, local_time +      time_step, constants)

        quantity__ += (k1 + 2*k2 + 2*k3 + k4)*time_step/6

        # solution.append(quantity)
        solution = np.vstack((solution,[quantity__]))

        h__, _= quantity__
        #update constants
        # constants = x,H,U,g,c,alpha,flowtype,Q,A,P_inv,I_N, \
        #     RHS_function, SAT_function, \
        #     mms_flag, generated_wave 

        local_time += time_step

    return solution

