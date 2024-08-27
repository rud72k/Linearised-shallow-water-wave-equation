import numpy as np
import list_initial_conditions as lic
import list_ibvp_solution as lisol
import list_swwe_function as lsf

def nonlinearSWWE_mms(
        x_array: np.ndarray, 
        time: float, 
        H: float, 
        U: float, 
        g: float = 9.81,  
        phase_constant_1: float = 0, 
        phase_constant_2: float = 0,
        Omega_1: float = 3,
        Omega_2: float = 2) -> tuple[np.ndarray, np.ndarray]:

    """
    Calculate the manufactured solution for linear shallow water wave equation.

    Parameters:
    x_array (ndarray): The x-array.
    time (float): The time.
    H (float): The average height.
    U (float): The average speed.
    g (float): The acceleration due to gravity.
    length (float): The length.
    phase_constant_1 (float): The first phase constant.
    phase_constant_2 (float): The second phase constant.
    Omega_1 (float): The first frequency.
    Omega_2 (float): The second frequency.
    Returns:
    The height, speed, and two force components of the solution.
    """
    length = x_array[-1] - x_array[0]

    h =                    np.cos(2*np.pi*time)*np.sin(2*np.pi*x_array*Omega_1/length + phase_constant_1) 
    hx =  2*np.pi*Omega_1*(np.cos(2*np.pi*time)*np.cos(2*np.pi*x_array*Omega_1/length + phase_constant_1))/length 
    ht =         -2*np.pi*(np.sin(2*np.pi*time)*np.sin(2*np.pi*x_array*Omega_1/length + phase_constant_1)) 

    u =                    np.sin(2*np.pi*time )*np.cos(2*np.pi*x_array*Omega_2/length + phase_constant_2)
    ux = -2*np.pi*Omega_2*(np.sin(2*np.pi*time )*np.sin(2*np.pi*x_array*Omega_2/length + phase_constant_2))/length
    ut =          2*np.pi*(np.cos(2*np.pi*time )*np.cos(2*np.pi*x_array*Omega_2/length + phase_constant_2))
    
    f1 = ht + U*hx + H*ux 
    f2 = ut + g*hx + U*ux
    p = np.array([h,u])
    F = np.array([f1,f2])
    return p, F

def nonlinearSWWE_mms(
        x_array: np.ndarray, 
        time: float, 
        H: float, 
        U: float, 
        g: float = 9.81,  
        phase_constant_1: float = 0, 
        phase_constant_2: float = 0,
        Omega_1: float = 3,
        Omega_2: float = 2) -> tuple[np.ndarray, np.ndarray]:

    """
    Calculate the manufactured solution for linear shallow water wave equation.

    Parameters:
    x_array (ndarray): The x-array.
    time (float): The time.
    H (float): The average height.
    U (float): The average speed.
    g (float): The acceleration due to gravity.
    length (float): The length.
    phase_constant_1 (float): The first phase constant.
    phase_constant_2 (float): The second phase constant.
    Omega_1 (float): The first frequency.
    Omega_2 (float): The second frequency.
    Returns:
    The height, speed, and two force components of the solution.
    """
    length = x_array[-1] - x_array[0]

    h =                    np.cos(2*np.pi*time)*np.sin(2*np.pi*x_array*Omega_1/length + phase_constant_1) 
    hx =  2*np.pi*Omega_1*(np.cos(2*np.pi*time)*np.cos(2*np.pi*x_array*Omega_1/length + phase_constant_1))/length 
    ht =         -2*np.pi*(np.sin(2*np.pi*time)*np.sin(2*np.pi*x_array*Omega_1/length + phase_constant_1)) 

    u =                    np.sin(2*np.pi*time )*np.cos(2*np.pi*x_array*Omega_2/length + phase_constant_2)
    ux = -2*np.pi*Omega_2*(np.sin(2*np.pi*time )*np.sin(2*np.pi*x_array*Omega_2/length + phase_constant_2))/length
    ut =          2*np.pi*(np.cos(2*np.pi*time )*np.cos(2*np.pi*x_array*Omega_2/length + phase_constant_2))
    
    uh = u*h 
    u2 = u*u

    uh_x = u*hx + ux*h
    u2_x = u*ux + ux*u

    h2_x = 2*h*hx
    u2h_x = u2*hx + h*u2_x
    
    f1 = ht + uh_x 
    f2 = ut + u2h_x + 0.5*g*h2_x
    
    p = np.array([h,u])
    F = np.array([f1,f2])
    return p, F

def advecting_wave_1(x_array:np.ndarray, time_local:float, H:float, U:float, g:float, some_function:callable)->np.ndarray:
    """
    some_function: callable: is function generated wave on the boundary.
    refering to g1 and g2 in the paper

    Returns:
    np.ndarray: The analytical solution for the advection wave at time=time_local.
    """
    n = len(x_array)
    adv_wave = np.zeros((2, n))
    for i in range(n):
        y_ = some_function(time_local-x_array[i]/(U + np.sqrt(g*H)))
        adv_wave[0,i] =                  y_
        adv_wave[1,i] = (1/np.sqrt(H/g))*y_
    
    return adv_wave


# boundary generated wave 

def gaussian_wave_1(t):
    return np.sin(np.pi*t)**4 if 0.0 <= t <= 1.0 else 0.0

def step_function_wave(t):
    return 1.0 if 0.0 <= t <= 1.0 else 0.0

def zero_wave(t):
    return 0

