# This file contains the initial conditions and boundary data for the simulation.
import numpy as np


def init_height_gaussian_perturbation(x_array: np.ndarray, HEIGHT: float) -> float:
    """
    Returns:
    float: The initial height of the Gaussian distribution at the given x-coordinate.
    """
    CENTER = 0.5*(x_array[0] + x_array[-1])
    SCALE = 1.0
    return np.exp(-(x_array - CENTER)**2 / SCALE) * HEIGHT *0.1

def init_height_gaussian(x_array: np.ndarray, HEIGHT: float) -> float:
    """
    Returns:
    float: The initial height of the Gaussian distribution at the given x-coordinate + constant HEIGHT.
    """
    CENTER = 0.5*(x_array[0] + x_array[-1])
    SCALE = 1.0
    return np.exp(-(x_array - CENTER)**2 / SCALE)*HEIGHT*0.1 + HEIGHT

def init_height_zero(x_array: np.ndarray, height: float = 0) -> float:
    """
    Returns:
    float: The initialized height, which is always zero.
    """
    return np.zeros_like(x_array)

def init_height_constant(x_array: np.ndarray, height: float) -> float:
    """
    Returns:
    float: The initialized height, which is always the same constant.
    """
    return height + np.zeros_like(x_array)


def init_speed_zero(x_array: np.ndarray)-> float:
    """
    Returns:
    float: The initialized speed, which is always zero.
    """
    return np.zeros_like(x_array)

def initial_condition(x_coord: np.ndarray, 
                      height: float,  
                      initial_type: int = 0) -> np.ndarray:
    """
    Set the initial conditions for a simulation.

    Parameters:
    x_coord (nd.array): The x-coordinate.
    use_mms (bool): Whether to use the method of manufactured solutions.
    height (float): The initial height.
    speed (float): The initial speed.
    initial_type (int): The type of initial condition for perturbation height if no MMS. 
                                  0 = drybed, 1 = Gaussian + constant 
                                  2 = Gaussian 
                                  8 = step function.

    Returns:
    tuple[float, float]: The initial height and speed.
    """
    initial_height_functions = {
        0: init_height_zero, 
        1: init_height_gaussian,
        2: init_height_gaussian_perturbation,
        8: init_height_constant
        }
    initial_speed_functions = {
        0: init_speed_zero
        }
    h = initial_height_functions[initial_type](x_coord, height)
    u = initial_speed_functions[0](x_coord) 
    p = np.array([h,u])
    return p




