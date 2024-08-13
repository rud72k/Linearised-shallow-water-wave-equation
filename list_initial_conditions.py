# This file contains the initial conditions and boundary data for the simulation.
import numpy as np


def init_height_gaussian(x_array: np.ndarray, HEIGHT: float) -> float:
    """
    Returns:
    float: The initial height of the Gaussian distribution at the given x-coordinate + constant HEIGHT.
    """
    CENTER = 0.5*(x_array[0] + x_array[-1])
    SCALE = 1.0
    return np.exp(-(x_array - CENTER)**2 / SCALE)*HEIGHT*0.01 + HEIGHT

def init_height_gaussian_perturbation(x_array: np.ndarray, HEIGHT: float) -> float:
    """
    Returns:
    float: The initial height of the Gaussian distribution at the given x-coordinate.
    This is only the perturbation, the constant height should be added in the main function.
    """
    CENTER = 0.5*(x_array[0] + x_array[-1])
    SCALE = 1.0
    return np.exp(-(x_array - CENTER)**2 / SCALE) * HEIGHT *0.01

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

def init_speed_zero(x_array: np.ndarray, speed)-> float:
    """
    Returns:
    float: The initialized speed, which is always zero.
    """
    return np.zeros_like(x_array)

def init_speed_constant(x_array: np.ndarray, constant)-> float:
    return constant + np.zeros_like(x_array)

def dam_break_torro(x_array: np.ndarray, height: float) -> float:
    """
    Returns:
    float: The dam break height.
    """
    return np.where(x_array <= 1000, height, 0)

def dam_break(x_array: np.ndarray, height: float) -> float:
    """
    Returns:
    float: The dam break height.
    """
    return np.where(x_array < (x_array[0]+x_array[-1])/2, height, 0.001*height)

def initial_condition(x_coord: np.ndarray, 
                      height: float,  
                      speed: float,
                      initial_height_type: int = 0, 
                      initial_speed_type: int = 0) -> np.ndarray:
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
                                  4 = dam break with almost dry bed
                                  5 = flat profile with step bottom
                                  6 = custom init_height_with_step_function_center 
                                  8 = flat profile with height = H 

    Returns:
    tuple[float, float]: The initial height and speed.
    """
    initial_height_functions = {
        0: init_height_zero, 
        1: init_height_gaussian,
        2: init_height_gaussian_perturbation,
        3: dam_break_torro,
        4: dam_break,
        5: init_height_with_step_bottom,
        6: init_height_with_step_function_center,
        8: init_height_constant
        }
    initial_speed_functions = {
        0: init_speed_zero,
        1: init_speed_constant
        }
    h = initial_height_functions[initial_height_type](x_coord, height)
    u = initial_speed_functions[initial_speed_type](x_coord, speed) 
    p = np.array([h,u])
    return p


# Doesn't have unit test

def bottom_bathymetry_flatbed(x_array: np.ndarray) -> np.ndarray:
    """
    Returns:
    float: The bottom bathymetry, which is always zero.
    """
    return np.zeros_like(x_array)

def step_function_bottom(x_array: np.ndarray, step_up_x_ratio) -> np.ndarray:
    """
    Returns:
    float: Step up bathymetry, which is zero up to some point, then stepped up.
    step_up_x_ratio: float, the x-coordinate ratio where the step up occurs.
    """
    step_up_x = step_up_x_ratio * x_array[-1]
    return np.where(x_array < step_up_x, 0, 1)

def slope_bottom(x_array: np.ndarray, slope, raise_up = 0) -> np.ndarray:
    """
    Returns:
    float: The slope bottom bathymetry, gradual changes.
    """
    return x_array * slope + raise_up

def bottom_bathymetry(x_array: np.ndarray,intial_type: int = 1, step_up_x_ratio = 0.5) -> np.ndarray:
    """
    Set the bottom bathymetry for a simulation.

    Parameters:
    x_array (nd.array): The x-coordinate.
    initial_type (int): The type of initial condition for perturbation height if no MMS. 
                                  0 = flatbed, 1 = Gaussian + constant 
                                  2 = Gaussian 
                                  8 = step function.

    Returns:
    tuple[float, float]: The initial height and speed.
    """
    bottom_bathymetry_functions = {
        # 0: bottom_bathymetry_flatbed,
        1: step_function_bottom
        # 2: slope_bottom
        }
    b = bottom_bathymetry_functions[intial_type](x_array, step_up_x_ratio)
    return b

def init_height_with_step_bottom(x_array: np.ndarray, height: float) -> float:
    """
    Returns:
    float: The initialized height, which is always the same constant.
    """
    return height - bottom_bathymetry(x_array, 1, 0.5) 


def init_height_with_step_function_center(x_array: np.ndarray, height: float) -> float:
    """
    Returns:
    float: The initialized height, which is always the same constant.
    this initial trying match with torro matlab code
    """
    CENTER      = 0.5*(x_array[0] + x_array[-1])
    RANGE_SCALE = 0.3*(x_array[-1] - x_array[0])
    y1 = np.where((CENTER - RANGE_SCALE/2 < x_array) & (x_array < CENTER+RANGE_SCALE/2), height, 0.01)
    return y1