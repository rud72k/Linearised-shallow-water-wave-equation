# Linearised Shallow Water Wave Equation

This repo the code used in the arxiv paper [https://doi.org/10.48550/arXiv.2309.16116](https://doi.org/10.48550/arXiv.2309.16116) version 2 (non-dimensional). 

Solving linear shallow water wave equation (SWE) 
## Contents

Main files: 
| Files | Explanation |
| ----- | ----------- | 
|**The notebook**                   |              | 
| ```solve.ipynb ```                | Working files for 1D linearised SWE               |
| ```solve_both.ipynb```            | Working files for 1D linear and non-linear SWE    | 
| ```convergencetest.ipynb```       | Notebook for convergence test for 1D linearised SWE | 
| ----- | ----------- | 
|**Python function**                |              | 
| ```list_initial_condition.py```   | List of function of various initial condition     |
| ```list_swwe_function.py```       | List of function that been used to solve the SWE  | 
| ```list_ibvp_solution.py```       | List of other function such as analytical solution,| 
|                                   | manufactured solution, generated incoming waves   |  
| ```flowtype.py```                 | routine to check flowtype: critical, subcritical, supercritcal | 
|   | | 
| ----- | ----------- | 
|**Unit test** | | 
| ```test_flowtype.py```    | 
| ```test_linearised_SWWE_matrix_supportive.py``` | checking $Q,A,P_inv$ matrix | 
| ```test_list_initial_conditions.py```| test some of the initial conditions (incomplete) | 
| ```test_list_ibvp_solution.py``` | test some of the function | 
|  (incomplete)         |   | 
| ----- | ----------- | 

## Some details

#### Notebook 
This repo has jupyter notebooks as the main files. 

In the notebooks, we set up parameters before calling the main functions. 

A variable ```mms``` is set to ```false``` whenever manufactured solution are not to be used. 

The equation (model) is called from ```list_swwe_function.py``` and put in variables named ```RHS``` and ```SAT```, later on will be packed in a variable named ```constants```.

This variable ```constants``` packed up variables that suppposed not to change during the loop of solving the equation. In the linear case, this is convinient since variables like ```U``` and ```H``` are not changed. But later, this way of coding is not effiecient in the non-linear case since we need to set up ```U``` and ```H``` each time step.  

The initial condition taken from the function in ```list_initial_condition.py```, called just before running the loop. 

#### Function
