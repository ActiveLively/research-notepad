import numpy as np
from scipy.optimize import minimize, differential_evolution

E = 210e6 # elasticity, GPa
sig_b = 165e3  #allowable_bending_stress, MPa
tau_s = 50e3  #allowable_shear_stress, MPa
rho = 7800 # mass_density, kg/m^3
w = 2.0 # wind_load, kN/m
H = 10 # height, m
P = 4.0 # concentrated_load_at_top, kN

def objective(x):
    d_out = x[0] / 100.0 # outer_diameter, m
    d_in = x[1] / 100.0 #inner_diameter, m
    
    A = (np.pi/4) * (d_out **2 - d_in**2) # cross_sectional_area, m^2
    mass = A * H * rho 
    return mass

def constraints(x):
    d_out = x[0] / 100.0
    d_in = x[1] / 100.0
    
    # Cross-sectional properties
    I = (np.pi/64) * (d_out**4 - d_in**4) # moment_of_inertia, m^4
    
    # Loads and stresses
    M = P * H + 0.5 * w * H**2 # moment at base, kNm
    sigma = (M / (2 * I)) * d_out # bending stress, kPa
    S = (P+w*H) # shear at the base, kN             
    tau = (S/(12*I))*(d_out**2 + d_out*d_in + d_in**2) # shear stress, kPa
    
    delta = (P*H**3)/(3*E*I) + (w*H**4)/(8*E*I) # deflection at top, m
    
    # Geometry Definitions
    t_m = (d_out - d_in) / 2       # thickness in meters
    d_mean_m = (d_out + d_in) / 2  # mean diameter in meters

    return [
        sig_b - sigma,              
        tau_s - tau,               
        0.10 - delta,                   
        60 - (d_mean_m / t_m),         
        t_m - 0.005,                   
        0.02 - t_m,                    
        d_out - d_in - 0.001     
    ]

bounds = ((5.0, 50.0), (4.0, 45.0))

# Initial guess in cm
x0 = [47.0, 45.0]

cons = {'type': 'ineq', 'fun': constraints}

# optimization via (Sequential Least Squares Programming) or diff evolution
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
# solution = differential_evolution(objective, bounds, constraints=cons, seed=42)

if solution.success:
    print(f"d_o: {solution.x[0]:.3f} cm")
    print(f"d_i: {solution.x[1]:.3f} cm")
    print(f"Thickness: {(solution.x[0] - solution.x[1])/2:.3f} cm")
    print(f"Mass: {solution.fun:.2f} kg")
else:
    print("\nOptimization Failed.")
    print(solution.message)
