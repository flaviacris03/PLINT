"""
EOS Data and Functions
"""

from .eos_properties import material_properties
from scipy.interpolate import interp1d
import numpy as np

def mie_gruneisen_debye(P, P0, rho0, K0, K0prime, gamma0, theta0, V0, T):
    """
    Calculates density from the Mie-Grüneisen-Debye EOS.
    """
    V = V0 * (P0 / (P + P0))**(1/K0prime)
    rho = rho0 * (V0 / V)

    # Thermal pressure correction (simplified)
    gamma = gamma0 * (V / V0)**1  # Assuming q = 1 for simplicity
    theta = theta0 * (V / V0)**(-gamma)
    P_thermal = (gamma * rho * 8.314 * (T - 300))  # Simplified thermal pressure

    return rho

def birch_murnaghan(P, P0, rho0, K0, K0prime, V0):
    """
    Calculates density from the 3rd order Birch-Murnaghan EOS.
    """
    eta = (3/2) * (K0prime - 4)
    V = V0 * (1 + (3/4) * (K0prime - 4) * ((P - P0)/K0))**(-2/(3 * (K0prime - 4)))
    
    density = rho0 * (V0 / V)

    return density


# --- Temperature Profile (Adiabatic) ---
"""def calculate_temperature(radius, core_radius, surface_temp, cmb_temp, core_temp, radius_guess):
    
    Calculates an adiabatic temperature profile.
    
    if radius > core_radius:
        # Mantle adiabat
        temperature = surface_temp + (cmb_temp - surface_temp) * (
            (radius_guess - radius) / (radius_guess - core_radius)
        )**0.45
    else:
        # Core adiabat
        temperature = core_temp * (1 - (radius / core_radius)**2)**0.3 + core_temp * 0.8
    return temperature"""

def calculate_temperature(radii, core_radius, surface_temp, cmb_temp, material_properties, gravity, density, K_s, dr):
    """
    Computes the temperature profile inward from the surface using the Runge-Kutta 4th order method (RK4).
    Parameters:
    - radii: Array of radii (m) from the center to the surface.
    - core_radius: Radius of core-mantle boundary (m).
    - surface_temp: Surface temperature (K).
    - cmb_temp: Core-mantle boundary temperature (K).
    - material_properties: Material-specific parameters (rho, K, gamma).
    - gravity: Array of gravitational accelerations (m/s^2) corresponding to radii.
    - density: Array of densities (kg/m^3) corresponding to radii.
    - K_s: Isentropic bulk modulus (Pa).
    - dr: Step size in radial direction (meters).

    Returns:
    - temperature_profile: Array of temperatures (K) corresponding to the radii, starting from the surface.
    """
    # Initialize temperature profile array
    temperature = np.zeros_like(radii)
    temperature[-1] = surface_temp  # Surface temperature as the starting point

    # Iterate inward from the surface (reverse order in the arrays)
    for i in range(len(radii)-1, 0, -1):
        radius = radii[i]
        
        # Define the adiabatic gradient equation
        def adiabatic_gradient(T, radius_idx):
            return -(density[radius_idx] * gravity[radius_idx] * gamma * T) / K_s
   
        # Handle mantle region
        if radius > core_radius:
            material = "mantle"
            gamma = material_properties[material]["gamma0"]
            #print(f"i={i}, radius={radius}, temperature={temperature[i]}")

        # Handle transition at CMB
        elif radius == core_radius:
            temperature[i] = cmb_temp
            #print(f"i={i}, radius={radius}, temperature={temperature[i]}")

        # Handle core region
        else:
            material = "core"
            #print(f"i={i}, radius={radius}, temperature={temperature[i]}")
            gamma = material_properties[material]["gamma0"]

        # Apply RK4 Method
        current_temp = temperature[i]
        k1 = -dr * adiabatic_gradient(current_temp, i)
        k2 = -dr * adiabatic_gradient(current_temp + 0.5 * k1, i - 1)
        k3 = -dr * adiabatic_gradient(current_temp + 0.5 * k2, i - 1)
        k4 = -dr * adiabatic_gradient(current_temp + k3, i - 2)

        # Update temperature at the previous radius
        temperature[i - 1] = current_temp + (k1 + 2 * k2 + 2 * k3 + k4) / 6


    return temperature


# --- EOS Calculation ---
def calculate_density(pressure, radius, core_radius, material, radius_guess, cmb_temp, core_temp, eos_choice, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS."""

    T = 0  # Temporary fix for tabulated EOS
    props = material_properties[material]

    props = material_properties[material]  # Shorthand

    if eos_choice == "Mie-Gruneisen-Debye":
        density = mie_gruneisen_debye(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["gamma0"], props["theta0"], props["V0"], T)
        return density
    elif eos_choice == "Birch-Murnaghan":
        density = birch_murnaghan(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["V0"])
        return density
    elif eos_choice == "Tabulated":
        try:
            eos_file = props["eos_file"]
            # Caching: Store interpolation functions for reuse
            if eos_file not in interpolation_functions:
                data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
                pressure_data = data[:, 1] * 1e9
                density_data = data[:, 0] * 1e3
                interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

            interpolation_function = interpolation_functions[eos_file]  # Retrieve from cache
            density = interpolation_function(pressure)  # Call to cached function

            if density is None or np.isnan(density):
                raise ValueError(f"Density calculation failed for {material} at {pressure:.2e} Pa.")

            return density

        except (ValueError, OSError) as e: # Catch file errors
            print(f"Error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None
        except Exception as e: # Other errors
            print(f"Unexpected error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None

    else:
        raise ValueError("Invalid EOS choice.")
    



