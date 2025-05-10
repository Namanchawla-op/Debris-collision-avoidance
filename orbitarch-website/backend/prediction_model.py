import numpy as np

# Compute perigee and apogee velocities from orbital parameters
def compute_velocities_from_orbit(a_km, e):
    """
    Calculate perigee and apogee velocities (in km/s) based on semi-major axis and eccentricity.
    """
    mu = 3.986e14  # Earth's gravitational parameter in m^3/s^2
    a_m = a_km * 1000  # Convert km to meters
    r_perigee = a_m * (1 - e)
    r_apogee = a_m * (1 + e)

    v_perigee = np.sqrt(mu * (2 / r_perigee - 1 / a_m))
    v_apogee = np.sqrt(mu * (2 / r_apogee - 1 / a_m))

    return v_perigee / 1000, v_apogee / 1000  # Return in km/s

# Compute collision probability using sigmoid of orbital differences
def compute_realistic_collision_probability(sat):
    """
    Estimate collision probability using normalized differences in orbital parameters.
    """
    norm_a = (sat['SEMIMAJOR_AXIS'] - 7000) / 100
    norm_e = (sat['ECCENTRICITY'] - 0.001) / 0.01
    norm_i = (sat['INCLINATION'] - 10) / 5

    distance = np.sqrt(norm_a**2 + norm_e**2 + norm_i**2)
    prob = 1 / (1 + np.exp(distance - 2.0))  # Sigmoid function
    return np.clip(prob, 0, 1)

# Predict the satellite's damage risk and orbital plane
def predict_damage_location_and_plane(sat):
    """
    Given satellite data, return its orbital plane, damage risk, and corrected collision probability in %.
    """
    # Compute missing velocity values
    if sat.get('VELOCITY_PERIGEE', 0) == 0 or sat.get('VELOCITY_APOGEE', 0) == 0:
        v_per, v_apo = compute_velocities_from_orbit(
            sat['SEMIMAJOR_AXIS'], sat['ECCENTRICITY']
        )
        sat['VELOCITY_PERIGEE'] = v_per
        sat['VELOCITY_APOGEE'] = v_apo

    # Determine orbital plane from inclination
    inclination = sat['INCLINATION']
    planes = {
        "Plane 1": (0, 10), "Plane 2": (10, 30), "Plane 3": (30, 50),
        "Plane 4": (50, 70), "Plane 5": (70, 90), "Plane 6": (90, 100)
    }
    plane = next((name for name, (lo, hi) in planes.items() if lo <= inclination < hi), "Unknown")

    # Calculate probability and velocity
    prob = compute_realistic_collision_probability(sat)
    avg_vel = (sat['VELOCITY_PERIGEE'] + sat['VELOCITY_APOGEE']) / 2

    # Determine damage risk
    if prob > 0.8 and avg_vel > 7.5:
        damage = "High Damage"
    elif prob > 0.5:
        damage = "Minor Damage"
    else:
        damage = "No Damage"

    # 💥 Final fix: if probability was wrongly scaled, divide by 100
    fixed_prob = prob if prob <= 1 else prob / 100

    # Convert to percentage format for display
    return plane, damage, round(fixed_prob * 100, 2)