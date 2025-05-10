import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('debris_parameters.csv')
if 'SATELLITE_MASS' not in df.columns:
    df['SATELLITE_MASS'] = 350

features = [
    'SEMIMAJOR_AXIS', 'ECCENTRICITY', 'INCLINATION',
    'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
    'MEAN_MOTION', 'VELOCITY_PERIGEE', 'VELOCITY_APOGEE'
]

def compute_realistic_collision_probability(sat, obj):
    norm_a = (sat[0] - obj[0]) / 100
    norm_e = (sat[1] - obj[1]) / 0.01
    norm_i = (sat[2] - obj[2]) / 5
    distance = np.sqrt(norm_a**2 + norm_e**2 + norm_i**2)
    prob = 1 / (1 + np.exp((distance - 2.0)))
    return np.clip(prob, 0, 1)

def predict_collision(sat, obj):
    prob = compute_realistic_collision_probability(sat, obj)
    rel_vel = np.linalg.norm(np.array(obj[7:9]) - np.array(sat[7:9]))
    return prob, rel_vel, np.array(obj[7:9]) - np.array(sat[7:9])

def determine_impact_area(prob, rel_vel, debris_density):
    thresholds = (0.7, 0.4) if debris_density > 0.8 else (0.6, 0.3)
    if prob > thresholds[0] and rel_vel > 2.0:
        return "High Impact Area"
    elif prob > thresholds[1] and rel_vel > 1.0:
        return "Medium Impact Area"
    return "Low Impact Area"

def calculate_escape_angle(vec):
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return angle % 360

def calculate_new_semi_major_axis(v, delta_v, r, mu=3.986e14):
    try:
        v_final = v + delta_v
        return mu / (2 / r - (v_final ** 2) / mu) / 1000
    except:
        return None

def calculate_thrust_and_adjustments(area, rel_vel, rel_vec, mass, I_sp=300, g=9.81):
    if area == "High Impact Area":
        delta_v = rel_vel
    elif area == "Medium Impact Area":
        delta_v = rel_vel / 2
    else:
        return {
            'thrust_needed': 0, 'fuel_used': 0,
            'adjustment_angle': 0, 'maneuver_time': None,
            'new_semi_major_axis': None
        }

    if mass <= 0 or delta_v <= 0 or I_sp <= 0 or g <= 0:
        return {
            'thrust_needed': 0, 'fuel_used': 0,
            'adjustment_angle': 0, 'maneuver_time': None,
            'new_semi_major_axis': None
        }

    thrust_needed = mass * delta_v / 20
    final_mass = mass * np.exp(-delta_v / (I_sp * g))
    fuel_used = mass - final_mass
    tw_ratio = thrust_needed / (mass * g)
    maneuver_time = delta_v / (tw_ratio * g) if tw_ratio > 0 else None

    return {
        'thrust_needed': round(thrust_needed, 2),
        'fuel_used': round(fuel_used, 2),
        'adjustment_angle': round(calculate_escape_angle(rel_vec), 2),
        'maneuver_time': round(maneuver_time, 2) if maneuver_time else None,
        'new_semi_major_axis': round(calculate_new_semi_major_axis(7.8, delta_v, 7000e3), 2)
    }

def evaluate_collisions_and_decisions(sat, debris_density=0.6):
    for _, obj in df.iterrows():
        obj_data = obj[features].values
        prob, rel_vel, rel_vec = predict_collision(sat, obj_data)
        area = determine_impact_area(prob, rel_vel, debris_density)
        if area != "Low Impact Area":
            maneuver = calculate_thrust_and_adjustments(area, rel_vel, rel_vec, obj['SATELLITE_MASS'])
            return {
                'thrust_needed': maneuver['thrust_needed'],
                'fuel_used': maneuver['fuel_used'],
                'adjustment_angle': maneuver['adjustment_angle'],
                'maneuver_time': maneuver['maneuver_time'],
                'new_semi_major_axis': maneuver['new_semi_major_axis'],
                'impact_area': area,
                'collision_probability': round(prob * 100, 2),
                'relative_velocity': round(rel_vel, 2)
            }

    return {
        'thrust_needed': 0, 'fuel_used': 0,
        'adjustment_angle': 0, 'maneuver_time': None,
        'new_semi_major_axis': None,
        'impact_area': "Low Impact Area",
        'collision_probability': 0.0,
        'relative_velocity': 0.0
    }