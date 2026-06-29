import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n_rows = 350

# Generate independent variables
building_age = np.random.uniform(1, 80, n_rows) # age in years
corrosion_level = np.random.uniform(0.0, 1.0, n_rows) # corrosion factor 0 to 1
crack_width = np.random.uniform(0.0, 10.0, n_rows) # crack width in mm
crack_density = np.random.uniform(0.0, 20.0, n_rows) # cracks per m^2
moisture_content = np.random.uniform(1.0, 15.0, n_rows) # moisture percentage
temperature = np.random.uniform(10.0, 45.0, n_rows) # temperature in Celsius
humidity = np.random.uniform(30.0, 90.0, n_rows) # relative humidity in %
load_stress = np.random.uniform(5.0, 35.0, n_rows) # load stress in MPa

# Calculate dependent variables with noise

# Compressive strength: base 45 MPa, decreases with age, corrosion, moisture
base_strength = 45.0
compressive_strength = base_strength - (building_age * 0.15) - (corrosion_level * 12.0) - (moisture_content * 0.4)
compressive_strength += np.random.normal(0, 2.0, n_rows)
compressive_strength = np.clip(compressive_strength, 12.0, 55.0)

# Structural Health Index (SHI): 0 to 100, lower is worse
shi = 100.0 - (building_age * 0.35) - (corrosion_level * 20.0) - (crack_width * 3.0) - (crack_density * 1.2) - (load_stress * 0.4) - (moisture_content * 0.5)
shi += np.random.normal(0, 3.0, n_rows)
shi = np.clip(shi, 5.0, 100.0)

# Risk Level: based on SHI
risk_level = []
for s in shi:
    if s > 75.0:
        risk_level.append("Safe")
    elif s > 45.0:
        risk_level.append("Moderate")
    else:
        risk_level.append("Critical")

# Remaining Useful Life (RUL): in years, decreases with age and lower SHI
base_life = 85.0
rul = (base_life - building_age) * (shi / 100.0)
rul += np.random.normal(0, 2.0, n_rows)
# RUL cannot be longer than base_life - building_age
rul = np.minimum(rul, base_life - building_age)
rul = np.clip(rul, 0.0, 85.0)

# Create DataFrame
df = pd.DataFrame({
    'building_age': np.round(building_age, 1),
    'corrosion_level': np.round(corrosion_level, 2),
    'crack_width': np.round(crack_width, 2),
    'crack_density': np.round(crack_density, 2),
    'moisture_content': np.round(moisture_content, 1),
    'compressive_strength': np.round(compressive_strength, 1),
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'load_stress': np.round(load_stress, 1),
    'risk_level': risk_level,
    'shi': np.round(shi, 1),
    'rul': np.round(rul, 1)
})

# Save to data/tabular/structural_data.csv
out_dir = os.path.join("data", "tabular")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "structural_data.csv")
df.to_csv(out_path, index=False)

print(f"Tabular dataset created at {out_path} with {len(df)} rows.")
