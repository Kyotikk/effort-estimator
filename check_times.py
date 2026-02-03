import pandas as pd
from datetime import datetime

# Windows time
df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/effort_estimation_output/parsingsim3_sim_elderly3/fused_aligned_10.0s.csv')
print('Windows t_center range:')
print(f'  Min: {df["t_center"].min()} ({datetime.fromtimestamp(df["t_center"].min())})')
print(f'  Max: {df["t_center"].max()} ({datetime.fromtimestamp(df["t_center"].max())})')

# ADL times
adl_df = pd.read_csv('/Users/pascalschlegel/data/interim/parsingsim3/sim_elderly3/scai_app/ADLs_1.csv', skiprows=2)
def parse_adl_time(time_str):
    try:
        dt = datetime.strptime(time_str, '%d-%m-%Y-%H-%M-%S-%f')
        return dt.timestamp()
    except:
        return None
adl_df['unix_time'] = adl_df['Time'].apply(parse_adl_time)
print('\nADL time range:')
print(f'  Min: {adl_df["unix_time"].min()} ({datetime.fromtimestamp(adl_df["unix_time"].min())})')
print(f'  Max: {adl_df["unix_time"].max()} ({datetime.fromtimestamp(adl_df["unix_time"].max())})')
