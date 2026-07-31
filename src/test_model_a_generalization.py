"""
Check Model A (Electrical Only) vs Model B (Electrical + Time) on Temporal Generalization Splits
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

DATA_PATH = r"c:\Users\saury\Desktop\nirvana_grid\data\processed\cleaned_dataset.csv"
df = pd.read_csv(DATA_PATH)
df['relay'] = df['relay'].map({'ON': 1, 'OFF': 0}).fillna(0)
df['timestamp_ist_dt'] = pd.to_datetime(df['timestamp_ist_dt'], format='mixed')

df['hour'] = df['timestamp_ist_dt'].dt.hour
df['day_of_week'] = df['timestamp_ist_dt'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['apparent_power'] = df['voltage'] * df['current']
df['reactive_power'] = np.sqrt(np.maximum(df['apparent_power']**2 - df['power']**2, 0))
df['current_pf'] = df['current'] * df['pf']
df['power_per_voltage'] = (df['power'] / df['voltage'].replace(0, np.nan)).fillna(0)

base_features = ['voltage', 'current', 'power', 'pf', 'frequency', 'relay']
elec_features = ['apparent_power', 'reactive_power', 'current_pf', 'power_per_voltage']
time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']

MODEL_A_FEAT = base_features + elec_features
MODEL_B_FEAT = base_features + elec_features + time_features

def comp_split(train_mask, test_mask, name):
    ma = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1).fit(df.loc[train_mask, MODEL_A_FEAT], df.loc[train_mask, 'label'])
    mb = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1).fit(df.loc[train_mask, MODEL_B_FEAT], df.loc[train_mask, 'label'])
    f1_a = f1_score(df.loc[test_mask, 'label'], ma.predict(df.loc[test_mask, MODEL_A_FEAT]), average='macro')
    f1_b = f1_score(df.loc[test_mask, 'label'], mb.predict(df.loc[test_mask, MODEL_B_FEAT]), average='macro')
    err_a = (df.loc[test_mask, 'label'] != ma.predict(df.loc[test_mask, MODEL_A_FEAT])).mean() * 100
    err_b = (df.loc[test_mask, 'label'] != mb.predict(df.loc[test_mask, MODEL_B_FEAT])).mean() * 100
    print(f"{name:35} | Model A (Elec Only): F1={f1_a:.4f} (Err={err_a:.2f}%) | Model B (+Time): F1={f1_b:.4f} (Err={err_b:.2f}%)")

print("==================================================================================================")
comp_split(df['is_weekend'] == 0, df['is_weekend'] == 1, "Train Weekdays -> Test Weekends")
comp_split(df['is_weekend'] == 1, df['is_weekend'] == 0, "Train Weekends -> Test Weekdays")
comp_split((df['hour']>=6)&(df['hour']<18), (df['hour']<6)|(df['hour']>=18), "Train Day -> Test Night")
comp_split((df['hour']<6)|(df['hour']>=18), (df['hour']>=6)&(df['hour']<18), "Train Night -> Test Day")
print("==================================================================================================")
