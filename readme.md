## code for the model training
````python
```

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

print("--- Script started. ---")


csv_file_name = 'newborn_health_monitoring_with_risk.csv'

OUTPUT_DIR = 'saved_model_assets'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'infant_health_risk_model.keras')
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
ENCODER_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')

try:
    df = pd.read_csv(csv_file_name)
    print(f"--- Successfully loaded '{csv_file_name}' ---")
except FileNotFoundError:
    print(f"--- ❌ ERROR: The file '{csv_file_name}' was not found. Please make sure it's in the same directory as this script. ---")
    exit()

df_processed = df.drop(['baby_id', 'name'], axis=1, errors='ignore')
if 'weight_kg' in df_processed.columns and 'birth_weight_kg' in df_processed.columns and 'age_days' in df_processed.columns:
    df_processed['weight_gain_per_day'] = (df_processed['weight_kg'] - df_processed['birth_weight_kg']) / df_processed['age_days']
    df_processed.replace([np.inf, -np.inf], 0, inplace=True)
    df_processed.fillna({'weight_gain_per_day': 0}, inplace=True)

for col in df_processed.select_dtypes(include=np.number).columns:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

X = df_processed.drop('risk_level', axis=1)
y = df_processed['risk_level']

print("--- Creating and fitting preprocessors on the full dataset... ---")

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
label_encoder = LabelEncoder()

preprocessor.fit(X)
label_encoder.fit(y)

with open(PREPROCESSOR_PATH, 'wb') as f:
    pickle.dump(preprocessor, f)
print(f"--- ✅ Preprocessor saved to '{PREPROCESSOR_PATH}' ---")

with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"--- ✅ Label Encoder saved to '{ENCODER_PATH}' ---")

print("\n--- Training final model on the full dataset... ---")

X_processed = preprocessor.transform(X).toarray()
y_encoded = label_encoder.transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

n_features = X_processed.shape[1]
n_classes = y_categorical.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_processed, y_categorical, epochs=50, batch_size=32, verbose=1)

model.save(MODEL_PATH)
print(f"--- ✅ Model saved to '{MODEL_PATH}' ---")
print("\n--- All files generated successfully! You can now move them to your Django project's 'ml_model' folder. ---")


```
````

## Example values for risk

    Weeks of Gestational Age at Birth: 34

    Birth Weight (kg): 2.2

    Birth Length (cm): 45

    Birth Head Circumference (cm): 30

    Apgar Score at Birth: 5

    Gender: Male

    Current Age (in days): 14

    Current Weight (kg): 2.1 (Has not regained birth weight)

    Current Length (cm): 45.5

    Current Head Circumference (cm): 30.5

    Body Temperature (°C): 36.2 (Slightly low)

    Heart Rate (beats per minute): 180

    Respiration Rate (breaths per minute): 65 (Tachypnea)

    Oxygen Saturation (%): 92 (Low)

    Feeding type: Formula

    Feeding Frequency (times per day): 5 (Infrequent)

    Urine Output (diapers in 24 hours): 3 (Sign of dehydration)

    Stool Count (in 24 hours): 1

    Jaundice Level (mg/dL): 16.0 (High, needs phototherapy)

    Immunizations Up to Date?: No

    Reflexes Appear Normal?: No (Described as weak/sluggish)


## Example values for not risk

     Weeks of Gestational Age at Birth: 40

    Birth Weight (kg): 3.4

    Birth Length (cm): 50

    Birth Head Circumference (cm): 35

    Apgar Score at Birth: 9

    Gender: Female

    Current Age (in days): 75

    Current Weight (kg): 5.8

    Current Length (cm): 59

    Current Head Circumference (cm): 40

    Body Temperature (°C): 37.0

    Heart Rate (beats per minute): 125

    Respiration Rate (breaths per minute): 40

    Oxygen Saturation (%): 99

    Feeding type: Breastfed

    Feeding Frequency (times per day): 8

    Urine Output (diapers in 24 hours): 7

    Stool Count (in 24 hours): 3

    Jaundice Level (mg/dL): 0.5

    Immunizations Up to Date?: Yes

    Reflexes Appear Normal?: Yes

