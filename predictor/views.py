from django.shortcuts import render
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from .forms import BabyDataForm
import os
from datetime import datetime

# --- Load Saved Model and Preprocessing Objects ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'infant_health_risk_model.keras')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'ml_model', 'preprocessor.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_model', 'label_encoder.pkl')

model, preprocessor, label_encoder = None, None, None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("--- ✅ Model and preprocessors loaded successfully! ---")
except Exception as e:
    print(f"--- ❌ CRITICAL ERROR on startup: Could not load model files: {e} ---")

def predict_wellness(request):
    prediction, prediction_proba, error_message = None, None, None
    form = BabyDataForm()

    if not all([model, preprocessor, label_encoder]):
        error_message = "A critical error occurred: The AI model files could not be loaded. Please check the server console for details."

    if request.method == 'POST':
        form = BabyDataForm(request.POST)

        if form.is_valid():
            if error_message:
                pass 
            else:
                try:
                    data = form.cleaned_data
                    input_df = pd.DataFrame([data])
                    
                    # --- FIX: Add missing dummy columns the preprocessor expects ---
                    # The preprocessor was likely trained with a 'date' column, which we don't use.
                    # We add it here to prevent an error.
                    input_df['date'] = datetime.now().strftime("%Y-%m-%d")

                    # Feature Engineering
                    if 'weight_kg' in input_df.columns and 'birth_weight_kg' in input_df.columns and 'age_days' in input_df.columns:
                        input_df['weight_gain_per_day'] = (input_df['weight_kg'] - input_df['birth_weight_kg']) / input_df['age_days']
                        input_df.replace([np.inf, -np.inf], 0, inplace=True)
                        input_df.fillna({'weight_gain_per_day': 0}, inplace=True)
                    
                    # Preprocess, Predict, and Decode
                    processed_input = preprocessor.transform(input_df).toarray()
                    pred_proba_array = model.predict(processed_input)[0]
                    pred_index = np.argmax(pred_proba_array)
                    prediction = label_encoder.inverse_transform([pred_index])[0]
                    prediction_proba = f"{pred_proba_array[pred_index] * 100:.2f}%"
                    print(f"--- ✅ PREDICTION SUCCESS: '{prediction}' with {prediction_proba} confidence. ---")

                except Exception as e:
                    print(f"--- ❌ ERROR during prediction logic: {e} ---")
                    error_message = f"An unexpected error occurred during prediction: {e}"
        else:
            error_message = "The form is not valid. Please check the values you entered."

    context = {
        'form': form,
        'prediction': prediction,
        'prediction_proba': prediction_proba,
        'error': error_message,
    }
    return render(request, 'predictor/home.html', context)