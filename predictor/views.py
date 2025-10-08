from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from .forms import BabyDataForm
from .models import WellnessPrediction # NEW: Import the model
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

                    # --- NEW: Save the valid form data and prediction to the database ---
                    try:
                        # Convert probability string '98.75%' to float 98.75 for saving
                        proba_float = float(prediction_proba.strip('%'))
                        
                        # Create the record in the database
                        WellnessPrediction.objects.create(
                            **data,
                            prediction=prediction,
                            prediction_proba=proba_float
                        )
                        print("--- ✅ Prediction saved to database. ---")
                    except Exception as e:
                        print(f"--- ❌ ERROR saving to database: {e} ---")
                        # You might want to handle this error, but for now, we'll just print it

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

def prediction_history(request):
    # Fetch all prediction records from the database, ordering by the newest first
    predictions = WellnessPrediction.objects.all().order_by('-created_at')
    
    context = {
        'predictions': predictions,
    }
    return render(request, 'predictor/history.html', context)

def delete_prediction(request, record_id):
    # We only want to allow deletion via POST request for security
    if request.method == 'POST':
        try:
            # Find the prediction record by its ID and delete it
            record = WellnessPrediction.objects.get(id=record_id)
            record.delete()
        except WellnessPrediction.DoesNotExist:
            # Handle the case where the record doesn't exist
            pass
    
    # Redirect back to the history page regardless of the method or outcome
    return redirect('history')


