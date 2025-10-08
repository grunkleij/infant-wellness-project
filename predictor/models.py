from django.db import models

class WellnessPrediction(models.Model):
    # --- Choices for CharFields (good practice) ---
    GENDER_CHOICES = [('female', 'Female'), ('male', 'Male')]
    FEEDING_CHOICES = [('breast milk', 'Breast Milk'), ('formula', 'Formula'), ('mixed', 'Mixed')]
    YES_NO_CHOICES = [('yes', 'Yes'), ('no', 'No')]

    # --- Input Fields from the Form ---
    gestational_age_weeks = models.IntegerField()
    birth_weight_kg = models.FloatField()
    birth_length_cm = models.FloatField()
    birth_head_circumference_cm = models.FloatField()
    apgar_score = models.IntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age_days = models.IntegerField()
    weight_kg = models.FloatField()
    length_cm = models.FloatField()
    head_circumference_cm = models.FloatField()
    temperature_c = models.FloatField()
    heart_rate_bpm = models.IntegerField()
    respiratory_rate_bpm = models.IntegerField()
    oxygen_saturation = models.FloatField()
    feeding_type = models.CharField(max_length=20, choices=FEEDING_CHOICES)
    feeding_frequency_per_day = models.IntegerField()
    urine_output_count = models.IntegerField()
    stool_count = models.IntegerField()
    jaundice_level_mg_dl = models.FloatField()
    immunizations_done = models.CharField(max_length=3, choices=YES_NO_CHOICES)
    reflexes_normal = models.CharField(max_length=3, choices=YES_NO_CHOICES)

    # --- Prediction Result Fields ---
    prediction = models.CharField(max_length=50)
    prediction_proba = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: {self.prediction.upper()} on {self.created_at.strftime('%Y-%m-%d')}"