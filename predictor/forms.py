from django import forms

# Define choices to avoid magic strings
GENDER_CHOICES = [('female', 'Female'), ('male', 'Male')]
FEEDING_CHOICES = [('breast milk', 'Breast Milk'), ('formula', 'Formula'), ('mixed', 'Mixed')]
YES_NO_CHOICES = [('yes', 'Yes'), ('no', 'No')]

class BabyDataForm(forms.Form):
    # Field names now EXACTLY match the training data columns
    gestational_age_weeks = forms.IntegerField(
        label="Weeks of Gestational Age at Birth",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 39'})
    )
    birth_weight_kg = forms.FloatField(
        label="Birth Weight (kg)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 3.4'})
    )
    birth_length_cm = forms.FloatField(
        label="Birth Length (cm)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 51'})
    )
    birth_head_circumference_cm = forms.FloatField(
        label="Birth Head Circumference (cm)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 35'})
    )
    apgar_score = forms.IntegerField(
        label="Apgar Score at Birth",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 9'})
    )
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    age_days = forms.IntegerField(
        label="Current Age (in days)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 90'})
    )
    weight_kg = forms.FloatField(
        label="Current Weight (kg)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 6.0'})
    )
    length_cm = forms.FloatField(
        label="Current Length (cm)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 61'})
    )
    head_circumference_cm = forms.FloatField(
        label="Current Head Circumference (cm)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 41'})
    )
    temperature_c = forms.FloatField(
        label="Body Temperature (°C)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 37.0'})
    )
    heart_rate_bpm = forms.IntegerField(
        label="Heart Rate (beats per minute)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 130'})
    )
    respiratory_rate_bpm = forms.IntegerField(
        label="Respiration Rate (breaths per minute)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 40'})
    )
    oxygen_saturation = forms.FloatField(
        label="Oxygen Saturation (%)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 99'})
    )
    feeding_type = forms.ChoiceField(
        choices=FEEDING_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    feeding_frequency_per_day = forms.IntegerField(
        label="Feeding Frequency (times per day)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 8'})
    )
    urine_output_count = forms.IntegerField(
        label="Urine Output (diapers in 24 hours)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 6'})
    )
    stool_count = forms.IntegerField(
        label="Stool Count (in 24 hours)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 3'})
    )
    jaundice_level_mg_dl = forms.FloatField(
        label="Jaundice Level (mg/dL)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 1.5'})
    )
    immunizations_done = forms.ChoiceField(
        label="Immunizations Up to Date?",
        choices=YES_NO_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    reflexes_normal = forms.ChoiceField(
        label="Reflexes Appear Normal?",
        choices=YES_NO_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )