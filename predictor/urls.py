from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_wellness, name='home'),
    path('history/', views.prediction_history, name='history'),
        path('delete/<int:record_id>/', views.delete_prediction, name='delete_prediction'),
]