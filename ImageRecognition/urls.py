from django.urls import path
from .views import analyze_image

urlpatterns = [
    path('analyze-image/', analyze_image, name='analyze_image'),
]
