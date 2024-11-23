from django.urls import path
from .views import debug_env_vars

urlpatterns = [
    path('debug-env/', debug_env_vars, name='debug_env_vars'),
]
