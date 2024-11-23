from django.urls import path
from .views import recommend_recipes

urlpatterns = [
    path('recommend/', recommend_recipes, name='recommend_recipes'),
]
