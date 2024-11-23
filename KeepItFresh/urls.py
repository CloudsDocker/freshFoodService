from django.contrib import admin
from django.urls import path, include
from shopping_list.views import homepage

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/image-recognition/', include('ImageRecognition.urls')),
    path('api/shopping-list/', include('shopping_list.urls')),  # Map to shopping_list app
    path('', homepage),
    path('api/leftover-recommendation/', include('leftover_recommender.urls')),# Root URL
    path('debug/', include('debug.urls')),
]
