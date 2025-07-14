from django.urls import path
from .views import microsoft_auth

urlpatterns = [
    path("auth/", microsoft_auth),
]
