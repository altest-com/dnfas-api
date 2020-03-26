from django.urls import path, include

from .views import (
    LoginAPIView,
    # RegistrationAPIView,
    UserRetrieveUpdateAPIView
)

app_name = 'users'

urlpatterns = [
    path('users/', UserRetrieveUpdateAPIView.as_view(), name='users'),
    path('login/', LoginAPIView.as_view(), name='login')
]
