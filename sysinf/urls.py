from django.urls import path

from .views import SystemStatsView

app_name = 'sysinf'

urlpatterns = [
    path('system/', SystemStatsView.as_view(), name='system')
]
