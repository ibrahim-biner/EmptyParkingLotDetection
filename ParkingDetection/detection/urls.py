from django.urls import path
from detection import views

urlpatterns = [
    # detection ile ilgili URL'ler
    path('photo/<int:photo_id>/', views.show_result, name='show_result'),
]
