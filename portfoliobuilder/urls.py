from django.urls import path

from portfoliobuilder import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('charts/', views.charts, name='charts')
]
