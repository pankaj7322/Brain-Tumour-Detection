# detection/urls.py
from django.urls import path
from . import views
from .views import login_view


urlpatterns = [
    path('', login_view, name='login'),
    path("home/", views.home, name = "home"),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path("preprocess", views.dataset_preprocessing, name = 'preprocess'),
    path("train-model",views.train_model,name='train_model'),
    path("pr",views.tumor_classification, name='tumor_classification'),
    path("graph/",views.graph,name='graph'),
]
