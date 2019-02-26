from django.urls import path
from . import views

urlpatterns = [
    path('', views.listBlogEntries, name='listBlogEntries'),
    path('post/<int:pk>/', views.BlogEntryDetails, name='BlogEntryDetails'),
    path('post/new/', views.createNewBlogEntry, name='createNewBlogEntry'),
    path('post/<int:pk>/edit/', views.editBlogEntry, name='editBlogEntry'),
]