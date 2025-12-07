from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UploadedDocumentViewSet

router = DefaultRouter()
router.register(r"uploads", UploadedDocumentViewSet, basename="uploads")

urlpatterns = [
    path("", include(router.urls)),
]
