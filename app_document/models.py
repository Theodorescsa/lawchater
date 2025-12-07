from django.db import models
from django.conf import settings


class UploadedDocument(models.Model):
    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("failed", "Failed"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="uploaded_documents",
    )
    file = models.FileField(upload_to="law_docs/")
    original_name = models.CharField(max_length=255)

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending"
    )
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.original_name
