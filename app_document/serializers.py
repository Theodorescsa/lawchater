from rest_framework import serializers
from .models import UploadedDocument


class UploadedDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocument
        fields = [
            "id",
            "original_name",
            "file",
            "status",
            "error_message",
            "created_at",
            "processed_at",
        ]
        read_only_fields = ["status", "error_message", "created_at", "processed_at"]
