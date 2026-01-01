import os
from django.db import models
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# 1. Định nghĩa Storage riêng trỏ thẳng vào 'core/data'
# settings.BASE_DIR trong Docker thường là /app
CORE_DATA_DIR = os.path.join(settings.BASE_DIR, 'core', 'data')
core_storage = FileSystemStorage(location=CORE_DATA_DIR)

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
    
    # 2. Thay đổi FileField để sử dụng storage mới
    # max_length=500 để tránh lỗi nếu đường dẫn file quá dài
    file = models.FileField(storage=core_storage, max_length=500)
    
    original_name = models.CharField(max_length=255)

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending"
    )
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.original_name

    # 3. (Tùy chọn) Tự động xóa file vật lý khi xóa bản ghi trong DB
    def delete(self, *args, **kwargs):
        if self.file:
            self.file.delete(save=False)
        super().delete(*args, **kwargs)