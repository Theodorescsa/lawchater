import os
from django.contrib import admin
from django.utils.html import format_html
from .models import UploadedDocument

@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ('original_name', 'status_badge', 'file_link', 'user', 'created_at', 'processed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('original_name', 'user__username', 'error_message')
    readonly_fields = ('created_at', 'processed_at', 'file_preview')
    
    actions = ['mark_as_pending']

    def status_badge(self, obj):
        """Hiển thị trạng thái bằng màu sắc"""
        colors = {
            'pending': 'orange',
            'processing': 'blue',
            'done': 'green',
            'failed': 'red',
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="color: white; background-color: {}; padding: 3px 8px; border-radius: 3px; font-weight: bold;">{}</span>',
            color, obj.status.upper()
        )
    status_badge.short_description = "Status"

    def file_link(self, obj):
        """Link click vào để tải file về"""
        if obj.file:
            return format_html('<a href="{}" target="_blank">Download</a>', obj.file.url)
        return "-"
    file_link.short_description = "File"

    def file_preview(self, obj):
        """Hiển thị đường dẫn vật lý thực tế trong Docker"""
        if obj.file:
            return f"Path: {obj.file.path}"
        return ""
    
    # --- ACTION TÙY CHỈNH ---
    def mark_as_pending(self, request, queryset):
        """Chuyển trạng thái các file đã chọn về Pending để worker quét lại"""
        updated = queryset.update(status='pending', error_message='', processed_at=None)
        self.message_user(request, f"Đã chuyển {updated} tài liệu về trạng thái Pending.")
    mark_as_pending.short_description = "Re-process selected documents (Set to Pending)"