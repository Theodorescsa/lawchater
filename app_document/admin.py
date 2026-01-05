import os
from django.contrib import admin
from django.utils.html import format_html
from .models import UploadedDocument

@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ('original_name', 'status_badge', 'file_link', 'user', 'created_at', 'processed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('original_name', 'user__username', 'error_message')
    readonly_fields = (
        'created_at',
        'processed_at',
        'status',
        'error_message',
        'file_preview',
    )

    actions = ["mark_as_pending", "ingest_selected_documents"]

    # ---------- UI ----------
    def status_badge(self, obj):
        colors = {
            'pending': 'orange',
            'processing': 'blue',
            'done': 'green',
            'failed': 'red',
        }
        return format_html(
            '<span style="color:white;background:{};padding:3px 8px;border-radius:3px;font-weight:bold;">{}</span>',
            colors.get(obj.status, 'black'),
            obj.status.upper()
        )
    status_badge.short_description = "Status"

    def file_link(self, obj):
        if obj.file:
            return format_html('<a href="{}" target="_blank">Download</a>', obj.file.url)
        return "-"
    file_link.short_description = "File"

    def file_preview(self, obj):
        return obj.file.path if obj.file else ""

    # ---------- ACTIONS ----------
    def mark_as_pending(self, request, queryset):
        safe_qs = queryset.exclude(status='processing')
        updated = safe_qs.update(
            status='pending',
            error_message='',
            processed_at=None
        )
        self.message_user(
            request,
            f"Đã chuyển {updated} tài liệu về Pending (bỏ qua file đang Processing)."
        )
    mark_as_pending.short_description = "Re-process selected documents (Set to Pending)"

    def ingest_selected_documents(self, request, queryset):
        from core.ingest_single import ingest_one_document

        count = 0
        for doc in queryset:
            if doc.status in ["done", "processing"]:
                continue
            ingest_one_document(doc)
            count += 1

        self.message_user(request, f"Đã ingest {count} document.")
    ingest_selected_documents.short_description = "Ingest selected documents"

    def has_delete_permission(self, request, obj=None):
        if obj and obj.status == 'processing':
            return False
        return super().has_delete_permission(request, obj)
