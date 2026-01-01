from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
import json
from .models import Conversation, Message

class MessageInline(admin.TabularInline):
    """
    Cho phép xem và sửa nhanh tin nhắn ngay trong màn hình Conversation
    """
    model = Message
    extra = 0  # Không hiện các dòng trống thừa
    readonly_fields = ('created_at', 'formatted_meta') # Chỉ đọc thời gian và meta đã format
    fields = ('role', 'content', 'formatted_meta', 'created_at')
    can_delete = True
    classes = ['collapse'] # Mặc định thu gọn nếu muốn gọn gàng

    def formatted_meta(self, instance):
        """Format JSON đẹp mắt trong Inline"""
        if not instance.meta:
            return ""
        # Convert JSON object sang string có thụt đầu dòng
        response = json.dumps(instance.meta, indent=2, ensure_ascii=False)
        # Bọc trong thẻ pre để giữ định dạng
        return format_html('<pre style="font-size: 11px; line-height: 1.2;">{}</pre>', response)
    
    formatted_meta.short_description = "Meta Data"


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title_preview', 'user_info', 'message_count', 'is_archived', 'updated_at')
    list_filter = ('is_archived', 'created_at', 'updated_at')
    search_fields = ('title', 'user__username', 'user__email')
    readonly_fields = ('created_at', 'updated_at')
    
    # Gắn Inline vào để xem tin nhắn bên dưới
    inlines = [MessageInline]
    
    # Tối ưu query (tránh N+1 query khi load user)
    list_select_related = ('user',)

    def user_info(self, obj):
        return f"{obj.user.username} (ID: {obj.user.id})"
    user_info.short_description = "User"

    def title_preview(self, obj):
        return obj.title if obj.title else "(No Title)"
    title_preview.short_description = "Conversation Title"

    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = "Msgs"


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """
    Quản lý tin nhắn riêng lẻ (thường dùng để debug hoặc search nội dung)
    """
    list_display = ('id', 'role', 'content_preview', 'conversation_link', 'created_at')
    list_filter = ('role', 'created_at')
    search_fields = ('content', 'conversation__title', 'conversation__user__username')
    readonly_fields = ('created_at', 'formatted_meta_detail')
    
    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content
    content_preview.short_description = "Content"

    def conversation_link(self, obj):
        # Tạo link bấm vào để nhảy sang trang Conversation cha
        url = f"/admin/app_chat/conversation/{obj.conversation.id}/change/"
        return format_html('<a href="{}">{}</a>', url, obj.conversation)
    conversation_link.short_description = "Conversation"

    def formatted_meta_detail(self, instance):
        """Format JSON cho trang chi tiết Message"""
        if not instance.meta:
            return "-"
        response = json.dumps(instance.meta, indent=2, ensure_ascii=False)
        return format_html('<pre>{}</pre>', response)
    formatted_meta_detail.short_description = "Meta Data (JSON)"