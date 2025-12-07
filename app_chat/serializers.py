from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "meta", "created_at"]
        read_only_fields = ["id", "meta", "created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "is_archived",
            "created_at",
            "updated_at",
            "messages",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "messages"]


class ConversationListSerializer(serializers.ModelSerializer):
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "is_archived",
            "created_at",
            "updated_at",
            "last_message",
        ]

    def get_last_message(self, obj):
        m = obj.messages.last()
        if not m:
            return None
        return {
            "id": m.id,
            "role": m.role,
            "content": m.content[:200],
            "created_at": m.created_at,
        }


class ChatRequestSerializer(serializers.Serializer):
    conversation_id = serializers.IntegerField(required=False)
    message = serializers.CharField()
    k = serializers.IntegerField(required=False, default=3)
