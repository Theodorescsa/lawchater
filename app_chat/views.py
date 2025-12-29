from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import Conversation, Message
from .serializers import (
    ConversationSerializer,
    ConversationListSerializer,
    MessageSerializer,
    ChatRequestSerializer,
)

from core.app import get_rag_service  # bạn đã có sẵn rag_service

rag_service = get_rag_service()
class ConversationViewSet(viewsets.ModelViewSet):
    """
    CRUD Conversation:
    - GET   /api/chat/conversations/
    - POST  /api/chat/conversations/
    - GET   /api/chat/conversations/{id}/
    - PATCH /api/chat/conversations/{id}/
    - DELETE /api/chat/conversations/{id}/
    """
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(
            user=self.request.user
        ).order_by("-updated_at")

    def get_serializer_class(self):
        if self.action == "list":
            return ConversationListSerializer
        return ConversationSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    - GET /api/chat/conversations/{conversation_id}/messages/
    """
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        conversation_id = self.kwargs["conversation_id"]
        conv = get_object_or_404(
            Conversation, id=conversation_id, user=self.request.user
        )
        return conv.messages.all()


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chat_with_rag(request):
    """
    Chat API có lưu lịch sử:

    Body:
    {
      "message": "câu hỏi",
      "conversation_id": 1,   // optional
      "k": 3                  // optional
    }
    """
    serializer = ChatRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    data = serializer.validated_data

    message_text = data["message"]
    conversation_id = data.get("conversation_id")
    k = data.get("k", 3)

    # 1. Lấy / tạo conversation
    if conversation_id:
        conv = get_object_or_404(
            Conversation, id=conversation_id, user=request.user
        )
    else:
        conv = Conversation.objects.create(
            user=request.user,
            title=message_text[:60],
        )

    # 2. Lưu message bên user
    user_msg = Message.objects.create(
        conversation=conv,
        role="user",
        content=message_text,
    )

    # 3. Gọi RAG
    result = rag_service.query(question=message_text, k=k)
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # 4. Lưu message bên assistant
    assistant_msg = Message.objects.create(
        conversation=conv,
        role="assistant",
        content=answer,
        meta={"sources": sources},
    )

    conv.save()  # update updated_at

    return Response(
        {
            "conversation_id": conv.id,
            "question": message_text,
            "answer": answer,
            "sources": sources,
            "messages": {
                "user": MessageSerializer(user_msg).data,
                "assistant": MessageSerializer(assistant_msg).data,
            },
        },
        status=status.HTTP_200_OK,
    )
    # return Response({
    #     "question": "Tôi ly hôn thì mất bao nhiêu tiền?",
    #     "answer": "… câu trả lời dài …",
    #     "sources": [
    #         {
    #         "content": "Điều 68. Tuyên bố mất tích…",
    #         "metadata": {
    #             "source": "/app/core/data/BLDS.docx"
    #         }
    #         },
    #         {
    #         "content": "Điều 68. Tuyên bố mất tích…",
    #         "metadata": {
    #             "source": "/app/core/data/BLDS.docx"
    #         }
    #         }
    #     ]
    #     }
    # )