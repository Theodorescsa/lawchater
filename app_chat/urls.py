from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ConversationViewSet, MessageViewSet, chat_with_rag

router = DefaultRouter()
router.register(r"conversations", ConversationViewSet, basename="conversation")

urlpatterns = [
    path("", include(router.urls)),
    path(
        "conversations/<int:conversation_id>/messages/",
        MessageViewSet.as_view({"get": "list"}),
        name="conversation-messages",
    ),
    path("chat/", chat_with_rag, name="chat-with-rag"),
]
