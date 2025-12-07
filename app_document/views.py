from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action

from django.utils import timezone

from .models import UploadedDocument
from .serializers import UploadedDocumentSerializer

# from core.ingest import main as ingest_main  # dùng lại hàm ingest bạn đã có


class UploadedDocumentViewSet(viewsets.ModelViewSet):
    """
    - GET  /api/docs/uploads/      -> list file của user
    - POST /api/docs/uploads/      -> upload + ingest
    - GET  /api/docs/uploads/{id}/ -> detail
    - DELETE /api/docs/uploads/{id}/
    - POST /api/docs/uploads/{id}/reingest/ -> ingest lại
    """
    serializer_class = UploadedDocumentSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UploadedDocument.objects.filter(
            user=self.request.user
        ).order_by("-created_at")

    def perform_create(self, serializer):
        file = self.request.FILES["file"]
        doc = serializer.save(
            user=self.request.user,
            original_name=file.name,
            status="pending",
        )

        try:
            doc.status = "processing"
            doc.save(update_fields=["status"])

            # chạy ingest_main (hiện tại ingest quét hết core/data)
            # ingest_main()

            doc.status = "done"
            doc.processed_at = timezone.now()
            doc.save(update_fields=["status", "processed_at"])
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            doc.save(update_fields=["status", "error_message"])
            # cho nổ ra 500 để FE biết fail
            raise

    @action(detail=True, methods=["POST"])
    def reingest(self, request, pk=None):
        doc = self.get_object()
        try:
            doc.status = "processing"
            doc.save(update_fields=["status"])

            ingest_main()

            doc.status = "done"
            doc.processed_at = timezone.now()
            doc.save(update_fields=["status", "processed_at"])
            return Response({"detail": "Re-ingest thành công"})
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            doc.save(update_fields=["status", "error_message"])
            return Response(
                {"detail": "Re-ingest thất bại", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
