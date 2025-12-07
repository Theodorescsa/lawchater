from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
# from core.app import get_rag_service
from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter, OpenApiResponse
import logging
# rag_service = get_rag_service()

logger = logging.getLogger(__name__)
@extend_schema(
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "k": {"type": "integer", "default": 3}
            },
            "required": ["question"]
        }
    },
    responses={
        200: {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}}
            }
        },
        400: OpenApiResponse(description="Missing question"),
        500: OpenApiResponse(description="Internal error")
    },
    examples=[
        OpenApiExample(
            "Example request",
            value={"question": "Tôi ly hôn thì mất bao nhiêu tiền?", "k": 3},
        )
    ],
)
@api_view(['POST'])
def chat(request):
    return Response({
        "question": "Tôi ly hôn thì mất bao nhiêu tiền?",
        "answer": "… câu trả lời dài …",
        "sources": [
            {
            "content": "Điều 68. Tuyên bố mất tích…",
            "metadata": {
                "source": "/app/core/data/BLDS.docx"
            }
            },
            {
            "content": "Điều 68. Tuyên bố mất tích…",
            "metadata": {
                "source": "/app/core/data/BLDS.docx"
            }
            }
        ]
        }
    )
    # """
    # API endpoint để chat với RAG system
    
    # Request body:
    # {
    #     "question": "Câu hỏi của bạn",
    #     "k": 3  // Tùy chọn: số lượng documents liên quan
    # }
    
    # Response:
    # {
    #     "answer": "Câu trả lời",
    #     "sources": [...],
    #     "question": "Câu hỏi gốc"
    # }
    # """
    # try:
    #     question = request.data.get('question', '').strip()
    #     k = request.data.get('k', 3)
        
    #     if not question:
    #         return Response(
    #             {'error': 'Vui lòng cung cấp câu hỏi'},
    #             status=status.HTTP_400_BAD_REQUEST
    #         )
        
    #     # Gọi RAG service
    #     result = rag_service.query(question=question, k=k)
        
    #     # Kiểm tra lỗi
    #     if 'error' in result and result.get('answer') == 'Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau.':
    #         return Response(
    #             {
    #                 'error': result['error'],
    #                 'answer': result['answer']
    #             },
    #             status=status.HTTP_500_INTERNAL_SERVER_ERROR
    #         )
        
    #     return Response({
    #         'question': question,
    #         'answer': result['answer'],
    #         'sources': result['sources'],
    #     })
        
    # except Exception as e:
    #     return Response(
    #         {'error': str(e)},
    #         status=status.HTTP_500_INTERNAL_SERVER_ERROR
    #     )

@extend_schema(
    responses={
        200: {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "service": {"type": "string"},
                "llm": {"type": "string"},
                "vectorstore": {"type": "string"},
            }
        }
    }
)
@api_view(['GET'])
def health(request):
    """Health check endpoint"""
    return Response({
        'status': 'ok',
        'service': 'LawChat API',
        'llm': 'connected',
        'vectorstore': 'ready'
    })


@api_view(['POST'])
def ingest(request):
    """
    Endpoint để trigger việc ingest lại dữ liệu
    Cần implement thêm logic bảo mật
    """
    try:
        # Import ingest logic
        from core.ingest import main as ingest_main
        
        # Chạy ingest
        ingest_main()
        
        return Response({
            'status': 'success',
            'message': 'Đã ingest dữ liệu thành công'
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
@api_view(['GET'])
def debug_vectorstore(request):
    """Debug endpoint để kiểm tra vectorstore"""
    try:
        # Lấy thông tin về collection
        collection = rag_service.vectorstore._collection
        count = collection.count()
        
        # Thử search một query đơn giản
        test_results = rag_service.vectorstore.similarity_search("an ninh mạng", k=2)
        
        return Response({
            'status': 'ok',
            'collection_name': rag_service.vectorstore._collection.name,
            'document_count': count,
            'test_search_results': len(test_results),
            'sample_docs': [
                {
                    'content': doc.page_content[:200] + '...',
                    'metadata': doc.metadata
                } for doc in test_results
            ] if test_results else []
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def ingest(request):
    """
    Endpoint để trigger việc ingest lại dữ liệu
    Cần implement thêm logic bảo mật
    """
    try:
        # Import ingest logic
        from core.ingest import main as ingest_main
        
        # Chạy ingest
        ingest_main()
        
        return Response({
            'status': 'success',
            'message': 'Đã ingest dữ liệu thành công'
        })
        
    except Exception as e:
        logger.error(f"Error in ingest: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )