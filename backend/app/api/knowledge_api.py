"""知识库 HTTP 路由：元数据 CRUD、文档列表与上传。"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.deps import DBManagerDep, MinioObjectStoreDep
from app.schemas.knowledge import (
    KnowledgeBaseOut,
    KnowledgeCreateBody,
    KnowledgeDocumentChunksData,
    KnowledgeDocumentIngestAccepted,
    KnowledgeDocumentListData,
    KnowledgeDocumentOut,
    KnowledgeDocumentPatchBody,
    KnowledgeListData,
    KnowledgeSearchBody,
    KnowledgeSearchData,
    KnowledgeUpdateBody,
    KnowledgeVectorBuildData,
)
from app.schemas.response import ApiResponse
from app.services.knowledge_svc import (
    ChunkRepositoryDep,
    KnowledgeBaseService,
    KnowledgeDocumentService,
)


def get_knowledge_base_service(db_manager: DBManagerDep) -> KnowledgeBaseService:
    return KnowledgeBaseService(db_manager)


def get_knowledge_document_service(
    db_manager: DBManagerDep,
    minio_store: MinioObjectStoreDep,
    chunk_repository: ChunkRepositoryDep,
) -> KnowledgeDocumentService:
    return KnowledgeDocumentService(db_manager, minio_store, chunk_repository)


KnowledgeBaseServiceDep = Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)]
KnowledgeDocumentServiceDep = Annotated[
    KnowledgeDocumentService,
    Depends(get_knowledge_document_service),
]

router = APIRouter()


@router.get(
    "",
    response_model=ApiResponse[KnowledgeListData],
    summary="知识库列表（分页）",
    operation_id="get_knowledge_bases_list",
)
async def list_knowledge_bases(
    svc: KnowledgeBaseServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    q: Annotated[str | None, Query(description="名称或 slug 模糊搜索")] = None,
    status: Annotated[str | None, Query(description="生命周期状态筛选")] = None,
) -> ApiResponse[KnowledgeListData]:
    data = await svc.list_knowledge_bases(page=page, page_size=page_size, q=q, status=status)
    return ApiResponse(message="ok", data=data)


@router.post(
    "",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="创建知识库",
    operation_id="post_knowledge_bases_create",
)
async def create_knowledge_base_route(
    body: KnowledgeCreateBody,
    svc: KnowledgeBaseServiceDep,
) -> ApiResponse[KnowledgeBaseOut]:
    try:
        out = await svc.create_knowledge_base(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=out)


@router.post(
    "/{kb_id}/vectors/build",
    response_model=ApiResponse[KnowledgeVectorBuildData],
    summary="从 Mongo 已有分片构建向量并写入 Milvus（不重跑 ingest；需 Milvus 与 vector/hybrid + embedding）",
    operation_id="post_knowledge_vectors_build",
)
async def build_knowledge_vectors_route(
    kb_id: int,
    svc: KnowledgeDocumentServiceDep,
) -> ApiResponse[KnowledgeVectorBuildData]:
    try:
        data = await svc.build_vectors_from_mongo_chunks(kb_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/{kb_id}/search",
    response_model=ApiResponse[KnowledgeSearchData],
    summary="知识库检索（可选 retrieval_override / recall_limit / rrf_k；向量/混合在 Milvus 未接入时降级为关键词）",
    operation_id="post_knowledge_kb_search",
)
async def search_knowledge_route(
    kb_id: int,
    body: KnowledgeSearchBody,
    svc: KnowledgeDocumentServiceDep,
) -> ApiResponse[KnowledgeSearchData]:
    try:
        data = await svc.search_in_kb(
            kb_id,
            q=body.q,
            limit=body.limit,
            doc_id=body.doc_id,
            retrieval_override=body.retrieval_override,
            recall_limit=body.recall_limit,
            rrf_k=body.rrf_k,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.get(
    "/{kb_id}/documents",
    response_model=ApiResponse[KnowledgeDocumentListData],
    summary="知识库文档列表（分页）",
    operation_id="get_knowledge_documents_list",
)
async def list_knowledge_documents(
    kb_id: int,
    svc: KnowledgeDocumentServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    q: Annotated[str | None, Query(description="文件名模糊搜索")] = None,
    chunk_left_panel: Annotated[
        bool,
        Query(
            description=(
                "为 true 时返回切块步骤左侧「待配置」列表：排除「已 indexed 且已设文档级 chunk_method」；"
                "恢复默认后仅 indexed 且无覆盖的文档会重新出现在此列表"
            ),
        ),
    ] = False,
) -> ApiResponse[KnowledgeDocumentListData]:
    try:
        data = await svc.list_documents(
            kb_id,
            page=page,
            page_size=page_size,
            q=q,
            chunk_left_panel=chunk_left_panel,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/{kb_id}/documents/upload",
    response_model=ApiResponse[KnowledgeDocumentIngestAccepted],
    summary="上传文档（本地落盘；切块/索引由后续任务处理）",
    operation_id="post_knowledge_document_upload",
    responses={
        202: {
            "description": "已接受异步索引任务（与 200 相同响应体，便于客户端按状态码区分）",
            "model": ApiResponse[KnowledgeDocumentIngestAccepted],
        },
    },
)
async def upload_knowledge_document_route(
    kb_id: int,
    svc: KnowledgeDocumentServiceDep,
    file: Annotated[UploadFile, File(description="待上传文件")],
    async_: Annotated[
        bool,
        Query(
            alias="async",
            description="为 true 时返回 HTTP 202（仍返回 document + task_id）",
        ),
    ] = False,
) -> ApiResponse[KnowledgeDocumentIngestAccepted] | JSONResponse:
    raw = await file.read()
    try:
        accepted = await svc.upload_document(
            kb_id,
            original_filename=file.filename or "",
            content_type=file.content_type,
            data=raw,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    body = ApiResponse(message="accepted" if async_ else "ok", data=accepted)
    if async_:
        return JSONResponse(status_code=202, content=body.model_dump(mode="json"))
    return body


@router.patch(
    "/{kb_id}/documents/{doc_id}",
    response_model=ApiResponse[KnowledgeDocumentOut],
    summary="更新文档（文档级切块策略覆盖；chunk_method 置 null 表示恢复知识库默认）",
    operation_id="patch_knowledge_document_by_id",
)
async def patch_knowledge_document_route(
    kb_id: int,
    doc_id: int,
    body: KnowledgeDocumentPatchBody,
    svc: KnowledgeDocumentServiceDep,
) -> ApiResponse[KnowledgeDocumentOut]:
    try:
        data = await svc.patch_document(kb_id, doc_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.post(
    "/{kb_id}/documents/{doc_id}/ingest",
    response_model=ApiResponse[KnowledgeDocumentIngestAccepted],
    summary="重新触发文档 ingest（删除同版本幂等任务后再次入队）",
    operation_id="post_knowledge_document_reingest",
    responses={
        202: {
            "description": "已接受异步索引任务",
            "model": ApiResponse[KnowledgeDocumentIngestAccepted],
        },
    },
)
async def requeue_knowledge_document_ingest_route(
    kb_id: int,
    doc_id: int,
    svc: KnowledgeDocumentServiceDep,
    async_: Annotated[
        bool,
        Query(
            alias="async",
            description="为 true 时返回 HTTP 202",
        ),
    ] = False,
) -> ApiResponse[KnowledgeDocumentIngestAccepted] | JSONResponse:
    try:
        accepted = await svc.requeue_document_ingest(kb_id, doc_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    body = ApiResponse(message="accepted" if async_ else "ok", data=accepted)
    if async_:
        return JSONResponse(status_code=202, content=body.model_dump(mode="json"))
    return body


@router.delete(
    "/{kb_id}/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档（软删；清理 Mongo 分片并尽力删除 MinIO 对象）",
    operation_id="delete_knowledge_document_by_id",
)
async def delete_knowledge_document_route(
    kb_id: int,
    doc_id: int,
    svc: KnowledgeDocumentServiceDep,
) -> Response:
    try:
        await svc.delete_document(kb_id, doc_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{kb_id}/documents/{doc_id}/chunks",
    response_model=ApiResponse[KnowledgeDocumentChunksData],
    summary="文档分片列表（Mongo，按 content_version 分页）",
    operation_id="get_knowledge_document_chunks",
)
async def list_document_chunks_route(
    kb_id: int,
    doc_id: int,
    svc: KnowledgeDocumentServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    content_version: Annotated[
        str | None,
        Query(description="留空则使用文档当前 content_version"),
    ] = None,
) -> ApiResponse[KnowledgeDocumentChunksData]:
    try:
        data = await svc.list_document_chunks(
            kb_id,
            doc_id,
            content_version=content_version,
            page=page,
            page_size=page_size,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.get(
    "/{kb_id}",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="知识库详情",
    operation_id="get_knowledge_base_by_id",
)
async def get_knowledge_base_route(
    kb_id: int,
    svc: KnowledgeBaseServiceDep,
) -> ApiResponse[KnowledgeBaseOut]:
    try:
        data = await svc.get_knowledge_base(kb_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.patch(
    "/{kb_id}",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="更新知识库元数据与切块策略",
    operation_id="patch_knowledge_base_by_id",
)
async def patch_knowledge_base_route(
    kb_id: int,
    body: KnowledgeUpdateBody,
    svc: KnowledgeBaseServiceDep,
) -> ApiResponse[KnowledgeBaseOut]:
    try:
        data = await svc.update_knowledge_base(kb_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiResponse(message="ok", data=data)


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除知识库（软删；派生存储清理见异步任务）",
    operation_id="delete_knowledge_base_by_id",
)
async def delete_knowledge_base_route(
    kb_id: int,
    svc: KnowledgeBaseServiceDep,
) -> Response:
    try:
        await svc.delete_knowledge_base(kb_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)
