"""知识库：文档状态、MinIO 对象键片段与上传字段长度（与 ``app.services.knowledge_svc`` / 表字段对齐）。"""

# ``knowledge_document.status``
DOCUMENT_STATUS_PENDING: str = "pending"
DOCUMENT_STATUS_PROCESSING: str = "processing"
DOCUMENT_STATUS_INDEXED: str = "indexed"
DOCUMENT_STATUS_FAILED: str = "failed"

# ``knowledge_task.task_type`` / ``status``
TASK_TYPE_INGEST: str = "ingest"
TASK_STATUS_QUEUED: str = "queued"
TASK_STATUS_RUNNING: str = "running"
TASK_STATUS_SUCCEEDED: str = "succeeded"
TASK_STATUS_FAILED: str = "failed"

# 未配置 ``knowledge_base.minio_prefix`` 时的默认前缀（``{kb_id}`` 为知识库 id）
MINIO_DEFAULT_PREFIX_TEMPLATE: str = "kb/{kb_id}"

# 对象键中固定段：``{prefix}/docs/{uuid}/{filename}``
MINIO_OBJECT_PATH_SEGMENT: str = "docs"

# 与 ``KnowledgeDocument.mime`` 列长（128）一致
MAX_KNOWLEDGE_UPLOAD_MIME_LENGTH: int = 128

# 上传文件名安全化上限（与 ``safe_upload_filename`` 调用处一致）
MAX_KNOWLEDGE_UPLOAD_FILENAME_LENGTH: int = 480

# 单文件上传大小上限（字节，约 50MiB）；不设环境变量，与业务校验一致
KNOWLEDGE_UPLOAD_MAX_BYTES: int = 52_428_800


def default_milvus_collection_name(kb_id: int) -> str:
    """未手动指定 ``milvus_collection`` 时使用的集合名：按知识库 id 隔离（Milvus 合法标识）。"""
    return f"kb_{int(kb_id)}_vectors"


# 混合检索：两路召回上限（与 ``limit`` 相乘后封顶）
HYBRID_RECALL_CAP: int = 100

# RRF（倒数排名融合）常数 k，越大则排名尾部权重越小
HYBRID_RRF_K: int = 60

# 混合检索：关键词子路 RRF 贡献乘数（>1 时更信任关键词排序）
HYBRID_KEYWORD_RRF_WEIGHT: float = 1.25

# 混合检索：同一文档最多保留的分片条数（减轻「同文档多片段刷屏」）
HYBRID_MAX_CHUNKS_PER_DOC: int = 2

# 关键词正则召回后参与排序的最大条数（再大则截断，避免单次扫描过多）
KEYWORD_SEARCH_SORT_POOL_CAP: int = 500
