"""知识库领域异常（适配器层仅抛出 / 透传，不在此定义业务语义）。"""


class DocumentTextExtractionError(RuntimeError):
    """原文抽取失败或格式不支持。"""


class EmbeddingAdapterError(RuntimeError):
    """嵌入端点 / 凭据 / 模型配置不合法。"""


class ChunkSplitFailedError(RuntimeError):
    """分片失败且已无法降级到字符窗策略。"""


class VectorDimensionMismatchError(RuntimeError):
    """查询向量维度与集合 / 索引期望维度不一致。"""
