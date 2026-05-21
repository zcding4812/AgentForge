"""门面仅从 ``app.knowledge.facades`` 导出。"""

from app.knowledge import EmbeddingAdapter, KnowledgeRetrievalFacade
from app.knowledge.facades import EmbeddingAdapter as EAAgain
from app.knowledge.facades import KnowledgeRetrievalFacade as FacadeAgain


def test_single_export_path() -> None:
    assert KnowledgeRetrievalFacade is FacadeAgain
    assert EmbeddingAdapter is EAAgain
