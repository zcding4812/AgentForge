"""工作区命名空间：独立资源表，与 Agent / 知识库通过 FK 关联。"""

from fastapi import APIRouter

from app.core.deps import DBManagerDep
from app.repositories.namespace_repo import WorkspaceNamespaceRepository
from app.schemas.response import ApiResponse
from app.schemas.workspace import WorkspaceNamespaceListData, WorkspaceNamespaceOut

router = APIRouter()


@router.get(
    "",
    response_model=ApiResponse[WorkspaceNamespaceListData],
    summary="命名空间列表",
    operation_id="get_workspace_namespaces_list",
)
async def list_workspace_namespaces(
    db_manager: DBManagerDep,
) -> ApiResponse[WorkspaceNamespaceListData]:
    rows = await WorkspaceNamespaceRepository.list_all_by_slug_order(db_manager=db_manager)
    items = [WorkspaceNamespaceOut.model_validate(r) for r in rows]
    return ApiResponse(message="ok", data=WorkspaceNamespaceListData(items=items))
