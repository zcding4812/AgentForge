"""LLM 提供商（Provider）与挂载模型 HTTP 路由；挂载前缀见 `app.api`（`/api/providers`）。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.constants import PROVIDER_API_MAX_PAGE_SIZE
from app.core.deps import DBManagerDep
from app.schemas.providers import (
    LlmModelCreate,
    LlmModelUpdate,
    ModelProviderCreate,
    ModelProviderUpdate,
    paged_models,
    paged_providers,
)
from app.schemas.response import ApiResponse
from app.services.provider_svc import ProviderService


def get_provider_service(db_manager: DBManagerDep) -> ProviderService:
    return ProviderService(db_manager)


ProviderServiceDep = Annotated[ProviderService, Depends(get_provider_service)]


router = APIRouter()


@router.get(
    "/models",
    response_model=ApiResponse[dict],
    summary="分页列出模型配置（sys_model：含 llm/embedding/tts 等）",
    operation_id="list_models",
)
async def list_models(
    providers: ProviderServiceDep,
    q: Annotated[str | None, Query()] = None,
    provider_id: Annotated[
        str | None,
        Query(description="提供商 provider_code（或数字主键），筛选该提供商下模型"),
    ] = None,
    model_type: Annotated[
        str | None, Query(description="chat/embedding/ocr 或原始 model_type")
    ] = None,
    status: Annotated[
        str,
        Query(description="enabled（默认）/ disabled / all / error"),
    ] = "enabled",
    provider_enabled_only: Annotated[
        bool,
        Query(
            description="为 true 时仅返回「提供商已启用」(status=1) 下的模型；"
            "管理台需查看停用时传 false"
        ),
    ] = True,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=PROVIDER_API_MAX_PAGE_SIZE)] = 10,
) -> ApiResponse[dict]:
    items, total = await providers.list_models(
        q=q,
        provider_id=provider_id,
        model_type=model_type,
        status=status,
        provider_enabled_only=provider_enabled_only,
        page=page,
        page_size=page_size,
    )
    data = paged_models(items, page, page_size, total)
    return ApiResponse(message="ok", data=data.model_dump())


@router.post(
    "/models",
    response_model=ApiResponse[dict],
    summary="创建模型配置",
    operation_id="create_model",
)
async def create_model(providers: ProviderServiceDep, body: LlmModelCreate) -> ApiResponse[dict]:
    try:
        out = await providers.create_model(body)
        return ApiResponse(message="ok", data=out.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch(
    "/models/{model_id}",
    response_model=ApiResponse[dict],
    summary="更新模型配置",
    operation_id="update_model",
)
async def update_model(
    providers: ProviderServiceDep, model_id: str, body: LlmModelUpdate
) -> ApiResponse[dict]:
    try:
        out = await providers.update_model(model_id, body)
        return ApiResponse(message="ok", data=out.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete(
    "/models/{model_id}",
    status_code=204,
    summary="删除模型配置",
    operation_id="delete_model",
)
async def delete_model(providers: ProviderServiceDep, model_id: str) -> None:
    try:
        await providers.delete_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在") from None


@router.post(
    "/models/{model_id}/probe",
    response_model=ApiResponse[dict],
    summary="探测模型连通性",
    operation_id="probe_model",
)
async def probe_model(providers: ProviderServiceDep, model_id: str) -> ApiResponse[dict]:
    try:
        res = await providers.probe_model(model_id)
        return ApiResponse(message="ok", data=res.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="模型不存在") from None


# --- 提供商 CRUD：根路径即列表/创建 ---


@router.get(
    "",
    response_model=ApiResponse[dict],
    summary="分页列出模型供应商",
    operation_id="list_model_providers",
)
async def list_model_providers(
    providers: ProviderServiceDep,
    q: Annotated[str | None, Query(description="名称或 ID 模糊搜索")] = None,
    status: Annotated[str | None, Query(description="enabled / disabled")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=PROVIDER_API_MAX_PAGE_SIZE)] = 10,
) -> ApiResponse[dict]:
    items, total = await providers.list_providers(
        q=q,
        status=status,
        page=page,
        page_size=page_size,
    )
    data = paged_providers(items, page, page_size, total)
    return ApiResponse(message="ok", data=data.model_dump())


@router.post(
    "",
    response_model=ApiResponse[dict],
    summary="创建模型供应商",
    operation_id="create_model_provider",
)
async def create_model_provider(
    providers: ProviderServiceDep, body: ModelProviderCreate
) -> ApiResponse[dict]:
    try:
        out = await providers.create_provider(body)
        return ApiResponse(message="ok", data=out.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.patch(
    "/{provider_id}",
    response_model=ApiResponse[dict],
    summary="更新模型供应商",
    operation_id="update_model_provider",
)
async def update_model_provider(
    providers: ProviderServiceDep, provider_id: str, body: ModelProviderUpdate
) -> ApiResponse[dict]:
    try:
        out = await providers.update_provider(provider_id, body)
        return ApiResponse(message="ok", data=out.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="供应商不存在") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete(
    "/{provider_id}",
    status_code=204,
    summary="删除模型供应商",
    operation_id="delete_model_provider",
)
async def delete_model_provider(providers: ProviderServiceDep, provider_id: str) -> None:
    try:
        await providers.delete_provider(provider_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="供应商不存在") from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/{provider_id}/probe",
    response_model=ApiResponse[dict],
    summary="测试供应商连通性（请求 Base URL /models）",
    operation_id="probe_model_provider",
)
async def probe_model_provider(
    providers: ProviderServiceDep, provider_id: str
) -> ApiResponse[dict]:
    try:
        res = await providers.probe_provider(provider_id)
        return ApiResponse(message="ok", data=res.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="供应商不存在") from None
