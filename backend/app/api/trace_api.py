from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.core.constants import (
    TRACE_LIST_DEFAULT_PAGE_SIZE,
    TRACE_LIST_MAX_PAGE_SIZE,
)
from app.schemas.response import ApiResponse
from app.schemas.trace import TraceDetailData, TraceListData
from app.services.trace_svc import TraceService


def get_trace_service() -> TraceService:
    return TraceService()


TraceServiceDep = Annotated[TraceService, Depends(get_trace_service)]


router = APIRouter()


@router.get(
    "",
    response_model=ApiResponse[TraceListData],
    summary="获取链路列表",
    operation_id="get_trace_list",
)
async def get_trace_list(
    request: Request,
    traces: TraceServiceDep,
    trace_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(
        default=TRACE_LIST_DEFAULT_PAGE_SIZE,
        ge=1,
        le=TRACE_LIST_MAX_PAGE_SIZE,
        description="每页条数，默认 10",
    ),
) -> ApiResponse[TraceListData]:
    data = await traces.get_trace_list(
        trace_id,
        page=page,
        page_size=page_size,
    )
    return ApiResponse(message="ok", data=data)


@router.get(
    "/{trace_id}",
    response_model=ApiResponse[TraceDetailData],
    summary="获取单条链路详情",
    operation_id="get_trace_detail",
)
async def get_trace_detail(
    request: Request,
    traces: TraceServiceDep,
    trace_id: str,
) -> ApiResponse[TraceDetailData]:
    data = await traces.get_trace_detail(trace_id)
    return ApiResponse(message="ok", data=data)
