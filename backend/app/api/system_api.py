from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.core.deps import DBManagerDep, SettingsDep
from app.core.tracing import trace_span
from app.schemas.response import ApiResponse
from app.schemas.system import HealthData, MonitorDashboardData
from app.services.monitor_svc import MonitorService
from app.services.system_svc import SystemService


def get_system_service(db_manager: DBManagerDep) -> SystemService:
    return SystemService(db_manager)


SystemServiceDep = Annotated[SystemService, Depends(get_system_service)]


def get_monitor_service(
    system: SystemServiceDep,
    db_manager: DBManagerDep,
) -> MonitorService:
    return MonitorService(system, db_manager)


MonitorServiceDep = Annotated[MonitorService, Depends(get_monitor_service)]


router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=ApiResponse[HealthData],
    summary="健康检查",
    operation_id="get_health",
)
async def health(request: Request, system: SystemServiceDep) -> ApiResponse[HealthData]:
    data = await system.get_health_data()
    return ApiResponse(message=data.status, data=data)


@router.get(
    "/api/system/monitor-dashboard",
    response_model=ApiResponse[MonitorDashboardData],
    summary="监控看板",
    operation_id="get_monitor_dashboard",
)
@trace_span(
    span_type="api",
    component="system-api",
    trace_id_arg="request",
)
async def monitor_dashboard(
    request: Request,
    monitor: MonitorServiceDep,
    settings: SettingsDep,
    token_days: int = Query(
        default=14,
        ge=1,
        le=90,
        description="Token 统计区间天数（含今日，最大 90）",
    ),
) -> ApiResponse[MonitorDashboardData]:
    data = await monitor.get_dashboard(request, token_days=token_days, settings=settings)
    return ApiResponse(message="ok", data=data)
