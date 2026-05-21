"""工作区命名空间 HTTP 契约。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WorkspaceNamespaceOut(BaseModel):
    id: int = Field(description="命名空间主键")
    slug: str = Field(description="与 API / URL 中 workspace_namespace 字符串一致")
    workbench_agent_id: int | None = Field(
        default=None,
        description="该命名空间唯一的工作台 Agent id；未创建工作台时为 null",
    )
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceNamespaceListData(BaseModel):
    items: list[WorkspaceNamespaceOut]
