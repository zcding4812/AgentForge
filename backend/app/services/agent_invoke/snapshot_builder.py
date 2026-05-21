"""从请求体或系统模型构造 ``ModelConfigSnapshot`` 及合并覆盖。"""

from __future__ import annotations

from dataclasses import replace

from app.agent.kernel import (
    InferenceHyperparameters,
    InputContentFilterConfig,
    ModelConfigSnapshot,
    ModelIdentity,
    ResponseConstraints,
    ToolChoicePolicy,
)
from app.domain.providers import provider_has_credentials
from app.infrastructure.db import SQLAlchemyDatabaseManager
from app.repositories.provider_repo import ProviderRepository
from app.schemas.agent import AgentInvokeRequest


class InvokeSnapshotBuilder:
    """模型配置快照：``from_request_body`` / ``merge_request`` 无状态；``from_sys_model`` 依赖注入 DB。"""

    def __init__(self, db_manager: SQLAlchemyDatabaseManager | None = None) -> None:
        self._db_manager = db_manager

    @staticmethod
    def from_request_body(body: AgentInvokeRequest) -> ModelConfigSnapshot:
        mid = body.model_identity
        hp = body.hyperparameters
        rc = body.response
        tc = body.tool_choice
        return ModelConfigSnapshot(
            config_id=body.config_id,
            identity=ModelIdentity(
                provider=mid.provider,
                model_name=mid.model_name,
                deployment=mid.deployment,
                base_url_override=mid.base_url_override,
                api_key=mid.api_key,
            ),
            hyperparameters=InferenceHyperparameters(
                temperature=hp.temperature,
                max_tokens=hp.max_tokens,
                top_p=hp.top_p,
                top_k=hp.top_k,
                frequency_penalty=hp.frequency_penalty,
                presence_penalty=hp.presence_penalty,
                stop=hp.stop,
                seed=hp.seed,
            ),
            response=ResponseConstraints(
                response_format=rc.response_format,
                json_schema_id=rc.json_schema_id,
                response_json_schema=rc.response_json_schema,
                parallel_tool_calls=rc.parallel_tool_calls,
            ),
            tool_choice=ToolChoicePolicy(mode=tc.mode, forced_tool_name=tc.forced_tool_name)
            if tc
            else None,
            input_content_filter=InputContentFilterConfig(),
        )

    async def from_sys_model(self, config_id: str | int) -> ModelConfigSnapshot:
        if self._db_manager is None:
            raise ValueError("使用 config_id 时数据库不可用")
        key = str(config_id).strip()
        if not key.isdigit():
            raise ValueError("config_id 须为系统模型数字主键（sys_model.id）")
        pk = int(key)
        row = await ProviderRepository.get_model_with_provider(pk, db_manager=self._db_manager)
        if not row:
            raise LookupError("系统模型不存在")
        m, p = row
        if m.is_enabled != 1:
            raise ValueError("系统模型未启用")
        if not provider_has_credentials(
            auth_type=p.auth_type,
            api_key=p.api_key,
            api_secret=p.api_secret,
        ):
            raise ValueError("所属提供商未配置有效凭据，无法调用")
        base_url = (m.endpoint or "").strip() or (p.base_url or "").strip() or None
        return ModelConfigSnapshot(
            config_id=str(m.id),
            identity=ModelIdentity(
                provider=(p.api_format or "openai").strip(),
                model_name=m.model_code.strip(),
                deployment=None,
                base_url_override=base_url,
                api_key=p.api_key,
            ),
            hyperparameters=InferenceHyperparameters(),
            response=ResponseConstraints(),
            tool_choice=None,
            input_content_filter=InputContentFilterConfig(),
        )

    @staticmethod
    def merge_request(base: ModelConfigSnapshot, body: AgentInvokeRequest) -> ModelConfigSnapshot:
        hp = body.hyperparameters
        rc = body.response
        tc = body.tool_choice
        identity = base.identity
        mid = body.model_identity
        if mid.api_key and str(mid.api_key).strip():
            identity = replace(identity, api_key=str(mid.api_key).strip())
        if mid.base_url_override and str(mid.base_url_override).strip():
            identity = replace(identity, base_url_override=str(mid.base_url_override).strip())

        return ModelConfigSnapshot(
            config_id=base.config_id,
            identity=identity,
            hyperparameters=InferenceHyperparameters(
                temperature=hp.temperature
                if hp.temperature is not None
                else base.hyperparameters.temperature,
                max_tokens=hp.max_tokens
                if hp.max_tokens is not None
                else base.hyperparameters.max_tokens,
                top_p=hp.top_p if hp.top_p is not None else base.hyperparameters.top_p,
                top_k=hp.top_k if hp.top_k is not None else base.hyperparameters.top_k,
                frequency_penalty=hp.frequency_penalty
                if hp.frequency_penalty is not None
                else base.hyperparameters.frequency_penalty,
                presence_penalty=hp.presence_penalty
                if hp.presence_penalty is not None
                else base.hyperparameters.presence_penalty,
                stop=hp.stop if hp.stop is not None else base.hyperparameters.stop,
                seed=hp.seed if hp.seed is not None else base.hyperparameters.seed,
            ),
            response=ResponseConstraints(
                response_format=rc.response_format,
                json_schema_id=rc.json_schema_id,
                response_json_schema=rc.response_json_schema,
                parallel_tool_calls=rc.parallel_tool_calls,
            ),
            tool_choice=ToolChoicePolicy(mode=tc.mode, forced_tool_name=tc.forced_tool_name)
            if tc
            else base.tool_choice,
            input_content_filter=base.input_content_filter,
        )
