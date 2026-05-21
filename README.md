# AgentForge

基于 **LangChain** 与 **LangGraph** 的全栈 AI Agent 平台，以 **Harness 编程模式** 展示如何设计通用 Agent 系统。适合学习 Agent 架构设计与多 Agent 编排。

---

## Harness 编程模式

Harness 是一套 **AI 辅助软件交付框架**——通过**闭环流程**、**全局宪法**、**技能系统**和**钩子机制**，将 Agent 的行为约束在精确定义的流程轨道上。

**核心理念**：规范先行、测试驱动、证据链完整、闭环强制——AI 不是自由发挥，而是高效执行。

| 传统 AI 辅助开发的痛点 | Harness 的解决方式 |
|------------------------|--------------------|
| AI 随意修改代码，无规范约束 | **闭环强制**：所有文件变更必须走对应闭环流程 |
| 需求→代码无追溯链 | **SDD 四阶段追溯**：requirements → specs → design → tasks → code → tests |
| 安全问题后知后觉 | **安全基线**：访谈→实现→审计→门禁→发布 |
| AI 自己 review 自己的代码 | **生成/评估分离**：独立上下文的 Agent Persona 执行 review |
| 流程改进无反馈通道 | **治理**: 分析执行数据，改进反馈到所有闭环 |

---

## 核心设计

### 四种 Agent 策略

| 策略 | 说明 |
|---|---|
| **Simple Chat** | 最简直通，不挂工具 |
| **ReAct** | 模型 ↔ 工具节点循环，带输入过滤 |
| **Plan & Execute** | 两阶段：先规划，再迭代执行 |
| **Workbench** | 多 Agent 编排器——规划、分发子 Agent（串行/并行）、重规划、扇入聚合 |

### 像素工作室（Workbench）

平台的核心交互入口。每个**工作区**是一个独立的命名空间，内含一个系统级编排 Agent 和若干子 Agent：

- **工作区管理** — 创建、搜索、进入工作区；命名空间隔离，互不干扰
- **对话工作台** — 与编排 Agent 对话，自动拆解任务并调度子 Agent；支持流式输出、思考过程展示、Markdown 渲染
- **配置控制台** — 可视化编辑编排参数：模型选择、温度/TopP/MaxTokens、工具挂载、系统提示词、知识库绑定
- **子 Agent 编排** — 在工作区内创建/管理子 Agent，每个子 Agent 独立配置策略、工具与知识库

### 知识库（RAG）

- 按 Agent 绑定知识库，支持**关键词**、**向量**、**混合**三种检索模式
- **MongoDB**（分片存储）+ **Milvus**（向量索引）+ **MinIO**（文件存储）
- 回答中自动标注来源

### MCP 工具集成

- 通过 [Model Context Protocol](https://modelcontextprotocol.io/) 动态加载外部工具
- 占位符预注册，调用时才建立连接

### 可观测性

- 分布式链路追踪，`X-Request-Id` 全链路透传
- 每轮 Token 用量统计与 Trace 持久化

---

## 技术栈

| 层级 | 技术 |
|---|---|
| **前端** | React 19、TypeScript、Vite、Ant Design、Zustand、Tailwind CSS |
| **后端** | Python、FastAPI、LangChain、LangGraph、SQLAlchemy |
| **基础设施** | MySQL、MongoDB、Redis、Milvus、MinIO |

---

## 快速启动

### 方式一：Docker Compose（推荐）

```bash
docker compose -f docker-compose.dev.yml up -d
```

一键拉起全部中间件 + 后端 + 前端，打开 http://127.0.0.1:5173 即可使用。


### 方式二：本地开发

**前置条件**：Python 3.11+ & [uv](https://docs.astral.sh/uv/)、Node.js 18+、MySQL、MongoDB

```bash
# 后端
cd backend
uv sync
cp .env.example .env    # 按实际环境填写
uv run alembic upgrade head
uv run fastapi dev --host 0.0.0.0 --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev
```

- Swagger 文档：http://127.0.0.1:8000/docs
- 前端页面：http://127.0.0.1:5173

---

## 许可证

[Apache-2.0](./LICENSE)
