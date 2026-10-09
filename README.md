# AML Memory：面向 Agent Memory Challenge 的轻量记忆服务

本仓库是 **Agent Memory Challenge（第二期）** 开源方法榜的参赛系统实现。

系统只提供比赛要求的两类记忆操作，不参与答案生成：

| 接口 | 作用 |
| --- | --- |
| `POST /add` | 接收评测记忆，完成存储与索引，返回成功时保证已持久化且可立即检索 |
| `POST /search` | 按问题返回相关记忆证据（按相关性排序），**不生成最终答案** |
| `GET /health` | 健康检查，无需鉴权 |

答案生成与打分由评测平台统一完成，本系统不参与。

---

## 实现概览

- **语言 / 依赖**：Python 3.8+，**仅使用标准库**（`http.server`、`sqlite3`、`json`、`hashlib` 等），无需安装任何第三方包
- **存储**：单机 SQLite（WAL 模式），按 `user_id` 强制隔离
- **检索**：BM25 关键词检索 + 中文/日文/韩文**字符 bigram** 分词 + 时间信息加权
- **写入**：同步写入，返回 HTTP 200 前完成落库并立即可查；相同 `request_id` 幂等去重
- **多模态**：`ContentPart[]`（text / image_url）原样透传与返回
- **模型使用**：**本实现不调用任何生成式大模型**（不涉及平台对 Add/Search 的模型限制）

---

## 目录结构

```
.
├── memory_server.py     # 服务主体：Add / Search / Health 三个接口
├── smoke_test.py        # 本地契约自测（覆盖 24 项契约检查）
├── requirements.txt     # 无第三方依赖
├── docs/
│   ├── method.md        # 方法说明（技术方案、选择理由、来源与原创性声明）
│   └── operations.md    # 运行说明（部署、配置、容量、超时、限流）
└── README.md
```

---

## 快速开始

```bash
# 1. 启动服务（无需安装任何依赖）
MEMORY_API_KEY=your-secret-key PORT=8080 python3 memory_server.py

# 2. 另开一个终端，运行契约自测
python3 smoke_test.py --base-url http://127.0.0.1:8080 --key your-secret-key
```

预期输出：

```
PASSED 24   FAILED 0
```

浏览器打开 `http://127.0.0.1:8080/` 可看到一个状态页；`http://127.0.0.1:8080/health` 返回 `{"status": "ok"}`。

---

## 接口契约

严格遵循 Agent Memory Leaderboard 现行规范。

### POST /add

请求：

```json
{
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "messages": [
    { "role": "user", "timestamp": 1704067200000, "content": "memory text" }
  ],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

响应：

```json
{
  "success": true,
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

### POST /search

请求：

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "top_k": 100
}
```

响应：

```json
{
  "data": [
    { "id": "mem_123", "content": "remembered fact text", "score": 0.87, "created_at": "2026-07-01T12:00:00Z" }
  ]
}
```

返回条数**不会超过 `top_k`**；无结果时返回 `{"data": []}`。

### 鉴权

支持 `Authorization: Bearer <key>`、`Authorization: Token <key>`、`X-Api-Key: <key>` 三种方式。
`/health` 无需鉴权；鉴权失败返回 `401`。

---

## 方法来源与原创性声明

- 本仓库代码为本次参赛**原创实现**，未复制、封装或改写任何第三方项目、论文代码或开源仓库。
- 检索算法采用公开的经典方法：**BM25**（Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009）与**字符 n-gram（bigram）分词**（中/日/韩文检索的常规做法）。
- **未使用任何生成式大模型**，因此不涉及平台关于 Add / Search 阶段模型使用的限制（`gpt-4o-mini`）。
- 未使用任何评测数据训练、微调或分析。

---

## 许可

MIT License，详见 `LICENSE`。
