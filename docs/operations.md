# 运行说明

> 系统版本：v0.2 ｜ 环境：Ubuntu 22.04（亦可在任意 Python 3.8+ 环境运行）

---

## 1. 接口清单

| 接口 | 方法 | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `/add` | POST | 需要 | 写入记忆（同步） |
| `/search` | POST | 需要 | 检索记忆，返回排序后的证据 |
| `/health` | GET | **不需要** | 健康检查，返回任意 2xx |
| `/` | GET | 不需要 | 状态页（非契约接口，便于人工查看） |

**服务地址**：`http://<你的服务器公网 IP>:8080`

---

## 2. 依赖与环境

| 项目 | 值 |
| --- | --- |
| 运行时 | Python 3.8 及以上（实测 3.10.12） |
| 第三方依赖 | **无**（仅 Python 标准库） |
| 存储 | 单机 SQLite，数据文件 `memory.sqlite3`（WAL 模式），位于进程工作目录 |
| 操作系统 | Ubuntu 22.04 LTS（x86_64） |

---

## 3. 启动方式

### 3.1 直接启动（前台）

```bash
cd /root
MEMORY_API_KEY='<Memory System Key>' PORT=8080 python3 memory_server.py
```

### 3.2 生产部署（systemd，推荐）

`/etc/systemd/system/aml-memory.service`：

```ini
[Unit]
Description=AML Memory Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/root
Environment=MEMORY_API_KEY=<Memory System Key>
Environment=PORT=8080
ExecStart=/usr/bin/python3 /root/memory_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用：

```bash
chmod 600 /etc/systemd/system/aml-memory.service
systemctl daemon-reload
systemctl enable --now aml-memory
```

该配置提供**开机自启**与**进程崩溃自动重启**（`Restart=always`，间隔 3 秒）。

---

## 4. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MEMORY_API_KEY` | 空 | 接口鉴权密钥。**为空时不校验鉴权**，仅可用于本地调试与公开 smoke；正式评测必须设置 |
| `PORT` | `8080` | 监听端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `MEMORY_DB` | `memory.sqlite3` | SQLite 数据文件路径 |
| `MEMORY_PREFIX_TIME` | `1` | 设为 `1` 时，返回的单条记忆会带上 `[YYYY-MM-DD]` 日期前缀（窗口行自带日期，不重复添加） |
| `MAX_BODY_BYTES` | 64 MiB | 请求体大小上限（高于契约允许的 30 MiB 载荷） |
| `WINDOW_SIZE` | `3` | 相邻消息滑动窗口覆盖的消息条数；设为 `0` 关闭窗口索引 |
| `WINDOW_MIN` | `2` | 至少几条消息才生成一个窗口 |
| `MMR_LAMBDA` | `0.82` | 相关性 vs 去重强度；设为 `1.0` 即关闭去重 |
| `MMR_POOL` | `220` | 参与去重的候选数量 |
| `RRF_K` | `60` | RRF 融合常数 |
| `SPARSE_TOP_N` | `120` | 每路关键词检索召回的候选数 |
| `EMBED_BACKEND` | `none` | `none` = 纯算法（默认）；`auto` / `st` 启用本地向量模型（非生成式，默认关闭） |

---

## 5. 容量、超时与限流

| 项目 | 值 | 说明 |
| --- | --- | --- |
| Add 并发上限 | **8** | 服务为线程模型，单机 2 vCPU 下的保守值；可按平台调度能力调整 |
| Search 并发上限 | **8** | 同上 |
| 单请求超时 | 远低于平台上限 30 分钟 | 实测单次 Add/Search 在百毫秒量级 |
| Search 延迟 | 典型 < 300 ms（公网往返）；服务端处理约 1–20 ms（数百条记忆规模），约 0.7 s（单会话 6000 条记忆） | 检索仅在该 `user_id` 作用域内计算 |
| 主动限流 | 未启用 | 不主动拒绝请求；服务端错误以 5xx 返回，平台按契约退避重试 |
| 磁盘占用 | 每条记忆约 1–2 KB；因窗口索引，行数约为消息数的 1.5–2 倍 | 40 GB 系统盘，足够覆盖全量评测 |
| 内存占用 | 空载约 10–20 MB | 2 GiB 内存有大量余量 |
| 可用性 | 7×24 常驻 | systemd 托管，支持开机自启与崩溃重启；评测期间保持公网可达 |

> 平台侧提示：若需要调整并发上限，请在提交申请时声明；**审核通过后不得再改动 API 契约、鉴权方式与容量设置**。

---

## 6. 运维与排查

```bash
systemctl status aml-memory --no-pager     # 查看状态
systemctl restart aml-memory               # 重启
journalctl -u aml-memory -n 50 --no-pager  # 最近日志
journalctl -u aml-memory -f                # 实时日志（Ctrl+C 退出）

curl -s http://127.0.0.1:8080/health       # 健康检查
ss -lntp | grep 8080                       # 确认端口监听
```

**日志格式**：每行一条结构化记录，包含时间、方法与路径、HTTP 状态码、处理耗时。
系统**不记录请求正文**，避免留存评测数据。

---

## 7. 数据与合规

- 所有记忆按 `user_id` 隔离存储，检索不会跨用户返回
- 不记录请求正文与检索结果正文
- 评测结束后 **30 天内删除全部评测数据**（删除 `memory.sqlite3` 即可）
- 接口 URL 不含任何凭据，且为公网可直接访问的地址（非私有、回环或链路本地）
