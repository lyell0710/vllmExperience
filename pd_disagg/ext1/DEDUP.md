# EXT-1 上游查重判定（2026-08-23）

## 查询

```
gh api "search/issues?q=repo:vllm-project/vllm+nixl+telemetry+request+in:title,body"
gh api "search/issues?q=repo:vllm-project/vllm+nixl+per-request+in:title,body"
gh api "search/issues?q=repo:vllm-project/vllm+kv+transfer+time+per+request+metrics+in:title,body"
```

## 判定:EXT-1 不做上游 PR,仅作本地测量 patch

**冲突项：PR #52859**(open draft,2026-08-19,kyleliang-nv) "[KV Connector][Observability] Add lifecycle tracing for NIXL push and pull"
- 覆盖内容：NIXL pull/push 两模式的 request 级 lifecycle tracing(KV availability、 READ staging/start、producer/consumer completion/failure)，带 hashed request / remote-request correlation 字段、结构化日志 schema、采样率配置。
- 与 EXT-1 的上游化方向（per-request 传输遥测关联）实质重叠且更完整。按 AGENTS.md fail-closed 规则，不开竞争 PR。

**相关但不冲突**（其作者在 #52859 body 里也列了）：#44402(core request timing)、
#32573(token 级 OTel)、#43005(server-span 传播)。
另：#25388（closed，已合入）= 现状 stats.py 的来源，无 request 关联。

## EXT-1 的最终定位

本地最小 patch（`nixl_req_telemetry_v0251.patch`，打在 ENV-B site-packages， 全部行带 `# EXT1` 标记可还原），唯一目的 = 解锁报告红线 "D 等待远端 KV 对 TTFT 的关键路径贡献占比"（P/D/NIXL 三段同 request 身份同时钟域关联）。不投上游；若 #52859 合入，未来引用上游机制即可。
