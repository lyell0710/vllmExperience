# EXP-003 · profiling 工装验证（torch profiler + nsys）

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（15:08–15:16Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；模型 Qwen/Qwen2.5-0.5B-Instruct |
| 状态 | 完成 |
| 关联清单项 | R0-5 |

## 1. 目的与假设
验证两条 profiling 路径在本机可用：torch profiler 经引擎 HTTP 端点直控；
nsys 在无特权容器内可采 CUDA trace。

## 2. 环境与配置
- 服务：`CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8100
  --enforce-eager --max-model-len 2048 --profiler-config.profiler=torch
  --profiler-config.torch_profiler_dir=<绝对路径>`
- 控制：`scripts/profile_ctl.sh start|stop 8100`（curl POST /start_profile|/stop_profile）
- nsys 冒烟：`nsys profile -o <out> python -c "<torch matmul×10>"`（ENV-B python）

## 3. 步骤
起服务 → start_profile → 1 条 completion → stop_profile → 检查 trace 落盘 → 停服务。

## 4. 原始数据
- `pd_disagg/profiling/r0_5_torch_profiler_check.txt`（provenance + trace 清单 + nsys 结论）
- `pd_disagg/profiling/traces_smoke/`（rank0 worker trace 6.9MB + async_llm 前端 trace + profiler_out_0.txt）
- `pd_disagg/profiling/nsys_smoke.nsys-rep`（250KB）
- 首次失败现场：`profiling/r0_5_server.log` 早期版本含
  `Unknown vLLM environment variable detected: VLLM_TORCH_PROFILER_DIR` 与端点 404 日志。

## 5. 结果
- torch profiler：start/stop 200，worker + 前端 trace 均落盘 → PASS
- nsys：容器内正常生成 .nsys-rep（CUDA trace 完整）→ PASS

## 6. 分析与结论
**版本演化发现（B3 素材）**：v0.25.1 弃用 `VLLM_TORCH_PROFILER_DIR` 环境变量，
改为 `--profiler-config.*` CLI（ProfilerConfig dataclass，端点仅在
profiler_config.profiler 非空时注册——这就是首次 404 的原因）；0.17.1 仍用环境变量。

## 7. 异常、偏差与开放问题
- 首次尝试按旧文档用环境变量 → 端点 404 + Unknown env var 告警；翻 venv 内源码
  （entrypoints/serve/profile/api_router.py、config/profiler.py）定位新接口。

## 8. 下游影响
- `scripts/profile_ctl.sh` 头注释固化新旧接口差异；一律直控引擎端口（代理不转发，
  EXPERIMENT_PLAN 核验 #12）。B2/D1 的 profiling 均走此工装。
