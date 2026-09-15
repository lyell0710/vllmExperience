# 实验日记（Lab Journal）

> 规则：每次运作（任何工作段落）结束在本文件末尾追加一节，同一文件顺写、时间正序。每节固定四问：**做了什么 / 为什么（决策依据）/ 关键数字 / 产物路径**，末尾记下一步。细节标准：**凭本文件 + records/ 能复现当天全部操作与决策**——包括弯路、报错原文关键词、被否掉的方案。写简历时以本文件（叙事）+ RESUME_EVIDENCE.md（句子装配）+ README.md 台账（状态速查）三件套为准。数字一律以 provenance 文件为最终依据。

---

# 2026-08-21 · Day 0

## §0 上午：三 venv + NIXL smoke + 版本裁决（~09:03–09:15Z，依据落盘文件重建）

- **做了什么**：搭 `~/venvs/{v0.17.1, v0.25.1, main}`（setup_envs.sh，uv）；按官方 `docs/features/nixl_connector_usage.md` 同机示例跑 1P1D smoke：模型 Qwen2.5-0.5B-Instruct，P=GPU0:8100(side-channel 5600)、D=GPU1:8200(side 5601)、 toy proxy 8192，`--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.7`， `kv_load_failure_policy=fail`；v0.25.1 与 main 双版本各一遍；按预定规则裁决。
- **为什么**：清单 R0-2/R0-3。裁决规则**预先定死**防拍脑袋：三项检查（①1P1D 跑通 ②日志出现 "KV Transfer metrics：" ③failure_policy=fail 被接受），平手取 release。
- **关键数字**：双版本 3/3 全 PASS，平手 → **锁定 v0.25.1 = ENV-B 主战场**。 v0.25.1：avg xfer 14.128ms / P90 23.692ms / 0.188MB/次 / 13.271MB/s / descriptors=24 / post 0.911ms；main：avg xfer 14.7ms；双方 ERROR=0。
- **产物**：`smoke/smoke_{v0.25.1,main}_*`、`DECISION.md`。详见 `records/EXP-001`（含 provenance sha 语义问题的说明）。

## §1 四工程环境体检与修复（~14:10–14:50）

- **做了什么**：应用户要求体检 /root/projects 下四个工程（Kernel_Optimazation / Resume / TRT_TritonServer_example / vllm）+ 核对环境文档。逐项：
  - **机器底账**：2×RTX 4090 24G，driver 610.57.04（CUDA 13.3），toolkit 装有 12.8/13.0/13.2（/usr/local/cuda→13.2），cmake 3.22.1，g++ 11.4，conda py312（Python 3.12.11），torch 2.11.0+cu130 双卡可见。
  - **Kernel_Optimazation**：cmake 报 `No CMAKE_CUDA_COMPILER could be found` → 根因 nvcc 不在 PATH → `export PATH=/usr/local/cuda/bin:$PATH` 写入 `~/.bashrc`；四个 kernel（softmax/cuda-reduce/gemv/int8-quantize）全部试编译通过（scratchpad 外部 build，不污染工程）；补装 matplotlib。
  - **Resume**：ENV.md 写 pdflatex 但 tex 用 fontspec/xeCJK → **必须 xelatex**（文档有误）；缺 `ctexhook.sty` → `apt install texlive-lang-chinese`；中文字体全无（PingFang/Source Han/Noto CJK 三级回退全 miss，仅 DroidSansFallback） → `fonts-noto-cjk`；`photo.jpg` 缺失（需用户提供，且 ENV.md 写的文件名 "美签照片.jpg" 与 tex 引用不符）；用 matplotlib 占位图验证整条编译链出合法 1 页 PDF。插曲：装字体后首跑报 `can't find file 'Arial'`，重跑消失——fontconfig 缓存首建的瞬时问题。
  - **TRT_TritonServer_example**：缺 tensorrt/jupyter notebook/pycocotools/netron/ trex/tritonclient，opencv 仅 headless → 批量补装（tensorrt 11.2.1.2）。 **假 trex 事件**：ENV.md 的 `pip install trex` 装到的是 PyPI 同名无关包（"Templatosaurus Rex" 模板工具），它钉 typer<0.5 把 typer 降到 0.4.2，破坏 vllm 依赖链（fastapi-cli/fastsafetensors 要 ≥0.16）→ 卸载假包、 typer 恢复 0.27.1；真 TREx 主分支已移除，从 NVIDIA/TensorRT **release/10.16** 分支 sparse clone `tools/experimental/trt-engine-explorer` 源码安装（0.2.1）。**numpy 震荡**：TREx 的 pandas==2.2.1 把 numpy 拖到 1.26.4（破坏 opencv 5.x 的 numpy≥2 要求）；`-U numpy` 又冲到 2.5.2（破坏 numba<2.5 / mistral-common<2.4，均为 vllm 依赖）→ 最终定格 **numpy 2.3.5（原版本）+ pandas 2.3.3**，torch/vllm/numba/mistral-common/ opencv/trex 全量 import 复验通过。残留 pip 警告仅 trex 的 pandas 钉版本（形式冲突，实测无碍）。
  - **docker**：装 29.1.3 + nvidia-container-toolkit；本机无 systemd， `dockerd --iptables=false --bridge=none` 可拉起 daemon，但镜像层解压 `mount ... operation not permitted`，vfs 驱动再试 `unshare: operation not permitted` → **结论：compshare 无特权容器跑不了任何 docker 容器**（平台级限制），dockerd 已停。TRT 课程第 05 课 Triton 需换特权机或 `pip install tritonserver` in-process API。已注记 ENV.md。
  - 三份 ENV.md 均已修正（xelatex/字体/照片名；trex 安装法+docker 限制；nvcc PATH）。
- **为什么**：这台是新租的云主机，环境不落定后面全是流沙；ENV.md 的错误（pdflatex、假 trex）都是踩过才知道的坑，修文档=防二次踩坑。
- **产物**：三份 ENV.md 修正；conda py312 依赖定格。此段属环境维护，无 EXP 记录。

## §2 vllm 主仓切 editable（ENV-C 就位，~14:50–15:00）

- **做了什么**：发现装的 vllm 0.25.1 wheel 与仓库 main@7aa248fc（0.26.1rc1.dev682）不是同一份——改源码不生效、traceback 指向 site-packages。用户确认要改源码 → 切 editable：`python use_existing_torch.py` 剥 torch 钉版本（仓库钉 torch==2.13.0，环境是 2.11.0+cu130，直接装会换 torch）→ `VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation`。 **首次失败**：`ModuleNotFoundError: setuptools_rust`（--no-build-isolation 需自备构建依赖；此 commit 的 setup.py 无条件 import setuptools_rust）→ 补 `requirements/build/cuda.txt` 后成功。装后验证：`vllm.__file__` 指向仓库、 C 扩展加载 OK、平台识别 cuda；requirements 改动 `git checkout --` 还原，工作区净。
- **为什么/发现**：
  - 预编译算子按 torch 2.13 构建，担心 ABI 冲突——实际此版本已改用 **torch stable ABI**（扩展名 `_C_stable_libtorch.abi3.so`），跨 torch 版本兼容，实测过。
  - **cwd 遮蔽坑**：在 /root/projects 下裸跑 python，目录名 vllm/ 被当命名空间包遮蔽 import（`__path__` 指向仓库根、`__file__=None`）——上午 setup_envs.log 里 "v0.17.1 import 失败/版本号错" 正是同一坑的假故障，各 venv 中立目录复验全健康。
  - 附带升级：flashinfer 0.6.13→0.6.16.post3 及 cutlass-dsl/flashmla/tilelang 等按仓库 requirements 就位。
  - 注意事项：precompiled 模式改 `csrc/` 不生效（需全量源码构建 30-60min）。
- **产物**：ENV-C（conda py312 + /root/projects/vllm editable）就位。

## §3 执行清单对齐 + 重复劳动止损（~15:00–15:04）

- **做了什么**：用户贴出 v3 锁定版清单，我按清单启动时误起了 pd-lab 下两个新 venv 安装任务——随即读到仓库里已有 `experiments/pd_disagg/`（smoke 结果、 DECISION、setup_envs、rr_proxy 都在）→ **发现 R0-3 已完成、R0-2 venv 已在 ~/venvs 约定位置** → 杀掉两个重复后台任务、删除 pd-lab 目录，全面转入 "先盘点存量再动手"。
- **为什么**：v3 清单与仓内 EXPERIMENT_PLAN v2 在"主战场"上有历史分歧，以 DECISION.md 裁决 + v3 清单为准（v2 的 "main=交付" 段作废）；R0-6 靶子（两个 bug 的"发现/修复"措辞）在本地 tex 中不存在，确认在线上稿（仅用户可改）。
- **教训**：动手前先盘点存量。清单状态 ≠ 磁盘状态。

## §4 R0-1 硬件三数（~15:04–15:07，EXP-002）

- **做了什么与弯路**：
  - `nvidia-smi topo -m` 报 `hwloc: Topology does not contain any PU`（容器读不到 CPU 拓扑）→ 换 `topo -p2p r` + PCIe link 查询替代，注记于 topo.txt。
  - cuda-samples 构建三次试错：新仓库目录已从 `Samples/` 改为 **`cpp/`** 前缀；样例 CMakeLists 还要仓库根 `cmake/`（InstallSamples.cmake）——sparse checkout 连补两次目录后编译通过。
  - nccl-tests：NCCL 取自 ENV-B venv 的 nvidia-nccl wheel（无 .so 链接符号 → 自建 nccl-home 软链 libnccl.so→libnccl.so.2），`make MPI=0`， `all_reduce_perf -b 1M -e 512M -f 2 -g 2`。
- **关键数字**：P2P connectivity=0、`topo -p2p r`=**GNS（驱动级禁用）**；单向 D2D **0.60–0.91 GB/s**（无 P2P 的 cudaMemcpyPeer 分段中转）、双向 **22.6–22.8 GB/s**（Gen4 x16 双向流水）、GPU 间延迟 14.5–15.9µs、本卡内 ~924 GB/s；NCCL allreduce（SHM 回退）avg bus bw **1.78 GB/s**（256M+ ~1.85）。单双向 25 倍差 = "无 P2P" 的定量指纹。
- **推论（当场写下，当晚全部应验）**：8K KV≈460MB 穿卡在秒级；TP=2 将被 allreduce 重压 → 四臂区分度会极大。
- **产物**：`hw/{topo,p2p_bandwidth_latency,all_reduce_perf}.txt`（均带 provenance）；工具构建在 /root/tools（仓库外）。红线 **"P2P 受限" 解锁**。

## §5 R0-5 profiling 工装（~15:08–15:16，EXP-003）

- **做了什么与失败现场**：0.5B 起真实引擎验证 torch profiler 直控。首次按旧文档设 `VLLM_TORCH_PROFILER_DIR` → 日志 `Unknown vLLM environment variable`、 `/start_profile` **404**。翻 venv 源码定位： `entrypoints/serve/profile/api_router.py` 仅在 `profiler_config.profiler` 非空时注册路由；`config/profiler.py` 的 ProfilerConfig 是新接口 → 改用 `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<绝对路径>` → start(200)→completion→stop(200)，worker trace(rank0, 6.9MB gz) + AsyncLLM 前端 trace + profiler_out_0.txt 落盘。nsys 容器内冒烟（torch matmul）正常出 rep。
- **B3 素材**：0.17.1 用环境变量、0.25.1 改 CLI config——接口演化实锤一例。代理不转发 profile 端点（核验 #12）→ 工装设计为直控 8100/8200。
- **产物**：`scripts/profile_ctl.sh`、`profiling/r0_5_torch_profiler_check.txt`、 `profiling/traces_smoke/`、`profiling/nsys_smoke.nsys-rep`。

## §6 证据仓库定型 + GitHub 私有备份（~15:20–15:45）

- **做了什么**：发现 experiments/ 被外层仓库 `.git/info/exclude` 忽略——所有证据 **零版本控制**，云主机一挂全丢 → 建独立嵌套 git 仓库（外层保持干净，符合 AGENTS.md 的 PR 卫生）。用户建 GitHub 私有仓库（我提醒关掉 Add README 避免远端初始 commit 分叉），gh 已登录 lyell0710 → `master→main` 改名、配 origin、首推两个 commit（fece3ca 地基 / 8597516 定型）。四层结构定型： README（约定+台账+红线）/ RESUME_EVIDENCE（句子装配）/ results/README（B1 schema）/ scripts。硬约定六条（provenance 首行 / 统一命名 / raw-derived 分离 / 单位入表头 / gate 同行存储 / 图表样式）。
- **产物**：github.com/lyell0710/vllmExperience（private, main）。

## §7 C2 远端查重收尾（~15:44）

- **做了什么**：gh 直连三查——①上游 main configs 目录：**无任何 E=30 文件**， E=60，N=704 仅 `AMD_Instinct_MI300X`；②PR 全状态搜索 "E=30 N=1408" / "E=60 N=704"：命中仅 #52651(GPTQ bugfix)/#24700（默认 config 分析，CLOSED）/
  #41834(DeepSeek SM12x)，无 config 类冲突；③issue "Qwen1.5-MoE 4090 config"：
  仅 #15561（旧加载问题）。#48309（4090D fp8）仍 OPEN 未合并，继续作相邻先例。
- **结论**：目标 tuple 空缺确认、无重复 → **红线"社区空缺"解锁**（引用本日）。
- **产物**：`moe_configs/DEDUP.md` 远端复核节。

## §8 SLO 方案锁定 + B1 colocate 归因（~15:47–15:52，EXP-004）

- **做了什么**：用户问 SLO 怎么设 → 定 DistServe 式相对方案并说明理由：固定绝对值在 512/2K/8K 三桶下必失效（200ms 让 8K 全零、5s 让 512 全满，信息量归零）→ **TTFT≤5×该桶无负载基线 + TPOT≤50ms 固定**（20 tok/s，约 3 倍人类阅读速度）+ 附录 SLO-scale(1.25/2.5/5/10×) 敏感性曲线（raw 存每请求延迟，可重算——回应"为什么是 5×"的完整防御）。服务：`CUDA_VISIBLE_DEVICES=0 vllm serve Qwen2-7B-Instruct --port 8100 --max-model-len 16384`（默认 CUDA graphs，84s 就绪）。跑 512/2048/8192×128、并发 1、32 请求/点（seed 42、ignore-eos、percentiles 50/90/99、 --save-result --save-detailed）。工装链（快照→bench→快照→collect_point→ runs.jsonl）首跑验证通过。
- **关键数字**（TTFT p50/p90/p99 // TPOT p50 // GPU·s/req）：
  - 512: 65.5/66.4/80.5 // 15.87 // 2.08 → SLO **328ms**
  - 2048: 178.3/181.6/186.2 // 15.93 // 2.20 → SLO **891ms**
  - 8192: 925.2/946.0/950.8 // 16.34 // 2.95 → SLO **4626ms**
  - 解读：bs=1 TPOT 恒 ~16ms（≈63 tok/s，decode 权重带宽约束）；TTFT 随输入近线性（prefill 计算主导）。SLO 表 commit 锁定，不回改。
- **产物**：runs.jsonl 1–3 行、`results/README.md` SLO 表、 `scripts/{run_point.sh,collect_point.py}`。

## §9 replica2 + 异常调查（功率帽）+ tp2（~16:24–16:45，EXP-005）

- **做了什么（时序）**：
  1. replica2 起双实例（8100 就绪 ~70s、8200 ~2s）+ rr_proxy:8300，三点归因： 512=65.2 / 2048=173.5 / **8192=714.6ms**。
  2. **异常**：8K 比 colocate(925) 快 30%？并发 1 下不应该。诊断三连（直连绕代理）：
     - diag-1 @8100 16 请求：p50 **901.2** / p90 929.0—— ≈ colocate
     - diag-2 @8200 同参：p50 **892.7** / p90 912.1—— 两卡无差异
     - 拆 colocate 8K 原始分布：**双段**！前 ~8 请求 702–739ms，其后 897–951ms； replica2 全部 697–733ms → 指向"单卡持续负载劣化"
     - diag-3 持续负载（8192×16out×40 请求）+ nvidia-smi 1.5s 采样：空闲 210MHz/14W/reason 0x1 → 负载 40→63°C、427–443W（帽 450W）、 SM 2820↔2460–2535MHz、**reason 0x4 = SW Power Cap**。TTFT 稳态 p50 905.6ms。**机理坐实：功率帽（非热，63°C）**；replica2 轮转=50% 占空比维持 boost。频率降 ~12% 与 TTFT +30% 不完全成比例（疑瞬时 boost/显存钟，未深究）。
  3. **方法论决定**：attribution=各臂占空比工况，headline 以 sweep 为准； run_point.sh 即刻加 GPU 遥测（2s 采样→runs.jsonl `gpu_telemetry`）； SLO 维持锁定值（5× 余量≫30% 效应，且换基线=回改）。
  4. tp2（`-tp 2`，96s 就绪，双卡各 23.8G）：512=62.8 / 2048=173.5 / 8192=693.7ms；TPOT **9.26–9.48ms**。
- **发现② TP2 不对称收益**：decode 16→9.3ms（**-42%**：每卡半份权重带宽分摊，小消息 allreduce 代价 ~1.3ms/token）；8K prefill **零加速**（694≈冷态单卡 700ms）：28 层 × 58.7MB 大消息 allreduce 撞 1.78GB/s collective 墙（§4 印证）——计算减半被通信吃光。
- **工装事故记录**：pkill 模式含字面量两次误杀自身 shell（exit 144）→ 改 `pkill -f '[v]llm serve'` 方括号技巧。诊断三连未存 raw（当时图快）→ 数字为终端级证据，完整命令补录于 EXP-005《replica2/tp2 归因 + 功率帽节流调查》 §4，并催生约定 #8。
- **产物**：runs.jsonl 4–9 行、`records/data/EXP-005_throttle_trace.csv`。

## §10 PD 探针 + pd1p1d 归因 + NIXL 大传输（~16:49–16:58，EXP-006）

- **做了什么（时序）**：
  1. 起 P（GPU0:8100/side5600/kv_producer）+ D（GPU1:8200/side5601/kv_consumer）
     + toy proxy：8192。**未用 enforce-eager，CUDA graphs 与 NIXL 共存正常**（相对 smoke 的升级）。单请求验证通路（"The capital of France is → Paris"）。
  2. **指标探针**（快照→1 请求→快照→diff）摸清 v0.25.1 指标体系：
     - 传输计数**全在 D 端**（Pull 语义）：`nixl_bytes_transferred_{sum,count}`、 `nixl_xfer_time_seconds_sum`、`nixl_post_time_seconds_sum`、 `nixl_num_descriptors_sum`；
     - P 端仅 `nixl_num_failed_{transfers,notifications}_total`、 `nixl_num_kv_expired_reqs_total`；`_created` 系列是时间戳须排除；
     - bonus：`prompt_tokens_by_source_total{source="external_kv_transfer"}` D 端逐 token 记账远端 KV——比 bytes 更硬的"传输真实发生"证据；
     - 单请求对账：bytes=917504 = **16 token × 57344B**（block=16 取整）， ext_kv_tokens=8（=9-1，D 自算最后一 token），desc=28（=28 层）。
  3. collect_point.py gate 判定改精确指标名（跨端口求和，弃子串猜测），字段扩展（xfer/post 时间、descriptors、external_kv_tokens、failed_notifications）。
  4. 三点归因：512=214.4 / 2048=554.6 / **8192=2685.4ms**，TPOT ~16ms， GPU·s/req 4.46/5.16/**9.24**。gate 全 PASS（transfers=32=completed， failed/expired 全 0）。
- **NIXL 大传输实测**（R0-1 第三数收尾）：
  | 桶 | bytes | MB/次 | avg xfer | post 总 | desc/次 | 有效吞吐 |
  |---|---|---|---|---|---|---|
  | 512 | 0.940GB | 29.4 | 113.9ms | 121ms | 1792 | 0.26GB/s |
  | 2048 | 2.820GB | 88.1 | 330.2ms | 142ms | 5380 | 0.27GB/s |
  | 8192 | 14.069GB | 439.7 | **1602.7ms** | 714ms | 26834 | **0.27GB/s** |
  - **有效吞吐跨尺寸恒定 0.26–0.27GB/s**：descriptor ≈16KB/个（每 block 每层单发，56/block=28 层×K，V）→ 碎片化小拷贝，量级与 §4 无 P2P 单向路径一致。
  - PD TTFT 分量对账：2685 ≈ P prefill（~900 热态） + xfer(1603) + D 首步/代理 ✓。
  - 措辞红线：只可称 telemetry-derived effective throughput；xfer 不与 post 相加。
- **开放问题 → B2**：D 实拉 **7668 token/req**（<8192；bytes/57344=7668 与 ext_kv_tokens/32=7667 独立互证）——疑与 block 取整/前缀缓存/末 block 自算的记账规则相关。定论前 KV 量一律引用 bytes 实测，不用 input_len 推算。
- **产物**：runs.jsonl 10–12 行、`snapshots/exp006_probe_*`（探针快照×4）。

## §11 实验记录体系建立 + 证据完备性修复（~17:00+）

- **做了什么**：用户要求"每次的实验记录完全写好"→ 建 `records/` 体系（TEMPLATE 八节：目的/配置/步骤/原始数据/结果/分析/异常/下游影响）， EXP-001~006 全量回填（含完整命令、失败现场、决策依据）；自查抓出两个漏洞并修复： ①功率帽采样 CSV 与探针快照还在会话临时目录（会话结束即丢）→ 入库； ②诊断三连没存 raw → 记录里如实标"终端级证据"+补录命令，定 **README 约定 #8**（任何 GPU 跑一律存 raw；记录当场写不隔夜）。用户再定死日记规矩： **每次运作=本文件末尾追加一节**（已存入 Claude 长期记忆，跨会话生效）。随后按"复现级"标准重写本日全部日记（本版）。
- **产物**：`records/`（TEMPLATE+EXP-001~006+data/）、README 索引表与约定 #7/#8。

## Day 0 未完成清单（诚实账）

- **sweep（offered-load 扫描）未跑**——S1 headline 数字与 goodput 全部来自它，8/22 首位。
- B2 归因未动（xfer 直方图桶分析、"7668 token"溯源）。
- R0-4 阻塞：0.17.1 课程脚本不在本机（等用户提供；超时降级为源码机理分析）。
- R0-6 线上稿（仅用户可改）。
- EXT-1/EXT-2 未动（弹性，不阻塞）。
- colocate/replica2 六个点无 gpu_telemetry（遥测工装晚于它们；如报告需要可低成本重跑）。

## 下一步（8/22）

1. sweep 网格设计（各桶 rps 档位由 attribution 吞吐推算）→ colocate 先行验证 sweep 流水线 → 四臂扫描。
2. B2：xfer_time 直方图桶分析 + "7668 token" 记账溯源（读 D 端调度/connector 源码）。
3. 每段收尾固定链条：EXP 记录 → 日记追加 → 台账更新 → commit+push。

# 2026-08-21 · Day 0 夜间：B1 sweep 战役（全臂完成）

## §12 sweep 战役 + 两个后台源码分析（~17:30–20:30，EXP-007）

- **做了什么（时序）**：
  1. 用户下达全量执行令。派两个后台 AI 分析任务（0.17.1 双 bug 机理 → `analysis/p2pnccl_bugs_id_chain.md`；7668 token 溯源 → `analysis/nixl_token_accounting.md`），GPU 战役同时开跑。
  2. **7668 溯源结果引爆方法论修正**：两计数器实为分毫不差（245,344 token 整； "7668/7667"是双重舍入假象），缺口=前缀缓存命中（511 块可源码定罪到 bench 的 test 请求：serve.py：824-871），且**同 seed 下短桶 prompt 是长桶精确前缀**——用快照 local_cache_hit 计数器实测證实：同 session 顺序跑时 2048 桶 25% 命中、 8192 桶 8.6%。→ **协议 v2：每点唯一 seed**（跨臂同点位同 seed 保可比）， colocate 全套重跑。干净 2048 基线 224.9ms（污染版 178，差值精确等于缓存效应），预测应验。
  3. 四臂各一个 session（fresh 栈 → attribution → saturation → sweep）： colocate 19 点、replica2 24 点、tp2 23 点、pd1p1d 21 点，v2 有效行 84。偶发 ServerDisconnected 3 次（1/192 请求级，失败行保留+同 seed 重跑）。 tp2 首启 OOM（0.9 利用率 warmup 差 26MB）→ 0.88 重启，偏差入记录。
  4. 0.17.1 双 bug 分析交付（assert 崩溃点 connector：433、随机后缀分叉点 input_processor.py：212、D 端无超时 Condition.wait 挂死 engine：317、GET 模式静默乱码、四层 ID 链、NIXL 身份拆分对照）——R0-4 降级路径完成，S2 弹药齐。
- **关键数字**（详表见 EXP-007《B1 四臂 offered-load 扫描战役》与 runs.jsonl）：
  - 饱和 req/s（512/2K/8K）：colocate 10.36/3.63/0.90（单卡）、replica2 15.58/7.00/1.78（2K/8K 近完美 2×）、tp2 12.31/4.16/1.02（双卡仅 +13-19%）、 **pd1p1d 7.84/2.12/0.54（双卡全面低于单卡；8K=0.54 与 0.27GB/s 传输墙理论上限 0.57 吻合）**
  - goodput 峰值：replica2 12.75/4.96/0.90 全场最高；pd1p1d 512 桶 66% 饱和度时 goodput 已崩至 1.59——传输延迟吃光 SLO 余量（"PD 税"定量化）
  - v2 同热工况归因：四臂 8K prefill 881-925ms 几乎无差（功率帽整平）， tp2 decode 优势 9.3ms 依旧
- **选型结论（S1 主句素材）**：互联受限双 4090 上，短请求 replica2（=colocate×2）吞吐王且 per-GPU 效率与单卡打平；tp2 只在需要 decode 延迟或单卡放不下时考虑； **PD 分离在 0.27GB/s 有效传输带宽下不可取**。
- **产物**：runs.jsonl（109 行）、records/EXP-007、analysis/ 两篇、协议 v2 工装（SEED 支持 + seed 字段入行）。
- **下一步**：出图（figures/）→ B4 报告 → B3 有限对照 → C1/C3 MoE 上卡 → 汇总单+教学手册。MoE 模型下载后台进行中。

## §13 出图 + B3 有限对照 + C1 MoE 上卡（~20:30–21:05，EXP-008/009）

- **做了什么（时序）**：
  1. **出图**（dataviz 流程：先选形式、调色板过验证器、色序固定、结论句标题、 provenance 脚注）：fig1 goodput 四臂曲线（+y=x 理想线）、fig2 TTFT p99+SLO 线、 fig3 per-GPU 成本、fig4 PD TTFT 分解（传输 54–64%）、fig5 NIXL 延迟地板→ 带宽墙（log-log）、fig6 SLO 敏感性（0.5–4× 臂间排序稳定）+ derived/sweep_summary.csv。亲眼检查全部渲染；修正 fig3 标题过度声明（512 桶 replica2 网格未达真实拐点）与 fig4 百分比区间。 v2 数据的意外佐证：干净 seed 下 D 拉取 = 469.8MB = 8192×57344 **精确全量**。
  2. **B3 有限对照**（0.17.1 vs 0.25.1 单实例，同协议同 seed）：无负载延迟 Δ<1%、TPOT 持平；**512 桶饱和 7.14→10.36（+45%）**、计算受限桶零差异；启动 308s vs 58s。只作 system-version comparison 表述。
  3. **C1 MoE 上卡成功**：Qwen1.5-MoE-A2.7B-Chat TP2+EP（util 0.88，启动 216s 含 AOT compile）。**C2 运行时铁证**：日志原文点名 `Config file not found at .../E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json`——证据链三重闭环（本地+远端+运行时）。未调优基线：TPOT **4.62ms**（dense 7B TP2 的 2.0×）、饱和 11.50 req/s@512——D2 调优的 before。
  4. 工装事故：MoE bench 首轮全 404——run_point 的 MODEL 默认值没改，教训 "多模型阶段 MODEL 必须显式设置"；失败行按规则保留。
  5. C3 checkpoint 锁定 Qwen/Qwen3-30B-A3B-GPTQ-Int4（官方 GPTQ Int4 = W4A16 路线），后台下载中。
- **产物**：figures/fig1-6、derived/sweep_summary.csv、EXP-008、EXP-009、台账更新（B1✅ B2◐ B3◐ R0-4✅ C1✅ C2 三重闭环）。
- **下一步**：C3 上卡 → B4 报告成稿 → 汇总单+教学手册 → RESUME_EVIDENCE 终更新。

## §14 报告成稿 + 汇总单/教学手册 + C3 上卡（~21:05–21:30，EXP-010）——Day 0 收官

- **做了什么**：
  1. **B4 报告 v1**（`pd_disagg/REPORT.md`）：一页结论（选型表+三机理发现）、硬件画像、四臂矩阵（归因/扫描/PD 公平陈述）、演化三句话展开、 bug 链路（诚实署名）、归因方法论声明、附录。
  2. **汇总单**（SUMMARY.md）+ **零基础教学手册**（STUDY_GUIDE.md：10 概念全用自家数字锚定 + 12 数字卡 + 15 题面试预演 + 4 小时学习路径），并发布网页版手册（artifact："四臂实验手册"，嵌 4 张核心图 + 折叠式自测）。
  3. **C3 上卡**：checkpoint 锁定 Qwen/Qwen3-30B-A3B-GPTQ-Int4（官方 GPTQ Int4 = W4A16，MarlinLinearKernel 确认）；TP2+EP 启动 184s，smoke 连贯； TPOT **4.93ms** / 饱和 10.02 req/s@512。有趣对比：30B 总参 W4A16 与 2.7B 激活 BF16 的 decode 速度相当（4.93 vs 4.62ms）——D1/D4 的现成切入点。注：该 shape 未出现 config 缺失告警，机制待 D 阶段核实，不作断言。
  4. RESUME_EVIDENCE 数字成稿（S1 候选句已填全部实测值）；README 台账/索引全同步。
- **Day 0 终账**：清单 R0 全绿（R0-6 线上稿除外）、B1✅ B2◐ B3◐（有限） B4✅v1、 C1✅ C2✅ C3✅；EXP-001~010；runs.jsonl 111 行；六图一表；GitHub 全量备份。磁盘余 12GB（D4 需对照模型时先清理）。
- **下一步（8/22）**：教用户过 STUDY_GUIDE（15 题自测）；用户侧两件事（线上稿排雷、课程脚本）；然后按 M2/M3 节奏进 D 阶段（D2 baseline 已备）。

## §15 EXT-2 push 单点 + R0-4 复现环境搭建（2026-08-22 上午，EXP-011）

- **做了什么**：①磁盘清缓存（uv 37G + pip 4.5G，纯缓存）；②EXT-2：从 v0.25.1 tag 提取 push 专用 proxy（disagg_proxy_pushconnector_demo.py），起 NixlPushConnector 1P1D，512/8192 归因跑通，与 pull 同 seed 对照（EXP-011《EXT-2 NixlPush 单点》）；③R0-4：从 v0.17.1 tag 提取官方 xPyD proxy+脚本，精简为本机 launch_1p1d.sh（Qwen2-7B 双卡 P2pNccl）， P/D 起成功、NCCL 握手 OK，装了 quart；一次经 proxy 的请求探测被额度中断未得结论。
- **关键数字**：push 8K TTFT 2537ms（pull 2718，-6.7%）、有效吞吐 0.30GB/s（pull 0.27）、计数在 P 端（WRITE 发起方）——但量级不变，**传输方向救不了 PD**。
- **为什么**：EXT-2 是清单弹性项但数据便宜（复用 pd 栈换 connector）；R0-4 发现课程脚本非必需——官方示例就在 git tag 里，可自建复现。
- **产物**：EXP-011、matrix/disagg_proxy_pushconnector_demo.py、 p2pnccl_repro/（launch_1p1d.sh + proxy + 日志）、collect_point is_pd 修复、 **HANDOFF.md（交接文档）**。
- **下一步（交接给下个 agent）**：见 HANDOFF.md §5——P1 跑完 R0-4 动态复现（bash launch_1p1d.sh 后发请求看 D 挂死）→ EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》 + B3 完整版；P2 EXT-1； P3 B4 v2；P4 九月 D 阶段。

## §16 R0-4 动态复现收官：两 bug 实机坐实（2026-08-23，EXP-012）

- **做了什么**：接手上会话搭好的 `p2pnccl_repro/` 栈，实机 1P1D（0.17.1 P2pNccl， Qwen2-7B 双卡）动态复现两个缺陷并落原始崩溃/挂死日志。
  - **bug2**（分叉挂死）：经 proxy 发正常请求 → 客户端挂死、D 端零 decode 日志、 D 全线程 wchan=futex_wait_queue、GPU util 0、连发两请求均挂、P /health 恒 200 → "单边 D 挂死不自愈" 签名完整。
  - **bug1**（：433 assert）：先试裸直连 P，**意外**先崩于 `connector:518` parse_request_id ValueError（裸 id 无地址串）——早于预期的：433；遂用手工 `X-Request-Id` 注入 `___prefill_addr..._decode_addr...___` 地址串 + max_tokens=16， **精确命中 `connector:433` AssertionError**，P EngineCore 崩溃、HTTP 500、/health 503。
- **关键数字/证据**：4 份 raw（bug2_evidence + bug1_Pdirect_crash + bug1_L433_assert + live_preflight）均在 `p2pnccl_repro/raw/EXP-012/`，bug1 两条有 EngineCore 原生 traceback。
- **为什么/意义**：把静态 file：line 升级为动态崩溃现场；**实证修正**静态分析——缺陷 1 触发需"地址串 id + max_tokens>1"两条件齐备，裸直连会先崩：518。措辞红线"复现/定位/验证" 三词现全有实测背书（仍禁"发现/修复"）。
- **取证限制（诚实）**：py-spy 精确 Python 栈帧未取——容器 ptrace_scope=1 且 /proc 只读、无 CAP_SYS_PTRACE、gdb 未装；bug2 的：317 定位以 wchan+行为学+静态 file：line 三方闭环。
- **产物**：EXP-012 记录、raw/EXP-012/（4 文件+服务端日志）、analysis 文档加"⚑实测修正"、 README 台账 R0-4 转 ✅ + EXP 索引补 010/011/012 + 措辞红线更新。
- **下一步**：B3 完整版表述据此定（0.17.1 PD 默认配置正常请求即触发 D 挂死→不可用对照臂， vs 0.25.1 NIXL 可用）；再往后 P2 EXT-1（解锁 KV 占比红线）、P3 B4 v2 定稿（8/31）。

## §17 EXT-1 request 级 KV 归因落地 + B4 v2 定稿（2026-08-23，EXP-013）

- **做了什么**：①上游查重：发现开放 draft PR **#52859**（NVIDIA，NIXL push/pull lifecycle tracing）已覆盖 EXT-1 上游化方向 → fail-closed，定位为**本地测量 patch**（`ext1/DEDUP.md`）。②Patch：ENV-B site-packages 16 行（`# EXT1` 标记全可还原，原件备份 ext1/orig/）——pull_worker 首见请求记时 + base_worker 传输 DONE 时按 req_id 聚合 telemetry 并输出 `EXT1_KV` 行 + 失败路径清理。③测量栈：instrumented proxy（6 epoch 打点+透传 X-Request-Id）+ 流式 client（每请求唯一 id/seed）+ EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》同配置 1P1D，3 桶×12 请求。④三方 join 分析 + B4 报告 v1→v2 全量升级。
- **关键数字**：**KV 等待占 TTFT 54.2% / 62.5% / 64.2%**（512/2K/8K，p50， p10–p90 ±2% 内）——红线"KV 占 TTFT X%"正式解锁为因果占比声明。三重互证：逐请求 bytes 和 = Prometheus 计数器**分毫不差**（7398752256）；kv_wait − xferDuration = 0.3–1.9ms（等待≈传输本身）；六段分解闭环误差 p50 <0.1%（最差桶 0.084%）。无扰动：patch 后 TTFT 218/727/2738 vs 矩阵 219/719/2719。36/36 身份匹配； idx=0 首请求显式观测到 handshake 一次性成本（512 桶 +292ms）。
- **为什么**：这是 B2 归因层最后一块——之前只能"分量对账"，现在 P/D/NIXL 三段同 request 身份同时钟域逐请求关联，因果占比可发布。
- **产物**：EXP-013、`pd_disagg/ext1/`（patch+DEDUP+工装 4 件+raw/EXP-013+ derived/ext1_per_request.csv）、README 台账（B2 ✅ 收官、EXT-1/EXT-2 ✅、红线解锁、索引+013）、**REPORT.md v2 定稿**（§2.2 因果占比、§2.4 推/拉对照、 §3 B3 完整版两维度表述、§4 动态复现三路径、§5 三重互证方法论）。
- **下一步**：M1 交付物已成稿（8/31 前富余）。转 P4 九月 D 阶段前置：D1 nsys MoE 分解（baseline EXP-009/010 在案）→ D2 config 调优 + PR 六件套准备。

## §18 D1 收官:MoE decode 反转点 + kernel 分解(2026-08-23,EXP-014)

- **做了什么**：①同轴扫描 MoE(A2.7B TP2+EP)vs dense(7B TP2)decode 吞吐（输入 128/输出 256，并发 1→128 八档）；②nsys kernel 级分解（cudaProfilerApi 控窗 + `--cuda-graph-trace=node`），bs=1/32 两窗；③分析工装 3 件 + 报告图。
- **关键数字**：**MoE 优势在 bs≈8 反转**——2.03×(bs=1)→0.97×(bs=8)→ 0.74–0.82×(bs≥16)；机理=top-4/60 命中并集随 batch 趋全量，28.6GB/step 读放大 > dense 14.2GB。kernel 占比：bs=32 时 **fused_moe grouped GEMM 56.4%**（bs=1 时 dense GEMV 40.9%，lm_head 0.31GB/token/rank 是隐性大头）； AllReduce 恒 ~14–15%（TP2 固定税）。D2/D3 目标由数据锁定：fused_moe 路径。
- **方法学收获（面试弹药）**：nsys 默认 graph-level trace 下 CUDA graphs 内 kernel 不单列，首采的"分解表"实为 prefill 混样（fused_moe 仅 4 step 实例）； node 级重采后 other 桶从 77%→1.2%。graphlevel 采集文件保留作对照证据。
- **产物**：EXP-014、moe_perf/{d1_sweep.sh,d1_nsys.sh,d1_analyze.py,d1_kernels.py}、 figures/d1_fig1_decode_scaling.png、derived/d1_{scaling,kernel_share_bs1,bs32}.csv、 raw/EXP-014/（16 bench JSON + 4 nsys rep + 遥测）。
- **下一步**：D2 调优已后台开跑（EP → 非 EP）；FP8 checkpoint(D4)已下载就位； D5 预研完成（qwen3_moe 支持 EPLB/qwen2_moe 不支持 → 用 30B-A3B-GPTQ， rearrange 证据锚点 eplb_state.py：748）。

## §19 D4/D5 收官 + 交付物对抗校验(2026-08-23 下午,EXP-016/017)

- **做了什么**：①D4：FP8 vs W4A16 双臂 bench（4 点）+ 同 token 集 wikitext PPL；②D5：三臂（w4a16 拒/fp8 重排/无-EPLB 对照组）gate 判定；③20 个校验 agent 对 REPORT/记录/台账做数字重算与一致性对抗校验，确认 20 条真实问题（台账重复行、陈旧引用、闭环误差口径 0.084%>0.08% 等）全部修复； ④调度重构：GPU 管线做成自驱动链（D4→PPL→D5→tune→AB），D2 全量调优移到最后长跑。
- **关键数字**：D4——W4A16 decode 全 regime 胜 23–48%(TPOT 4.91 vs 7.10ms@bs1)，FP8 仅 c128 TTFT 反超（497 vs 613ms）+ PPL 优 3.3% 相对（7.663 vs 7.922，同 31212 计分 token）；Ada 落地解释钉到 oracle/fp8.py：103-122（capability 90/100 快路径跳过 SM89 → TRITON）。 D5——GPTQ 拒于 routed_experts.py：151；FP8 臂 2 次真实重排（balancedness 0.53–0.74）；**对照组逐字节一致 → 分歧因果归属 EPLB**，定性数值性； 按 gate 规则 D5 不上简历（维持默认）。
- **事故与教训（三则，均已记录在案）**：①pkill 连坐第三形态：后台命令 wrapper 的 cmdline 含 heredoc 全文，pattern 匹配 wrapper 自身 → heredoc 写脚本与执行必须分开投递；②被杀链的子进程成为孤儿继续跑，与新链抢 GPU → 清理必须按 PID 全树，且清理模式可能误伤新链同名进程（D5 首败即此竞态， 复跑排除）；③服务健康检查窗必须按最慢臂设置（GPTQ 加载 11 分钟 vs 10 分钟窗，差 1 分钟被误判启动失败）。
- **产物**：EXP-016/017、moe_perf/raw/EXP-01{6,7}/ 全套 raw、d4_ppl.py、 d5_control.sh、对抗校验修复批次（commit cc473de）、台账/简历证据/WEEKLY 同步、SUMMARY 快照横幅。
- **下一步**：链 5 长跑中（D2 全量调优 EP→非 EP→AB，预计 10–14h）→ 出数后 EXP-015《D2 MoE config 调优》 + PR 分支 + PR_DRAFT 回填（提交必须由用户本人）；推送队列在途（大文件 ~13KB/s 慢爬，本地 commit 为锚）。

## §20 D2 收官:两个空缺 config 交付 + 六件套齐备(2026-08-23 傍晚,EXP-015)

- **做了什么**：①全量调优两 tuple（EP 8916s / 非 EP 4097s，1920 配置×19 M 档[8/24 勘正：实为 18 档，曾误计元键，见 §21]， ray 双卡）；②kernel A/B + e2e A/B（default→装 JSON→tuned）；③correctness（main venv 补 pytest/tblib 后 120 passed）；④识别并消除 Triton 首跑 JIT 伪影（c32 TTFT 1021ms→warmup 复测 225ms）；⑤PR 分支 moe-config-4090- qwen15moe 建好、两 JSON 暂存、PR_DRAFT 六件套数字全回填。
- **关键数字**：kernel 两端改善（M=1：EP **-8.5%**/非 EP -3.8%；M≥128： -3.3~-3.9%；中段持平——如实陈述，不吹"全面提升"）；e2e TPOT **+0.8~1.2%** 三档一致（≈kernel 增益×fused_moe 占比 56.4%，机理自洽）； 吞吐/TTFT 会话噪声内持平（D1 参照证明会话间漂移 ±5~8% > 效应）。
- **D3 依数据判定**：中段 M 打平 → config 即最优杠杆，不做无数据支撑的 kernel 改动（这本身是结论，也是面试口径）。
- **推送风波（已根治）**：GitHub 100MB pre-receive 拒收 + 管道 tail 吞返回码导致两次"假成功"；plumbing 重写 9 commit 移除超限 blob 时又因真实 index 未同步把大文件带回（git commit 提交的是整个 index！）——二次全量重写 + git reset 后干净落地（c8663cd）。三条教训全部入 README/HANDOFF。
- **产物**：EXP-015、configs_{ep,noep} JSON、kernel/e2e 全套 raw、 correctness_tail、PR_DRAFT 终稿、d2_e2e_rerun.sh。
- **状态**：清单 D1–D5、EXT-1/2、持续项 P1（材料层）/P2/P3 全部完成； M1/M2 提前达成，M3 达成至"材料齐备待用户提交"。唯余用户侧：R0-6 线上简历排雷、PR 本人 review+签名+提交。

## §21 SGLang config 侦察 + 19→18 定档修正(2026-08-24 上午)

- **做了什么**：①SGLang 侧同构空缺侦察（只查重记档，未动工）：全库 363 个 fused MoE config 中 4090 仅 2 个旧 fp8 文件，E=30，N=1408 与 E=60，N=704 在全部 Triton 版本目录均缺；gh 远端三组关键词查重无冲突 → 判定第二 PR 机会开放（须 sglang 运行时 A/B 验证后再提；源码已 shallow clone，venv 未装， 用户暂停待指示）。②定档修正批次（commit 4bb075f）：D2 交付 JSON 的 M 档数 19→18 勘正（triton_version 元键曾被计入档数）、e2e +0.8~1.2% 降级出 headline（低于跨会话漂移）、红线表 +2 行（D2 e2e / EXT-1 patch 定性）、 PR hardening 清单。
- **为什么**：第二 PR 机会作 9 月池备选需先查重留痕；简历句/PR 材料引用前， 数字必须与 raw JSON 键数定档一致（铁律 6 主张有据）。
- **关键数字**：tuned JSON 实际 M 档 = **18**（数值键 1–4096；第 19 键为 triton_version 元键）——`moe_perf/raw/EXP-015/configs_ep/E=30,N=1408, device_name=NVIDIA_GeForce_RTX_4090.json` 直接可数；SGLang 4090 config 存量 = 2（旧 fp8，与 vLLM 同源）。
- **产物**：`moe_configs/DEDUP.md`（SGLang 节）、commit 205c654 / 4bb075f、 README 红线表新行、EXP-015 §4 勘注。
- **下一步**：全仓"19 M 档"残留清扫与审计 findings 收尾（→ §22）。

## §22 审计收尾批次(2026-08-24)

- **做了什么**：应用外部审计已确认 findings（GPU 被另一实验占用，本批次全程无 GPU 运行）：①"19 M 档"残留清扫——README 台账 97 行、RESUME_EVIDENCE 98 行、EXP-015 §7 改 18 并附 8/24 勘正；§20 史料原文加勘注留痕不改叙事。 ②moe_perf/derived 与 ext1/derived 共 4 个 CSV 补 `# source` 注释行：改三个生成脚本（d1_analyze.py / d1_kernels.py / analyze_ext1.py）后纯 CPU 重算生成（nsys stats 从本地 .nsys-rep 重导出属 CPU 操作），重算值与既有版本逐行 diff 一致。③EXP-012 raw 补目录级 manifest.txt——bug2_curl.txt 无首行 provenance 由 manifest 统一登记，raw 本体一字未动；EXP-012 §4 表述改如实（含 repro 日志实际落位）。④b1_matrix/raw 补 manifest.txt，声明 provenance 权威 = runs.jsonl 行内字段按文件名前缀关联。⑤EXP-007 §7 登记 3 个 0 字节 tp2 1911 前缀空快照（疑 8100 端口未起抓空；runs.jsonl 无行引用，数据侧无影响）。⑥README EXP 索引补日期列 + 表前声明（关键数字统一见证据台账）。 ⑦新建 `docs/talk/TALK.md` 现行讲稿（整合 RESUME_EVIDENCE 防御 + analysis 口径，数字全带 EXP 锚）与 `docs/theory/` 两篇五节笔记（01 MoE dispatch 链路 / 02 PD KV 通路，实证节指 EXP-013/014/015 数字）。⑧HANDOFF §7 重写三行制式（HEAD/硬件占用/下一步）。
- **为什么**：审计闭环——数字与 raw 一致、provenance 全覆盖（铁律 4）、单一事实源（铁律 1）、面试材料按 STANDARDS §7 制式落位；raw 不可变与禁 GPU 两条硬约束全程遵守。
- **关键数字**：M 档定档 18（raw JSON 数值键直数）；4 个 derived CSV 重算与 committed 版本逐行一致（56.44% / 40.89% / 54.2-64.2% 等零漂移）。
- **产物**：本节所列文件 + 本 commit（审计收尾批次）。
- **下一步**：唯余用户侧动作——R0-6 线上简历排雷、D2 PR 本人 review + `git commit -s` + 提交；本批次无 GPU 补测欠账。

## §23 README 门面化批次（2026-08-25）

- **做了什么**：README 升级为 GitHub 门面级：顶部新增一句话定位、Headline 结果表（6 行，数字全部取自证据台账既有条目并带 EXP/文件指针）、图表区（复用 fig1/fig4/d1_fig1，新增 fig7 四臂饱和吞吐总览）、EXT-1 patch 代码导览（patch 原文节选 + 三段关联思路）、复现 Quickstart、目录树更新（补 moe_perf/ext1/p2pnccl_repro/analysis）、相关仓链接、红线表下新增「方法论：诚实度文化」三条。实验记录索引/证据台账/措辞红线表/硬约定/备份逐字保留，RESUME_EVIDENCE 指向红线表的锚点未破坏。新增 `pd_disagg/scripts/make_fig7_overview.py`：从 runs.jsonl 重算饱和吞吐（completed/wall_time_s，协议 v2 行 + gates.pass），输出 `figures/fig7_saturation_overview.png`（dpi 220，四臂固定配色沿用 make_figures.py，脚注 provenance）。全程无 GPU 运行，data/raw 未动。
- **为什么**：面试官 30 秒扫读需要数据/图/代码/方法论前置；同时守住 CORE 铁律 1（README 台账仍是唯一状态源，未删改）、铁律 6（每句量化主张带指针）与仓内硬约定 6（四臂一色到底，fig7 沿用 fig1-6 色序）。
- **关键数字**：fig7 从 runs.jsonl 重算的饱和 req/s 与台账 B1 行逐位一致（colocate 10.36/3.63/0.90 · replica2 15.58/7.00/1.78 · tp2 12.31/4.16/1.02 · pd1p1d 7.84/2.12/0.54）；README 无新造数字，D2 e2e 未上 headline（红线）。
- **产物**：README.md、pd_disagg/scripts/make_fig7_overview.py、 pd_disagg/figures/fig7_saturation_overview.png、本 commit。
- **下一步**：不变——用户侧 R0-6 线上简历排雷 + D2 PR 本人 review 与提交（见 HANDOFF §5）。

## §24 fork rebase + 社区空缺复验 + 公开仓瘦身调查(2026-08-29)

- **做了什么**：①A-1：外层 `/root/projects/vllm` 浅克隆先 `fetch --unshallow upstream`（42s）拉全历史——浅克隆边界恰是 cacc429f62，本地 git 算不出 merge-base，交接单的「676 提交/无分叉」实为 GitHub API 所算；unshallow 后验证无分叉（is-ancestor=YES）、落后 676、本地零独有提交，`main` ff 到 cacc429f62 并 push myfork（903a02192f..cacc429f62），`moe-config-4090-qwen15moe` rebase 到 cacc429f62，2 个 config JSON 保持 staged 未 commit（遵守「不用 agent 身份提交」）。②A-2：用 `git ls-tree -r cacc429f62` 精确复验「社区空缺」（排除 staged 干扰）——上游仍无 E=30、E=60,N=704 仍仅 MI300X、4090 仍仅 2 个 fp8，空缺成立；另核 PR_DRAFT 复核一「fused_moe.py 0 提交」实为 6 行插入（A_scale 0-D reshape，量化路径 bugfix，不触及 BF16 config），「无需重测」结论不变。DEDUP/LEDGER 更新后 commit c6fc416。③A-5：vllmExperience 瘦身调查——**澄清交接单 A-5 措辞**：`.sqlite` 从未进历史（.gitignore 的 *.sqlite 一直生效，本地 1.6GB 中约 1GB 是未跟踪的 sqlite/nsys-rep，不影响公开仓）；历史里仅 72MB `.nsys-rep`（4 文件，属 LEDGER「证据箱：nsys-rep 入 git，体积换可信度」设计）。公开仓 clone 实际 400MB（非 1.6GB）。
- **为什么**：D2 PR 需基于最新 upstream main（落后 676 提交）；「空缺」是主张，主张需复验（交接单 A-2）；A-5 瘦身需先核清「多少在历史、多少是本地」再决定是否动历史。
- **关键数字**：落后 676 提交、behind_by=0 无分叉；fused_moe.py 自 7aa248fcfe 起 6 行插入（@857 invoke_fused_moe_triton_kernel 内 A_scale ndim==0 reshape）；历史 .nsys-rep 合计 72.05MB / .git 400MB / 历史 .sqlite 0。
- **产物**：myfork main = cacc429f62、moe 分支 rebase、experiments commit c6fc416（DEDUP.md + LEDGER.md）。
- **下一步**：C 类基础设施（/root/projects/README.md 七仓总入口 + /root/work 无远端）；B 类待 GPU 空闲。**历史重写瘦身（filter-repo 移除 72MB .nsys-rep + force push）未做**——收益 18%、代价为全部 commit hash 失效（文档内 cc473de/7aa248fc 等 10+ 处引用）与 force push 破坏性，属用户决策，待其明确授权。

## §25 NCCL allreduce size 扫描(EXP-018)+ LEDGER D23/D24 补账(2026-08-29)

- **做了什么**：①补 llm-engine LEDGER 缺的 EXP-D23/D24 台账行（records/LAB_JOURNAL 早有，LEDGER 台账表只到 D22，违反铁律 1 单一事实源），补关键数字 + 红线 + 待办三处。②EXP-018：NCCL allreduce size 扫描——补 EXP-002 只测大消息(1M–512M)的缺口，分小消息(8B–1M, n=100)与大消息(1M–512M, n=20)两区间，写 `scripts/nccl_size_scan.sh` 采集脚本。
- **为什么**：交接单 B 类「NCCL allreduce size 扫描（双卡，本机能做）」；EXP-002 只测了大消息单点区间，decode 级小消息(8 KiB)的纯 NCCL 延迟从无实测，llm-engine#EXP-D22 的「88µs/次」是反推的完整开销。
- **关键数字**：延迟地板 **~14µs**（8 KiB decode 消息 13.8µs，与 EXP-002 的 GPU 间延迟 14.5–15.9µs 同阶）；大消息平台 **6.2 GB/s**（16M–256M 稳定 6.40–6.50）。**核心发现**：EXP-002 的「1.78 GB/s 带宽墙」复现不了，同二进制同 NCCL 2.28.9 同参数复测得 6.2 GB/s，差 3.5 倍——原因待查（EXP-002 provenance 未记 PCIe 运行态/系统负载，候选：8/21 B1 战役并发负载或 PCIe 未升 Gen4；实测 allreduce 运行时 PCIe 升 Gen4）。与 EXP-D22 对账：88µs/次 = 14µs 传输 + 74µs torch.distributed 调度/同步，强化「派发主导」归因。
- **产物**：`records/EXP-018_nccl_allreduce_size_scan.md`、`scripts/nccl_size_scan.sh`、`pd_disagg/hw/20260829T104705_allreduce_size_scan_{small,large}.txt`、EXP-002 §7 复测差异勘注、LEDGER EXP 索引 + R0-1 行更新。
- **下一步**：**待用户裁决**——EXP-002 的 1.78 GB/s 权威数字是否更新为 6.2，以及是否连带修订 EXP-005「prefill 零加速归因」与 RESUME_EVIDENCE「1.78GB/s 带宽墙」措辞（涉及多处下游结论，见 EXP-018 §7/§8）。其余 B 类（三算子 autotune 可跳、sglang router S02-S07 需 sglang venv 未装）。

## §26 1.78 vs 6.2 机制调查(EXP-019)+ LICENSE 补漏(2026-08-30)

- **做了什么**：①EXP-019：按「先 diff 环境、不先跑 bench」排查 EXP-002 的 1.78 vs EXP-018 的 6.2 差 3.5 倍的机制。NCCL_DEBUG=INFO 抓自报 + 逐行读 nccl-tests 计时代码 + 单点对照（默认 vs NCCL_SHM_DISABLE=1）。②补 vllmExperience 的 Apache-2.0 LICENSE（用户追认协议 + 点名漏了它）。
- **为什么**：交接单任务 1（最高优先）；用户押注「计时区混入非传输开销（与 reduce 同型）」，需用证据证伪或证实。
- **关键数字**：计时区内无 malloc 混入（证伪「固定开销」假设）；单点对照 SHM **3.96** vs Socket **0.76 GB/s**（差 5.2 倍）；NCCL 拓扑自报 NET 路径 1.2 / CPU 中转 24.0。判定**升级为真实环境差异**，候选 H1=PCIe 未升 Gen4/走了 Socket、H2=8/21 并发负载。
- **产物**：`records/EXP-019_nccl_bw_discrepancy_rootcause.md`、EXP-018 §7 闭环、EXP-002 §7 教训（provenance 缺 NCCL 环境变量+PCIe 运行态+DEBUG 日志）、LEDGER EXP-019 索引、LICENSE。
- **下一步**：任务 2（EXP-D22 分项账重做，用 14µs 锚）；1.78 复现实验设计已固化在 EXP-019 §8（扫 NCCL_P2P_LEVEL × NCCL_SHM_DISABLE 找复现档），不再阻塞任何对外主张。

## §27 nsys trace 诊断报告批次(2026-09-07,ncu-nsys-analysis)

- **做了什么**：用 ncu-nsys-analysis 工具链对既有 5 个 nsys rep 出 5 份诊断报告——pd_disagg 冒烟 trace(nsys_smoke)1 份 + EXP-014《D1 MoE decode 分解》的 4 个 MoE trace(bs1/bs32 × node/graphlevel)各 1 份;全部数字从 rep 现场 `nsys stats --force-export` 重导,未采用任何旁路 sqlite。附带处置旁路 sqlite:`pd_disagg/profiling/nsys_smoke.sqlite`(比 rep 新 17 天、来历不明、全仓 grep 零引用,报告下一步行动第 3 条建议清理)已删除;EXP-014 raw/ 下 4 个 sqlite 按 raw 不可变一律不动,只在 `d1_sweep_manifest.txt` 追注(旁路导出、来源未验证、分析时已从 rep 重导)。
- **为什么**：把躺在盘上的 nsys raw 变成可引用的结论(铁律 6 主张有据,证据前置);同时借两个 graphlevel rep 的元数据实查,验证 EXP-014 §7 graph-trace 口径踩坑记录是否成立。
- **关键数字**：①nsys_smoke 实为 4096³ matmul ×10 冒烟脚本 trace(非 vLLM 服务),GPU 窗口 idle 77.26% 的 99.65% 由单个 94.063 ms 冷启动懒加载空隙贡献(cuLibraryLoadData 等 API 交叠 48.278 ms/51.3%),稳态 sgemm 逐次 StdDev 仅 10.1 μs——77% 空闲是冷启动伪影不是稳态问题(`nsys_smoke_analysis.md`);②bs1 decode:gemvx 族 T=32.3% 第一大户、fused_moe 18.7%、GPU busy 96.4%(`d1_nsys_moe_bs1_analysis.md`);③bs32:fused_moe_kernel T=56.4% 一家独大、busy 99.0%,与 `moe_perf/derived/d1_kernel_share_bs32.csv` 及 EXP-014 既有结论同源一致(`d1_nsys_moe_bs32_analysis.md`);④两个 graphlevel rep 经 META_DATA_CAPTURE 实查 MODE=Graph:kernel 表仅 18.3 万/4.2 万行 vs node 版 517.7 万/134.6 万行,占比表整体降级「参考」,独立价值=graph 执行画像与 cudaGraphLaunch 均值 4.2× node 采集伪影对照(两份 `*_graphlevel_analysis.md`)——EXP-014 §7 的踩坑记录被 rep 元数据实证。
- **产物路径**：`pd_disagg/profiling/nsys_smoke_analysis.md`、`moe_perf/raw/EXP-014/d1_nsys_moe_bs1_analysis.md`、`moe_perf/raw/EXP-014/d1_nsys_moe_bs1_graphlevel_analysis.md`、`moe_perf/raw/EXP-014/d1_nsys_moe_bs32_analysis.md`、`moe_perf/raw/EXP-014/d1_nsys_moe_bs32_graphlevel_analysis.md`,以及 `moe_perf/raw/EXP-014/d1_sweep_manifest.txt` 追注。
- **下一步**：d1 同负载 ncu 复采两热点拿 ES——bs1 的 gemvx(T=32.3%)与 bs32 的 fused_moe_kernel(T=56.4%),按 G=ES×T 回填全局预估收益(两份 node 版报告的 P0 条目)。另记:HANDOFF §5 与 LEDGER D2 行的「PR 提交留用户」已过时——PR vllm-project/vllm#54372 已由用户本人于 2026-08-29 提交(gh 实查 OPEN 未合并,正文存档 `moe_perf/PR_BODY.txt`,commit 1252684);HANDOFF §5 本批次已勘,LEDGER 行在本批次改动授权范围外,留待下批次修。

## 2026-09-15 总览技术文档 + 缺口清扫 + 四实验补跑

**做了什么**：①写成 `docs/TECH_DOC_vllm_engineering.md`（原理 9 节 11 张 mermaid、项目说明、数据表、分析方法 7 条、140 道分类面试题、附录缺口审计/口径不一致/术语/补跑结果），挂进 README 与《怎么读》。②只读审计全仓 66 条缺口，文档滞后 14 处同步（PR #54372 状态、记录数、EXP-006/007/008 勘注回填、HANDOFF §8 编号、correctness 双口径括注）。③补跑四个挂账实验：EXP-020《NCCL 旋钮矩阵复现 1.78 GB/s》、EXP-021《NCCL allreduce dtype 扫描》、EXP-022《D2 大 M kernel A/B》、EXP-023《replica2@512 饱和复测（SAT_CONC=128）》，记录八节齐、raw 带 provenance。

**为什么（决策依据）**：用户要一份"原理 + 项目 + 数据 + 分析方法 + 分类面试题"的总览文档，并要求把仓里没跑完的东西清掉。审计发现真正阻塞对外主张的只有 NCCL 带宽一条（约 20 处"待复核"挂在它上面），故 EXP-020 优先；EXP-022/023 是 EXP-015/007 §7 明写的复测项；ncu ES 复采因主力机无计数器权限跳过。

**关键数字**：EXP-020 Socket 路径 1.51–1.70 GB/s 复现落窗 5/5，SHM 默认 9.07，但 SHM 亦见 1/28 次塌陷 2.1–2.3（H3）→ 定因不唯一；EXP-021 地板 13.5–14.4 µs 三 dtype 差 <7%，平台被 ±30% 运行间噪声盖住（未决）；EXP-022 tuned 在 M=512–4096 全段 EP −6.4~−14.0%、非 EP −2.8~−11.8%，8/8 > 2σ；EXP-023 replica2@512 conc128 20.87 vs 15.58（+34%），扩展效率 1.63×。PR #54372 gh 实查：OPEN、pre-run-check ×2 失败（缺 `ready` label / 作者 0 merged PR）、无人类 review。

**产物路径**：`docs/TECH_DOC_vllm_engineering.md`；`records/EXP-020~023`；`pd_disagg/hw/20260915T*`、`hw/derived/20260915T*`；`moe_perf/raw/EXP-022/`、`moe_perf/derived/20260915T0304_exp022_bigM_ab.csv`；`pd_disagg/results/b1_matrix/raw/20260915T03*`、`runs.jsonl` +3 行；新脚本 `scripts/nccl_knob_matrix{.sh,_analyze.py}`、`scripts/nccl_size_scan_dtype.sh`、`scripts/nccl_dtype_scan_analyze.py`、`scripts/nccl_shm_collapse_probe.sh`、`scripts/nccl_h2_concurrent_load_probe{,_min}.sh`、`moe_perf/d2_bigM_ab.sh`、`moe_perf/d2_bigM_analyze.py`、`pd_disagg/scripts/replica2_stack.sh`。

**下一步**：用户裁决 NCCL 带宽措辞（R0-1 停用 → 带路径引用？连带 EXP-005 §6 / RESUME_EVIDENCE / README 约 20 处）；用户去 vLLM Slack #pr-reviews 求 `ready` label；agent 侧：EXP-020 H3 长跑探针（≥50 轮带 PCIe 采样）、fig7 按 EXP-023 重算（排除污染行）；ncu ES 需采集主机。
