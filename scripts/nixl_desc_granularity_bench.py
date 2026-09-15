#!/usr/bin/env python3
"""EXP-026 · NIXL descriptor 粒度实验（把「描述符碎片化」从推断变成实测）

背景：EXP-009/011 与 EXP-020 附录 A 把 NIXL KV 通路的 0.26–0.27 GB/s 归因于
「descriptor ≈16 KiB 的碎片化小拷贝、固定开销主导」，但这一条一直是**推断**；
而 NIXL 的 `make_prepped_xfer` 有一个 `skip_desc_merge=False` 的默认参数，
意味着 **NIXL 默认会合并描述符**——若属实，则该推断需要修正。

本脚本用与 vLLM 同一条代码路径（prep_xfer_dlist + make_prepped_xfer，READ/pull，
mem_type=VRAM，backend=UCX）在两张 4090 之间测：

  A. 粒度曲线：固定传输总量 S，descriptor 大小 G ∈ {16K…S}，量有效带宽
  B. 布局：contiguous（描述符首尾相接，可合并）vs scattered（间隔一个 G，不可合并）
  C. 合并开关：skip_desc_merge ∈ {False(默认), True} —— 直接回答「NIXL 到底合不合并」
  D. 尺寸扫描（G 固定 16 KiB，S 变）→ 拟合 t = N_desc·(α + m/β) 里的 α 与 β

用法（两个进程）：
  CUDA_VISIBLE_DEVICES=0 python3 nixl_desc_granularity_bench.py --role P --workdir <dir>
  CUDA_VISIBLE_DEVICES=1 python3 nixl_desc_granularity_bench.py --role D --workdir <dir>

结果：D 侧把逐点结果写成 CSV（--out），并在 stdout 打印汇总表。
铁律 8：任何取不到的数据（telemetry/状态）一律抛错，不回退默认值。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

KIB = 1024
MIB = 1024 * 1024

# vLLM 实际口径（EXP-006/013 实测）：一个 descriptor = 16 token × KVH4 × D128 × 2B = 16,384 B
VLLM_DESC = 16 * KIB

# 主扫描：descriptor 粒度
GRANULARITIES = [16 * KIB, 64 * KIB, 256 * KIB, 1 * MIB, 4 * MIB, 16 * MIB]
# 尺寸扫描（G 固定为 vLLM 的 16 KiB），用于分离 α 与 β
SIZE_SWEEP = [8 * MIB, 32 * MIB, 128 * MIB]
ITERS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_agent(name: str, cuda_device: int):
    import torch  # noqa: F401  确保 CUDA 上下文先于 nixl 初始化
    from nixl._api import nixl_agent, nixl_agent_config

    torch.cuda.set_device(cuda_device)
    # 与 vLLM 完全同参：base_worker.py 在 backend 全为 UCX 时用
    # nixl_agent_config(num_threads=<默认 4>, capture_telemetry=True)。
    # telemetry 必须显式开启，否则 get_xfer_telemetry 抛 NIXL_ERR_NO_TELEMETRY
    # ——vLLM 开了它，所以 EXP-006/013 的 telemetry 数字是合法的。
    cfg = nixl_agent_config(num_threads=4, capture_telemetry=True)
    return nixl_agent(name, cfg)


def alloc_and_register(agent, nbytes: int, cuda_device: int):
    """分配 GPU 缓冲并注册为 NIXL VRAM region。返回 (tensor, base_addr)。"""
    import torch

    t = torch.zeros(nbytes, dtype=torch.uint8, device=f"cuda:{cuda_device}")
    addr = t.data_ptr()
    descs = agent.get_reg_descs([(addr, nbytes, cuda_device, "")], "VRAM")
    agent.register_memory(descs, backends=["UCX"])
    return t, addr


def build_offsets(total: int, gran: int, layout: str) -> list[tuple[int, int]]:
    """返回 [(offset, length)] 列表。

    contiguous: 首尾相接（i*gran）—— NIXL 有机会把它们合并成一个大传输
    scattered : 每隔一个 gran 放一个（i*2*gran）—— 中间有洞，不可合并
    """
    n = total // gran
    if n * gran != total:
        raise ValueError(f"total={total} 不是 gran={gran} 的整数倍")
    stride = gran if layout == "contiguous" else 2 * gran
    return [(i * stride, gran) for i in range(n)]


def run_one(agent, remote, local_addr, remote_addr, offsets, cuda_device,
            skip_merge: bool, iters: int, xfer_timeout: float = 120.0):
    """跑一个点：n_iters 次 READ(pull)，返回 (中位耗时 s, 遥测聚合)。"""
    # get_xfer_descs 的 tuple 形态要求 3 元组 (addr, len, device_id)——传 4 元组会被
    # 拒（"3-tuple list needed for transfer"）。reg_descs 那条路径接受 4 元组，两者不同。
    local_list = [(local_addr + off, ln, cuda_device) for off, ln in offsets]
    remote_list = [(remote_addr + off, ln, cuda_device) for off, ln in offsets]

    lh = agent.prep_xfer_dlist("", local_list, "VRAM", backends=["UCX"])
    rh = agent.prep_xfer_dlist(remote, remote_list, "VRAM", backends=["UCX"])
    idx = list(range(len(offsets)))

    times, teles = [], []
    for _ in range(iters):
        h = agent.make_prepped_xfer(
            "READ", lh, idx, rh, idx, b"", backends=["UCX"], skip_desc_merge=skip_merge
        )
        t0 = time.perf_counter()
        state = agent.transfer(h)
        # transfer() 返回投递后的状态：'PROC'（进行中）是正常值，'DONE' 表示同步完成。
        if state not in ("PROC", "DONE"):
            raise RuntimeError(f"transfer() 返回 {state!r}（期望 PROC 或 DONE）")
        while True:
            st = agent.check_xfer_state(h)
            if st == "DONE":
                break
            if st in ("ERR", "ERROR"):
                raise RuntimeError("传输失败：check_xfer_state == ERR")
            if time.perf_counter() - t0 > xfer_timeout:
                # 某些 UCX_TLS 组合会「选到了但传不动」——表现为状态永远停在 PROC
                # 且 GPU 利用率归零。必须有超时，否则整个扫描挂死（EXP-026 实测踩到）。
                raise TimeoutError(
                    f"传输超时 {xfer_timeout}s（状态停在 {st}）—— 该 UCX_TLS 组合不可用"
                )
            time.sleep(0.0005)
        dt = time.perf_counter() - t0
        tel = agent.get_xfer_telemetry(h)
        if tel is None:
            raise RuntimeError("get_xfer_telemetry 返回 None（铁律 8：取不到就报错）")
        times.append(dt)
        teles.append((tel.totalBytes, tel.xferDuration, tel.descCount))
        agent.release_xfer_handle(h)

    agent.release_dlist_handle(lh)
    agent.release_dlist_handle(rh)

    times.sort()
    med = times[len(times) // 2]
    tb = {t[0] for t in teles}
    dc = {t[2] for t in teles}
    if len(tb) != 1:
        raise RuntimeError(f"telemetry totalBytes 不一致：{tb}")
    xd = sorted(t[1] for t in teles)[len(teles) // 2]
    return med, tb.pop(), xd, dc.pop()


def role_p(args) -> None:
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    nbytes = args.region
    agent = make_agent("EXP026_P", 0)
    _t, addr = alloc_and_register(agent, nbytes, 0)
    (work / "P_meta.bin").write_bytes(agent.get_agent_metadata())
    (work / "P_ready").write_text(f"{addr},{nbytes}\n")
    log(f"P ready: addr={addr} region={nbytes/1e6:.1f} MB ... 等 D 结束")
    stop = work / "P_stop"
    while not stop.exists():
        time.sleep(0.2)
    log("P 收到 stop，退出")


def role_d(args) -> None:
    work = Path(args.workdir)
    t_wait = time.perf_counter()
    while not (work / "P_ready").exists():
        # 必须有超时：某些 UCX_TLS 下 P 侧会在 createBackend 处直接失败（NIXL_ERR_BACKEND），
        # 永远不会写出 P_ready——没有这个超时 D 会干等（EXP-026 实测踩到）。
        if time.perf_counter() - t_wait > args.wait_timeout:
            raise TimeoutError(
                f"等 P_ready 超时 {args.wait_timeout}s —— P 侧可能初始化失败，"
                f"看 {work}/P.log"
            )
        time.sleep(0.2)
    p_addr, p_nbytes = (int(x) for x in (work / "P_ready").read_text().strip().split(","))
    if p_nbytes != args.region:
        raise RuntimeError(f"P 的 region={p_nbytes} 与本地 {args.region} 不一致")

    agent = make_agent("EXP026_D", 0)
    _t, d_addr = alloc_and_register(agent, args.region, 0)
    remote = agent.add_remote_agent((work / "P_meta.bin").read_bytes())
    log(f"D ready: local={d_addr} remote_agent={remote} region={args.region/1e6:.1f} MB")

    if (work / "D_meta.bin").exists():
        raise RuntimeError("D_meta.bin 已存在，拒绝覆盖")
    (work / "D_meta.bin").write_bytes(agent.get_agent_metadata())

    if args.init_only:
        # 只做「UCX_TLS 这个组合能不能起来」的筛选：agent + 注册 + 远端握手
        log(f"INIT_ONLY OK: ucx_tls={os.environ.get('UCX_TLS','(unset)')} remote={remote}")
        (work / "P_stop").write_text("init_only\n")
        return

    rows = []
    vol = args.volume
    if 2 * vol > args.region:
        raise RuntimeError(
            f"scattered 布局需要 region ≥ 2×volume（{2*vol} > {args.region}）"
        )

    def write_rows() -> None:
        """落盘已有行（部分结果也写，铁律：失败档也是有信息的结果）。"""
        if not rows:
            log("无任何成功测量点，不写 CSV")
            return
        out = Path(args.out)
        if out.exists():
            out = out.with_suffix(".partial.csv")
        with out.open("w", newline="") as fh:
            fh.write(f'# provenance: env=sys sha=n/a cmd="CUDA_VISIBLE_DEVICES=0/1 python3 '
                     f'scripts/nixl_desc_granularity_bench.py --role P/D --workdir {args.workdir} '
                     f'--region {args.region} --volume {args.volume}" '
                     f'date={time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())} '
                     f'gpu="RTX 4090 x2" driver=610.57.04 exp=EXP-026 '
                     f'ucx_tls={os.environ.get("UCX_TLS", "(unset)")} '
                     f'path=prep_xfer_dlist+make_prepped_xfer/READ/VRAM/UCX iters={args.iters} '
                     f'xfer_timeout={args.xfer_timeout}\n')
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"写出 {out}（{len(rows)} 行）")

    try:
        _sweep_all(agent, remote, d_addr, p_addr, rows, vol, args)
    except Exception as exc:  # 首个失败点即中止（坏 UCX_TLS 会在第一点就挂）
        log(f"中止：{type(exc).__name__}: {exc}")
        write_rows()
        (work / "P_stop").write_text("aborted\n")
        sys.exit(2)

    write_rows()
    (work / "P_stop").write_text("done\n")
    log("D 完成")


def _sweep_all(agent, remote, d_addr, p_addr, rows, vol, args) -> None:
    # ---- A/B: 粒度 × 布局（skip_desc_merge = False，即 vLLM 的默认路径） ----
    for layout in ("contiguous", "scattered"):
        for g in GRANULARITIES:
            if g > vol:
                continue
            offs = build_offsets(vol, g, layout)
            med, tb, xd, dc = run_one(agent, remote, d_addr, p_addr, offs, 0,
                                      skip_merge=False, iters=args.iters,
                                      xfer_timeout=args.xfer_timeout)
            bw = tb / med / 1e9
            rows.append(dict(phase="A_granularity", layout=layout, gran_bytes=g, total_bytes=tb,
                             n_desc=len(offs), iters=args.iters, skip_desc_merge=0,
                             wall_median_s=round(med, 6), bw_GBps=round(bw, 4),
                             bw_per_desc_GBps=round(bw / len(offs), 6),
                             us_per_desc=round(med / len(offs) * 1e6, 2),
                             tel_total_bytes=tb, tel_xfer_s=round(xd, 6), tel_desc_count=dc))
            log(f"A {layout:11s} G={g//KIB:5d}KiB n_desc={len(offs):6d} "
                f"{med*1000:8.2f}ms {bw:6.3f} GB/s {med/len(offs)*1e6:7.2f} us/desc")

    # ---- C: 合并开关（决定性） ----
    for g in (16 * KIB, 1 * MIB):
        for layout in ("contiguous", "scattered"):
            offs = build_offsets(vol, g, layout)
            for sm in (False, True):
                med, tb, xd, dc = run_one(agent, remote, d_addr, p_addr, offs, 0,
                                          skip_merge=sm, iters=args.iters,
                                          xfer_timeout=args.xfer_timeout)
                bw = tb / med / 1e9
                rows.append(dict(phase="C_merge_switch", layout=layout, gran_bytes=g,
                                 total_bytes=tb, n_desc=len(offs), iters=args.iters,
                                 skip_desc_merge=int(sm), wall_median_s=round(med, 6),
                                 bw_GBps=round(bw, 4),
                                 bw_per_desc_GBps=round(bw / len(offs), 6),
                                 us_per_desc=round(med / len(offs) * 1e6, 2),
                                 tel_total_bytes=tb, tel_xfer_s=round(xd, 6),
                                 tel_desc_count=dc))
                log(f"C {layout:11s} G={g//KIB:5d}KiB skip_merge={int(sm)} "
                    f"{med*1000:8.2f}ms {bw:6.3f} GB/s tel_descCount={dc}")

    # ---- D: 尺寸扫描（G = 16 KiB 对齐 vLLM），用于 α/β 拟合 ----
    for s in SIZE_SWEEP:
        if s > args.region:
            continue
        offs = build_offsets(s, VLLM_DESC, "contiguous")
        med, tb, xd, dc = run_one(agent, remote, d_addr, p_addr, offs, 0,
                                  skip_merge=False, iters=args.iters,
                                  xfer_timeout=args.xfer_timeout)
        bw = tb / med / 1e9
        rows.append(dict(phase="D_size_sweep", layout="contiguous", gran_bytes=VLLM_DESC,
                         total_bytes=tb, n_desc=len(offs), iters=args.iters,
                         skip_desc_merge=0, wall_median_s=round(med, 6),
                         bw_GBps=round(bw, 4), bw_per_desc_GBps=round(bw / len(offs), 6),
                         us_per_desc=round(med / len(offs) * 1e6, 2),
                         tel_total_bytes=tb, tel_xfer_s=round(xd, 6), tel_desc_count=dc))
        log(f"D size={s//MIB:5d}MiB G=16KiB n_desc={len(offs):6d} "
            f"{med*1000:8.2f}ms {bw:6.3f} GB/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["P", "D"], required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--region", type=int, default=128 * MIB, help="注册的 region 字节数（scattered 需 ≥2×volume）")
    ap.add_argument("--volume", type=int, default=64 * MIB, help="每个测量点的传输总量")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--init_only", action="store_true",
                    help="只验证 agent/注册/握手能否在该 UCX_TLS 下成功，不做传输")
    ap.add_argument("--wait_timeout", type=float, default=120.0,
                    help="D 侧等 P_ready 的超时秒数")
    ap.add_argument("--xfer_timeout", type=float, default=120.0,
                    help="单次传输超时秒数（UCX_TLS 某些组合会选到但传不动）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.role == "D" and not args.out:
        ap.error("--role D 需要 --out")
    (role_p if args.role == "P" else role_d)(args)


if __name__ == "__main__":
    main()
