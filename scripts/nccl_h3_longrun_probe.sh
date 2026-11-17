#!/bin/bash
# EXP-020 附录 C（预注册，判据见记录）：H3「SHM 路径间歇塌陷态」长跑探针。
# SHM 默认档大消息扫描 ×60 轮（float/half 交替），每轮 200ms PCIe/SM/mem/pstate/power 采样；
# 塌陷 = 该轮 16M-256M 平台 < 3.0 GB/s；抓到第 2 次塌陷提前停。
set -uo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
OUTDIR=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
SHA=717b683182
ARGS="-b 1M -e 512M -f 2 -g 2 -n 20"
Q="index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,pstate,power.draw"
MAXROUNDS=60
COLLAPSE_THRESHOLD=3.0
BUDGET_S=2700
mkdir -p "$OUTDIR/derived"
SUM="$OUTDIR/derived/${STAMP}_nccl_h3_longrun.csv"
[ -e "$SUM" ] && { echo "FATAL: $SUM 已存在"; exit 1; }
printf '# provenance: env=sys sha=%s cmd="bash scripts/nccl_h3_longrun_probe.sh (STAMP=%s)" date=%s gpu="RTX 4090 x2" driver=%s exp=EXP-020-appendixC threshold="collapse iff plateau(16M-256M mean) < %s GB/s; stop at 2 collapses"\n' \
  "$SHA" "$STAMP" "$(date -u +%FT%T+00:00)" "$DRV" "$COLLAPSE_THRESHOLD" > "$SUM"
echo "round,dtype,plat_GBps,avg_GBps,gen4_frac_loaded,sm_max,mem_max,pstates,verdict,file" >> "$SUM"

echo "=== H3 longrun STAMP=$STAMP $(date -u +%FT%TZ)  maxrounds=$MAXROUNDS"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
T0=$(date +%s); collapses=0
for ((i=1;i<=MAXROUNDS;i++)); do
  dt=float; (( i % 2 == 0 )) && dt=half
  out="$OUTDIR/${STAMP}_nccl_h3_${dt}_r${i}.txt"; pcie="$OUTDIR/${STAMP}_nccl_h3_${dt}_r${i}_pcie.csv"
  echo "# provenance: env=sys sha=$SHA cmd=\"nvidia-smi --query-gpu=$Q --format=csv -lms 200\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=$DRV exp=EXP-020-appendixC dtype=$dt round=$i" > "$pcie"
  nvidia-smi --query-gpu=$Q --format=csv -lms 200 >> "$pcie" 2>&1 & local_s=$!; sleep 0.5
  { echo "# provenance: env=sys sha=$SHA cmd=\"env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT $BIN -d $dt $ARGS\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=$DRV tool_sha=$SHA nccl=libnccl.so.2.28.9 exp=EXP-020-appendixC dtype=$dt round=$i"
    env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT "$BIN" -d "$dt" $ARGS 2>&1; echo "# exit_code=$?"; } > "$out"
  sleep 0.5; kill $local_s 2>/dev/null; wait $local_s 2>/dev/null
  if ! grep -q '# exit_code=0' "$out"; then echo "[$i] $dt 非零退出，跳过"; continue; fi
  read -r plat avg g4 smx memx pst verdict <<<"$(python3 - "$out" "$pcie" "$COLLAPSE_THRESHOLD" <<'PY'
import re,sys,csv
from statistics import mean
f,pf,thr=sys.argv[1],sys.argv[2],float(sys.argv[3])
t=open(f,errors='replace').read()
pts={int(m.group(1)):float(m.group(2)) for m in re.finditer(r"^\s+(\d+)\s+\d+\s+\w+\s+sum\s+-1\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s+",t,re.M)}
miss=[s for s in (1<<20,16<<20,256<<20) if s not in pts]
if miss: print("NA NA NA NA NA NA MISSING"); raise SystemExit
plat=mean(pts[s<<20] for s in (16,32,64,128,256))
avg=float(re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)",t).group(1))
rows=[r for r in csv.reader(open(pf)) if r and r[0].strip().isdigit()]
loaded=[r for r in rows if float(r[4].split()[0])>1000]
g4=sum(1 for r in loaded if r[2].strip()=='4')/len(loaded) if loaded else -1
smx=max(float(r[4].split()[0]) for r in rows); memx=max(float(r[5].split()[0]) for r in rows)
pst='|'.join(sorted({r[6].strip() for r in loaded}))
print(f"{plat:.3f} {avg:.3f} {g4:.3f} {smx:.0f} {memx:.0f} {pst} {'COLLAPSE' if plat<thr else 'ok'}")
PY
)" || { echo "[$i] $dt 解析失败"; continue; }
  echo "$i,$dt,$plat,$avg,$g4,$smx,$memx,$pst,$verdict,$(basename $out)" >> "$SUM"
  [ "$verdict" = "COLLAPSE" ] && collapses=$((collapses+1))
  echo "[$(date -u +%T)] r$i $dt plat=$plat gen4=$g4 sm=$smx mem=$memx $verdict  (collapses=$collapses)"
  [ "$collapses" -ge 2 ] && { echo "已抓到 2 次塌陷，提前停（预注册）"; break; }
  [ $(( $(date +%s) - T0 )) -gt $BUDGET_S ] && { echo "预算 ${BUDGET_S}s 用尽，停"; break; }
done
echo "=== H3 longrun end $(date -u +%FT%TZ)  实际轮数=$i  塌陷次数=$collapses"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
