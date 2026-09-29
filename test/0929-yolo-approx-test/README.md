# YOLO26 백본을 GEMM만 있는 DAG로: torch 실행 vs MPK 실행 (dynamic-multimodel-scheduler/test/0929-yolo-approx-test)

YOLO26 backbone(layer 0–9)의 구조(채널 수, 블록 반복, 분기와 합류, shortcut)를 실제 Ultralytics 모델에서
가져와 **GEMM만으로 된 DAG**를 만든다. 모델 크기 {n, m, l} × batch {1, 8} × 입력 {240, 320, 640}²의
18개 조합마다 DAG의 텐서 shape을 따로 계산하고, 각 DAG를 PyTorch(cuBLAS, eager / CUDA graph)와 MPK
(megakernel 하나)로 돌려 지연 시간과 nsys GPU 자원 지표를 비교한다. 결과 해석은 [RESULTS.md](RESULTS.md),
전체 표는 `results/summary.md`.

```
arch/gen_arch.py   Ultralytics yolo26{n,m,l}.yaml ──> arch/yolo26{n,m,l}.json   conv DAG (실제 모델과 수치 대조)
common/dag.py      arch + (batch, res)            ──> dags/<model>_b<batch>_r<res>.json   GEMM DAG 18개
01_torch/run.py    DAG → torch.mm / torch.addmm (eager, --graph)
02_mpk/run.py      DAG → MPK linear 태스크, megakernel 하나
sweep.py           18 조합 × 3 backend 실행 + nsys ──> collect.py ──> results/summary.md
```

## 1. conv DAG (`arch/`)

`arch/gen_arch.py`가 `yolo26<model>.yaml`로 만든 모델의 `model[0:10]`을 따라가며 Conv2d 하나당 노드 하나를
적는다. 노드: 이름(Ultralytics 모듈 이름), `in`(채널 방향으로 이어 붙이는 입력들, 보통 하나), `res`(출력에
더하는 shortcut), `k`, `s`, `cin`, `cout`.

- 뺀 것: BN(배포 시 conv 가중치에 접힘), SiLU, SPPF의 MaxPool 3개(항등으로 보고 cv2가 cv1 출력을 네 번 읽음),
  layer 10(C2PSA)과 head.
- C3k2.cv1은 출력을 `chunk(2)`로 나눠 쓰므로 출력 절반씩의 conv 두 개(`cv1.0`, `cv1.1`)로 적는다.
- 확인: BN/SiLU/MaxPool을 항등으로 바꾼 실제 layer 0–9(fp64)와, JSON만 보고 `F.conv2d`로 계산한 결과가
  세 모델 모두 같다(상대 차이 0). 노드 수는 n 37, m 47, l 75(Conv2d 33/43/71개 + C3k2 4개의 cv1 분할).

## 2. GEMM DAG (`common/dag.py` → `dags/`)

노드는 모두 `out[M, N] = A[M, K] @ W[N, K]ᵀ (+ res[M, N])` 하나다. 각 GEMM은 자기 버퍼 하나에 쓴다.
버퍼는 NHWC를 펼친 `[rows, C]`(한 행 = 한 픽셀)이고, GEMM은 A 버퍼의 **앞쪽 M·K개 원소를 `[M, K]`로**
읽으며 출력과 residual은 버퍼 전체를 `[M, N]`으로 본다. 실행기는 conv를 모르고 GPU에서는 GEMM만 돈다
(im2col, pooling, 정규화, 활성 함수 없음).

conv를 GEMM으로 바꾸는 규칙(이전 실험 `_0929-yolo-approximation`과 같은 규칙, 이 저장소에는 없음). 픽셀 묶기:
`[rows, C]`를 `[rows/g, g·C]`로 보면 GEMM 한 행에 연속한 픽셀 g개가 들어가고, dense 가중치
`W[g·Cout, g·Cin]`이면 출력 픽셀 하나가 그 묶음의 입력 픽셀 g개 모두에 의존한다.

| conv | GEMM (M, N, K) | conv 대비 FLOPs |
|---|---|---|
| 1×1 | (P/g, g·Cout, g·Cin), 채널이 적을 때만 g > 1 | g배 |
| 3×3, stride 1 | 같은 식, g = 8: 출력 픽셀이 연속한 입력 8픽셀에 의존(실제는 3×3 = 9픽셀) | 8/9 |
| 3×3, stride 2 | 입력 앞쪽 f·P행을 (P/g, g·f·Cin)으로(space-to-depth), g·f = 8, f = 4(행 수가 모자라면 2) | 8/9 |
| stem 3×3 s2 (3채널) | 그래프 입력 = 이미지 im2col `[P1, 32]`(27열 + 0 5열, 호스트에서 준비), 그 뒤는 1×1과 같음 | 같음 |
| concat → 1×1 | 조각마다 GEMM 하나, residual로 이어 누적: `out_j = part_j @ W_jᵀ + out_prev` | 같음 (정확한 변환) |
| shortcut 더하기 | 그 conv(의 마지막) GEMM의 residual | - |

- P = 출력 레벨의 행 수. MPK linear 태스크(sm_86 cutlass, 태스크 하나가 16×64 출력 타일)가 요구하는
  K % 128 = 0, N % 64 = 0을 만족하는 가장 작은 g를 쓴다(3×3은 8과의 최소공배수). concat 조각들은 같은 g.
- 레벨(P1–P5)의 행 수 = 픽셀 수를 `16 × lcm(그 레벨에 쓰는 GEMM들의 g)`의 배수로 올림. stride 2의 f가 이
  행 수에 따라 정해지므로 고정점 반복으로 푼다. 패딩 행은 0–2.4%.
- concat 조각은 늦게 만들어진 것부터 더한다: MPK는 fork와 join을 동시에 하는 층을 거부하는데(아래 3), 이
  순서면 문제되는 간선이 더 긴 경로에 함의되어 MPK가 지운다.

DAG 파일 하나에 버퍼 목록(이름, 레벨, C, rows, 유효 행), GEMM 목록(실행 순서; 이름, 원래 conv, 종류, A,
residual, f, g, M, N, K, 대응 conv FLOPs), 요약, `sha`가 한 줄에 하나씩 들어 있다. 두 실행기는 같은 파일과
같은 가중치(GEMM마다 `N(0, 1/K)`, seed 고정)와 입력을 쓴다. 요약은 `dags/summary.md`(아래 범위는 batch 1, 8 기준):

| | n | m | l |
|---|---|---|---|
| GEMM 수 (그중 residual) | 50 (20) | 62 (24) | 98 (40) |
| MPK 태스크 | 2,459 – 118,400 | 10,188 – 521,600 | 14,100 – 722,400 |
| 실행 GFLOP | 0.86 – 39.5 | 6.8 – 323 | 8.5 – 410 |
| 실행 / conv FLOPs | 1.74 – 2.09 (채널 16–64: g > 1인 1×1이 많음) | 1.05 – 1.23 | 1.06 – 1.25 |

batch 2, 4를 포함한 36개 조합의 GEMM 목록(이름, f, g, M, N, K, 피연산자, residual)은 이전 실험의 `yolo_approx.plan()`과
하나도 다르지 않다(대조함). `python common/dag.py --show n 1 640`이 한 조합의 GEMM 표를 보여 준다.

## 3. 실행과 측정

| backend | 실행 |
|---|---|
| `torch` | `01_torch/run.py`: GEMM마다 `torch.mm` / `torch.addmm` 한 번(cuBLAS, bf16). 피연산자 view는 미리 만들어 둠 |
| `torch_graph` | `01_torch/run.py --graph`: 같은 forward를 CUDA graph로 한 번 캡처해 replay |
| `mpk` | `02_mpk/run.py`: GEMM마다 MPK `linear` / `linear_with_residual` 태스크, grid (N/64, M/16). launch 한 번 = DAG 전체 |

- 측정: warm-up 뒤 GPU 시간 약 0.5초(`--target-ms`, 10–1000회)를 연달아 실행하고 회마다 CUDA event로 잰
  시간의 중앙값. MPK는 launch 하나(prepare + worker + scheduler 커널, persistent kernel 기동 포함)를 재고,
  launch 사이의 re-arm(`init_request_func`)은 측정 밖이다.
- 확인: torch는 같은 DAG의 fp32 실행과(`--graph`면 replay 결과가 eager와 비트 단위로 같은지도), MPK는
  첫 launch와 측정 뒤에 torch bf16 실행과 GEMM 출력마다 비교한다. 모든 GEMM에서
  `max|차이| / max|기준| < 5%`면 통과(실제로는 1–2%, bf16 반올림이 층을 따라 누적된 크기).
- MPK 커널은 DAG마다 한 번 컴파일한다(`02_mpk/out/<tag>/`, 1–4분). `compile.json`에 DAG의 sha를 적고,
  DAG가 바뀌면 다시 컴파일한다. 컴파일은 GPU를 MPK 초기화(할당)에만 쓰고 launch하지 않는다.

## 4. MPK 쪽 제약과 대응 (sm_86, mirage `mpk` 1f3338f9, mirage 소스는 수정하지 않음)

1. **linear 태스크의 K.** cutlass 커널(`use_cutlass_kernel=True`)은 K = 128이 맞고 K = 64는 틀린다. PTX
   커널(기본값)은 K = 128도 틀린다(0928 7.4절). → cutlass를 쓰고 K를 128의 배수로(위의 g).
2. **같은 rank의 reshape view가 틀린다.** `[1024, 32] → [256, 128]` 같은 2-D → 2-D view는 부모의 stride를
   물려받아(`src/kernel/view.cc` `set_view_strides`) 행 stride 32로 읽는다. `narrow`는 맞다. → 3-D view
   (`[M, K/C, C]`)를 거쳐 rank를 바꾸면 row-major stride가 새로 잡힌다(`common/mpk_exec.py`).
3. **fork/join 규칙.** 의존 분석(`annotated_graph.cc`)은 한 층이 여러 층에 출력을 주면서(fork) 입력이 여럿인
   층에 출력을 주는(join) 경우를 거부한다. C3k2 concat을 조각 GEMM 사슬로 바꾸면 y1이 다음 블록과 자기
   조각 양쪽에 들어가 이 경우가 된다. → 늦게 만들어진 조각부터 더한다(위 2절).
4. **태스크 수 한도.** launch 시작 때 모든 태스크를 worker 큐(8192칸 × 64 worker = 524,288)에 넣는다.
   l / batch 8 / 640은 722,400개라 넘는다. → 이 경우만 `persistent_kernel.cuh`의 큐 길이를 16384로 바꾼 복사본을
   `02_mpk/out/<tag>/include_overlay/`에 두고 컴파일 명령의 `-I` 맨 앞에 넣는다.
5. **반복 실행.** test_mode에서 두 번째 launch는 일을 하지 않는다(0928). `compile()` 뒤에는
   `init_request_func`가 없고 `load_mpk_kernel()`로 불러오면 있다. launch마다 이것을 먼저 부르면 매번 전체를
   다시 계산한다(출력을 0으로 지우고 확인). 그래서 컴파일은 별도 프로세스(`--compile-only`), 측정은 불러온
   커널로 한다.
6. **view 간선은 층 단위 배리어.** view로 읽는 소비자는 생산자 전체를 기다리고, 같은 배치로 읽는 간선만
   타일 단위 이벤트다. 3×3과 stride 2 GEMM은 입력을 view로 읽으므로 대부분 층 단위로 동기화된다.
7. **GPU를 독점해야 한다.** 같은 GPU에서 다른 프로세스가 CUDA 작업(할당, 복사, 커널)을 하는 동안 MPK를
   launch하면 멈추거나 illegal memory access로 끝난다. 재현: 다른 프로세스가 할당/복사/matmul을 반복하는
   중에 n_b8_r320을 측정하면 10번째 launch 즈음에서 멈춰 300초 타임아웃까지 끝나지 않았다(같은 명령을
   단독으로 돌리면 2000회 launch가 통과). 이전 실험의 컴파일 확인 실패 6건 가운데 5건이 이것이었다. 병렬
   컴파일 프로세스들이 같은 GPU에서 버퍼를 만들고 MPK를 초기화하는 동안 다른 프로세스의 확인 launch가
   돌았다(실패한 n_b8_r320과 통과한 n_b2_r640은 GEMM shape이 모두 같다). n_b8_r320, m_b1_r640, m_b2_r320,
   m_b4_r320, l_b8_r240은 GPU를 혼자 쓰면 통과한다. **m_b8_r240은 예외다**: GPU를 혼자 써도 첫 launch에서
   매번 illegal memory access가 난다. task graph의 텐서 접근 범위, linear 커널 variant(K, stride), 이벤트
   표는 정상이었고(CPU에서 확인), 행 수와 f가 같은 l_b8_r240은 통과한다. 원인은 아직 모른다.
   → 컴파일은 launch하지 않고, 실행은 GPU마다 lock 파일(`02_mpk/out/.gpu<N>.lock`)을 잡으며, `sweep.py`는
   GPU 하나에 작업 하나만 돌린다.
8. **초기화 실패가 조용히 지나간다.** 런타임의 `gpu_malloc`은 `cudaMalloc` 결과를 확인하지 않는다
   (`persistent_kernel.cuh`). 할당이 실패하면 나중에 illegal memory access로 나타난다. → 초기화 직후
   `cudaGetLastError`(런처와 같은 `libcudart.so.12`)를 확인해 바로 오류로 만든다(`mpk_exec.check_cuda_error`).
9. **nsys 아래에서 멈춘 MPK는 GPU를 막는다.** mpk m_b2_r320은 nsys 없이는 측정과 확인을 통과했지만, nsys
   GPU metrics 수집(ga10x) 중 첫 launch 뒤 `synchronize()`에서 멈췄다. 당시 스윕은 타임아웃(900초)으로
   상위 스크립트만 종료하고 멈춘 커널이 남은 GPU에 다음 작업을 올렸다. 그 작업은 드라이버 안에서 멈춰 SIGKILL도
   듣지 않았고, GPU 6은 새 CUDA 프로세스를 받지 못했으며 `--gpu-reset`도 실패했다(2026-09-29, 호스트 재부팅
   필요). → 타임아웃을 없애고 MPK nsys를 맨 마지막 단계로 옮겼다. 원인(nsys의 샘플링이 방아쇠인지)은 확인하지
   못했다. 그 전의 MPK nsys 수집 약 18건은 정상이었다.

torch 쪽: `import ultralytics`/모델 생성이 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정하고, torch 2.7.1+cu118에서는
이 변수가 있으면 cuBLAS 호출마다 호스트 시간이 약 70 µs 늘어난다(이전 실험). 이 실험의 실행기는
ultralytics를 import하지 않는다(`arch/gen_arch.py`만 쓰고, 끝나면 변수를 되돌린다).

## 5. 실행

```bash
source test/0929-yolo-approx-test/env.sh      # 저장소 루트에서; CUDA 12.0 nvcc, common/, GPU 6, $PY
cd $T0929G
$PY arch/gen_arch.py                    # (한 번) Ultralytics → arch/yolo26{n,m,l}.json, 실제 모델과 대조
$PY common/dag.py                       # 18개 DAG → dags/ (GPU 불필요, 바뀌지 않은 파일은 그대로)
$PY common/dag.py --show m 4 320        # 한 조합의 GEMM 표

# 개별 실행 (dags/m_b4_r320.json)
$PY 01_torch/run.py --model m --batch 4 --res 320 [--graph]
$PY 02_mpk/run.py --model m --batch 4 --res 320 --compile-only   # 02_mpk/out/m_b4_r320/
$PY 02_mpk/run.py --model m --batch 4 --res 320                  # 불러와서 확인 + 측정 + 확인

# nsys 자원 사용량 (두 metric set을 차례로)
$PY common/nsys_metrics.py run --out /tmp/x --range mpk_run -- $PY 02_mpk/run.py --model m --batch 4 --res 320 --no-check

# 전체 (18 조합 × 3 backend, GPU 6, 세션과 분리), 표 만들기
setsid nohup $PY sweep.py --gpus 6 > results/logs/sweep.nohup.out 2>&1 < /dev/null &
$PY collect.py                          # results/summary.md, results/summary.csv
```

`sweep.py`: (0) `common/dag.py`로 DAG 갱신, (1) DAG가 바뀐 MPK 커널 컴파일(nvcc 8개 동시, launch 없음),
이어서 GPU마다 작업 하나씩, 큰 조합부터:
(2) 모든 backend의 측정과 확인(nsys 없음) → `results/<backend>/<tag>/run.json`,
(3) torch, torch_graph의 nsys 두 번 → `results/<backend>/<tag>/nsys/metrics.json`,
(4) MPK의 nsys를 맨 마지막에, 측정이 통과한 조합만. nsys 아래에서 MPK launch가 멈춰 GPU가 복구되지 않은
적이 있어서(아래 4-9) 핵심 결과가 다 모인 뒤에 돌린다.
타임아웃은 없다: 작업이 멈추면 스윕도 거기서 멈추고, 사람이 보고 정리한다. 실패한 측정은 한 번 더 시도하고
`results/logs/sweep.log`에 남긴다. nsys sqlite(회당 최대 약 1 GB)는 요약한 뒤 지운다(`--keep-reports`로 보존).
명령별 로그는 `results/logs/`.

## 6. nsys 지표 (`common/nsys_metrics.py`)

한 nsys 세션은 metric set 하나만 모으므로 같은 명령을 두 번 돌린다. 두 번 모두 CUDA 커널 trace를 켠다
(CUDA graph는 `--cuda-graph-trace=node`로 노드 단위).

| 알고 싶은 것 | 지표 (set) |
|---|---|
| SM | SM Active, SM Issue (ga10x), SM Throughput, SM Issue Active, Active Thread Groups in SM (gfxt) |
| warp | Compute Warps in Flight, Unallocated Warps in Active SMs (ga10x), Warps Eligible (gfxt) |
| tensor core | Tensor Active (ga10x) |
| register | CS Register Allocation (gfxt, 레지스터 파일 중 할당된 %), 커널별 registers/thread (trace) |
| shared memory | CS Shared Memory Allocated (Sync) (gfxt), 커널별 static + dynamic shared memory/block (trace) |
| L1 / L2 | L1 Throughput, L1 Hit Rate, L2 Throughput, L2 Hit Rate, L2 Hit Rate from L1 (gfxt) |
| DRAM | DRAM Read/Write Throughput (ga10x), VRAM Throughput (gfxt) |
| 이론 occupancy | 커널별 block 크기, registers, shared memory로 sm_86 규칙에 따라 계산 (제한 요인 포함) |

값은 GA10x 최대치 대비 %다. `metrics.json`의 `range`는 측정 루프 NVTX 구간의 평균, `busy`는 그 구간에서
커널이 실행 중인 샘플만의 평균, `p90`. 표는 `range`를 쓴다. 읽을 때 주의:

- MPK의 worker CTA(64개 × 128 thread, shared memory 약 98 KB → SM당 1개, 이론 occupancy 8%)와 scheduler
  CTA(72개 × 32 thread)는 일이 없어도 SM에 상주하며 폴링한다. MPK의 SM Active, Unallocated Warps는 일의
  양과 무관하게 높고 SM Issue에는 폴링 명령이 섞인다. 비교에 쓸 것은 Tensor Active, DRAM/L2 throughput,
  할당된 register/shared memory다. 커널 표의 시간 가중 평균은 동시에 도는 worker와 scheduler를 섞는다
  (커널별 값은 `kernels`).
- 샘플 간격이 50 µs(20 kHz)라 이보다 짧은 커널(작은 설정의 torch eager)은 `busy`로 걸러지지 않는다.

## 7. 파일

| 파일 | 내용 |
|---|---|
| `arch/gen_arch.py`, `arch/yolo26{n,m,l}.json` | Ultralytics → conv DAG, 실제 모델과 대조 |
| `common/dag.py`, `dags/*.json`, `dags/summary.md` | conv DAG + (batch, res) → GEMM DAG, 가중치/입력 생성 |
| `common/torch_exec.py` | DAG → `torch.mm`/`addmm` (`TorchDag`) |
| `common/mpk_exec.py` | DAG → MPK PersistentKernel (view 우회, 큐 길이 overlay, 초기화 오류 확인) |
| `common/mpk_common.py` | MPK 환경 설정(nvcc 12.0, c++17 고정), 컴파일/export, test_mode 파라미터 (0928/tools에서 가져옴) |
| `common/runutil.py`, `common/nsys_metrics.py` | 공용 인자, 측정 루프, 비교 / nsys 두 번 실행과 집계 |
| `01_torch/run.py`, `02_mpk/run.py` | 두 테스트 (`02_mpk/out/<tag>/`: task graph JSON, 생성된 .cu, launcher .so, `compile.json`) |
| `sweep.py`, `collect.py` | 전체 실행, 표 |
| `results/` | `summary.md`, `summary.csv`, backend별 `run.json` / `nsys/metrics.json`, `logs/` |

이전 실험 `_0929-yolo-approximation`(이 저장소에는 없음)과 비교하면 규칙과 GEMM shape은 같고, 다음이 다르다. 구조를 JSON으로
고정해 실제 모델과 대조한다. 조합별 DAG를 파일로 두고 실행기는 GEMM만 안다. torch eager는 view를 미리
만들어 호출당 Python 시간이 줄었다(n_b1_r240: 1.39 → 0.59 ms, CUDA graph는 0.220 ms로 같음). cuDNN conv
기준선은 뺐다. 컴파일과 실행을 나누고 GPU 독점을 지킨다.
