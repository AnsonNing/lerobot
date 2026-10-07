# Streaming Flow V2 訓練交接

本目錄保存這個 fork 的 **Streaming Flow V2** 真機 joint-position 實驗設定。
新訓練以 `screw_gripper25_20261005` 的 **8/1、16/1** 為預設；兩個 camera
各自有一個獨立、完整可訓練的 ResNet18。這裡的設定與 upstream LeRobot 預設不同，
請使用本 repo 的程式與 checkpoint 配套 processor。

## 先看哪些檔案

| 檔案 | 用途 |
| --- | --- |
| [train_streaming_flow_v2.sh](../../scripts/train_streaming_flow_v2.sh) | 可換機器、路徑的訓練入口，讀取完整歷史 JSON；支援 dry run 與一步 smoke |
| [最新實驗設定](experiments/screw_gripper25_20261005/) | 兩個 variant 的原始 `train_config.json`、split、資料報告、結果、loss 記錄、checkpoint SHA256 |
| [PROVENANCE.json](experiments/PROVENANCE.json) | 歷史檔案的原始位置與逐檔 SHA256；快照保留原文 |
| [版本與架構說明](../../src/lerobot/policies/streaming_flow/README.md) | V2–V5 差異、模型與推論介面 |
| [環境版本](environment-observed-20261007.json) | 交接時實測的 Python / 套件 / FFmpeg 版本 |

Git 保存程式、設定與小型報告。資料 ZIP、影片、模型權重及 optimizer state 需要另外交接。
`experiments/*/{8_1,16_1}` 內的 JSON 是設定快照，缺少權重與 processor tensors，不能直接當作 inference checkpoint。
報告中的 `/home/ningan`、`/mnt/data/ningan` 是原機器位置；新入口不要求使用這些路徑。

原機器上最新 prepared dataset 位於
`/mnt/data/ningan/screw_datasets/cut_pull_screw_all_hover_gripper25_merged_20261005`，
完整訓練 outputs 位於
`/mnt/data/ningan/screw_datasets/outputs/streaming_flow_v2_screw_gripper25_20261005`，
已選定的 inference 交付目錄位於
`/home/ningan/screw_gripper25_streaming_flow_v2_checkpoints_20261005`。
接手前請決定另外傳送原 ZIP 或完整 prepared dataset；僅 clone repo 不會取得 demonstration。

## 環境與版本

歷史訓練以 Git `d011631afbec1fc033158ef47a1c6e499e49cf79` 加上當時的本機修改執行。
本次交接把 V2 的 per-dimension normalization、step scaling 開關及 gripper-close
weighting 一起版本化。不要只 checkout 舊 base commit；請保留這次交接 commit 的 SHA。

交接當天可執行測試的環境為 Linux x86_64、Python 3.12.12、PyTorch 2.11.0+cu128、
torchvision 0.26.0+cu128、torchcodec 0.11.1。原機器 Python 位於
`/mnt/data/ningan/lerobot_mimicgen/venv/bin/python`，實際 import 的程式來自本 repo 的 `src`。
版本記錄是在 2026-10-07 觀測，並非原訓練當下的完整 lockfile。

新機器可建立 Python 3.12 的環境並使用以下安裝方式。CUDA driver 必須支援選用的 PyTorch wheel，
系統也需要 `ffmpeg`、`ffprobe`。第一次訓練會下載 ImageNet ResNet18 權重。

```bash
git clone https://github.com/AnsonNing/lerobot.git
cd lerobot
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[training]' \
  -c examples/streaming_flow_v2/constraints-observed-20261007.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -c 'import torch, torchcodec, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__file__)'
ffprobe -version
```

constraints 只釘住關鍵套件，不是獨立 requirements 或完整依賴鎖。
其他平台或 CUDA 版本請依 `pyproject.toml` / `uv.lock` 安裝，再做 smoke 與影片解碼檢查；
不要混用不相容的 torch / torchvision / torchcodec。安裝指令未在全新的機器上驗證。

## 資料契約與前處理

最新原始檔為 `cut_pull_screw_all_hover_gripper_25_assemble_episode.zip`，SHA256：

```text
455e75f5cc6c943aacfa96f4c2f237de1b6dc2cf69d92686684c87d4251d4ec5
```

ZIP 包含 200 個各自獨立的單 episode LeRobot v3 dataset，不能直接把 ZIP 當作 `dataset.root`。
合併後是 200 episodes、21,845 frames、30 FPS；兩個來源/task label 各 100 episodes。
180 episodes / 19,714 frames 用於訓練；20 episodes / 2,131 frames 用於離線 checkpoint selection。
split seed 為 100000，於每個 task/source 的每十個連續 episodes 留出一個。
這組 selection 資料沒有再當作獨立 final test。

輸入 camera key 必須為 `observation.images.front`、`observation.images.side`。
原始影像 metadata shape 是 `[480,640,3]`，模型 tensor 是 `[3,480,640]`。
`observation.state` 和 `action` 均為 float32 的六維 **關節位置目標**，順序為：

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

這是馬達校正座標下的 `.pos` command，不是末端位姿或位移量。
部署要保留錄製時的 motor/gripper 校正、`use_degrees` 設定與 command units；
不要自行假設值為 radians。影像視角、方向與 robot 起始姿態也應與 demonstration 一致。

有兩種接手方式：

1. **拿到已合併的完整 dataset**：包含 `meta`、`data`、兩個 `videos` 目錄、
   `meta/stats.json`、`meta/gripper25_split_20261005.json`、`PREPARE_REPORT.json`。
   核對交接資料與報告後即可訓練，不需重新做時間裁切或重新 split。
2. **只有原始 ZIP**：先解壓、合併、檢查並計算 train-only normalization stats。

第二種方式的指令如下，所有路徑請換成自己的絕對路徑：

```bash
export ARCHIVE=/absolute/path/cut_pull_screw_all_hover_gripper_25_assemble_episode.zip
export EXTRACT_ROOT=/absolute/path/extracted_screw
export DATA_ROOT=/absolute/path/screw_gripper25_merged
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
sha256sum "$ARCHIVE"
unzip "$ARCHIVE" -d "$EXTRACT_ROOT"

# SOURCE_ROOT 是實際包含 episode_000 ... episode_199 的資料夾。
export SOURCE_ROOT="$EXTRACT_ROOT/cut_pull_screw_all_hover_gripper_25_each_episode"
python scripts/merge_screw_gripper25_dataset.py \
  --source-root "$SOURCE_ROOT" --output-root "$DATA_ROOT" --expected-episodes 200
python scripts/prepare_screw_histbalance_dataset.py \
  --root "$DATA_ROOT" --archive "$ARCHIVE" --split-file gripper25_split_20261005.json
```

合併入口拒絕覆蓋既有 output；不需要上一個 histogram dataset。
`--comparison-root` 可另外做兩批 action/state 的逐 episode 比較。
前處理會在指定 dataset 內寫入 metadata、split 與 stats，請先保留來源副本。
它檢查 finite action/state、重複 episode、index/frame/timestamp、episode/video 邊界及兩個影片的
codec、frame count、640×480 / 30 FPS，修正來源錯寫成 AV1 的 metadata 為實際 H264。
影片以 packet stream copy 合併，保留 action/state 與時間序列；重算合併後的 index statistics。
state/action normalization 只以 train episodes 計算，RGB 在模型內使用 ImageNet normalization。
準備完成後，把產生的 split 與 [保存的 split](experiments/screw_gripper25_20261005/DATASET_SPLIT.json)
比較，原始資料重現應完全一致。

這兩支前處理與 screw evaluator 針對單一 `data/chunk-000/file-000.parquet`、
單一 episode metadata parquet、兩個 front/side H264 videos 的六維資料設計，
並不是任意 LeRobot dataset 的通用轉換器。新資料若有不同分片、相機或 action 格式，需要先調整。

原始提供的序列已調整 hover 與 closing phase，本次沒有額外去重、加速、插值或 closing weighting。
CSV 中 closing phase 約 25%；以 normalized gripper 每幀減少超過 0.02 計算的 closing transitions
約 7.40%，兩者定義不同。靜止動作也可能對接觸與 assembly 有用，不能直接全部刪除。

## 兩個預設訓練設定

| 項目 | 8/1 | 16/1 |
| --- | --- | --- |
| 預測新的 actions | 8 | 16 |
| `chunk_size` | 9 | 17 |
| `n_action_steps` | 8 | 16 |
| `execution_horizon` | 1 | 1 |
| observation history | 2 frames | 2 frames |
| previous-action alignment | true | true |
| 初始 previous action | observed state | observed state |

`chunk_size` 包含 `a_(t-1)`，所以 8/1 要設 9，16/1 要設 17。
模型預測整段，但每次只執行第一個 command，接著取新 observation 重規劃。
兩個 variant 分別從頭訓練，不能把其中一個的 optimizer state 換 horizon 後續訓。

| 共用設定 | 值 |
| --- | --- |
| camera encoder | 每個 camera 獨立 ImageNet ResNet18，全部可訓練，GroupNorm |
| 影像前處理 | resize H×W = 240×320；train random crop = 228×304；eval/inference center crop |
| RGB / state / action | RGB `[0,1]` 後模型內 ImageNet normalize；state/action MIN_MAX；action `per_dim` |
| additional image augmentation | 關閉 dataset image transforms |
| batch / optimizer steps / seed | 16 / 20,000 / 100000 |
| optimizer | AdamW lr 1e-4、betas (0.9,0.999)、eps 1e-8、weight decay 1e-6、grad clip 1.0 |
| scheduler | cosine，500 warmup steps |
| AMP / EMA | AMP 開啟；EMA decay 上限 0.9999，`ema_update_after_step=0` |
| flow | 4 train points、adaptive frequency 與 step scaling 開啟，sigma0=0.4、k=10 |
| frequency | min 0.2、max 5.0，eval clamp 開啟、train clamp 關閉 |
| expert | conditional UNet，down dims [256,512,1024]，embedding 256，kernel 5，linear up/downsample |
| data loading | workers 4、prefetch 2、persistent workers |
| checkpoint / log | 每 5,000 steps 存；每 100 steps 記錄 `train_metrics.jsonl` |
| eval / tracking | `eval_freq=0`；訓練後另外做 offline evaluation；W&B / push_to_hub 關閉 |
| extra sample weighting | 無 |

完整可解析的設定以各 variant 的 `train_config.json` 為準，包含上表未列的參數。
seed 固定但 `cudnn_deterministic=false`，不同 GPU、套件或 worker 數不保證 bitwise 相同。

95% crop 在原圖座標最多從單邊切掉 24 pixels 高或 32 pixels 寬；center crop 每側是 12 / 16 pixels。
接手時先沿用 228×304，確認螺絲、gripper 與接觸區在 crop 範圍內。
不要先把 MP4 resize/crop，再讓模型重做一次。dataset 層的 transforms 關閉不代表模型沒有 random crop。

## 啟動、smoke、監看

在啟用環境並切到 repo 後，設定完整 prepared dataset 的路徑，以及新的 output 目錄。
可以先只印出指令，確認沒有指向舊機器：

```bash
export DATA_ROOT=/absolute/path/screw_gripper25_merged
export DATASET_REPO_ID=ningan/cut_pull_screw_all_hover_gripper25_merged_20261005
export SPLIT_FILE="$DATA_ROOT/meta/gripper25_split_20261005.json"
export RUN_ROOT=/absolute/path/runs/screw_gripper25
# 如未 activate，可另外 export PYTHON_BIN=/absolute/path/venv/bin/python

DRY_RUN=1 OUTPUT_DIR="$RUN_ROOT/8_1" bash scripts/train_streaming_flow_v2.sh 8_1
DRY_RUN=1 OUTPUT_DIR="$RUN_ROOT/16_1" bash scripts/train_streaming_flow_v2.sh 16_1
```

一步 smoke 會建立模型與完整 checkpoint，以 batch 2、workers 0 執行一次 optimizer update；
只用來檢查介面、影片解碼與 finite loss。它不代表資料 quality、收斂或真機成功。
請用獨立 smoke output，避免占用正式 run 路徑：

```bash
SMOKE=1 CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR="$RUN_ROOT/smoke_8_1" \
  bash scripts/train_streaming_flow_v2.sh 8_1
SMOKE=1 CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR="$RUN_ROOT/smoke_16_1" \
  bash scripts/train_streaming_flow_v2.sh 16_1

# 正式訓練：有兩張 GPU 時分別在兩個 terminal 執行；只有一張就依序跑。
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR="$RUN_ROOT/8_1" \
  bash scripts/train_streaming_flow_v2.sh 8_1
CUDA_VISIBLE_DEVICES=1 OUTPUT_DIR="$RUN_ROOT/16_1" \
  bash scripts/train_streaming_flow_v2.sh 16_1
```

入口直接讀取保存的完整 JSON，僅換 dataset root/repo id/train episodes、output 與 device。
允許 `STEPS`、`BATCH_SIZE`、`NUM_WORKERS`、`SAVE_FREQ`、`LOG_FREQ`、`DEVICE` 覆寫；
這些變更請記錄為新的實驗。`SPLIT_FILE` 是完整路徑。
`CONFIG_PATH` 可以改為其他保存的 8/1 或 16/1 JSON，但 split 也要一起換，不能只換資料路徑。
入口會檢查 feature 名稱/shape/FPS、split 不重疊與 horizon 一致；資料數值與 normalization stats
仍要先透過前處理檢查。`DATASET_REPO_ID` 是 local dataset 識別，不會替你上傳或下載資料。

```bash
tail -f "$RUN_ROOT/8_1/train_metrics.jsonl"
ls "$RUN_ROOT/8_1/checkpoints"
```

先估算容量：每個 variant 的 `pretrained_model` 約 0.85 GB；含 optimizer、scheduler、RNG 的
完整 training checkpoint 在這批原機器約 1.58 GB。兩個 variants 各四個存點約 12.7 GB，
另加交付副本、smoke、資料及暫存寫入空間。這是原機器量級，請以新機器實測為準。
要中斷前先確認最近完整 checkpoint，不要刪除選定、最後或暫停復原用的存點。

## 離線選 checkpoint 與交付

正式訓練不會自動跑 simulator evaluation。用保存的 validation split 分別評估 5k、10k、15k、20k：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for variant in 8_1 16_1; do
  for step in 005000 010000 015000 020000; do
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_screw_hover5_streaming_flow_v2.py \
      "$RUN_ROOT/$variant/checkpoints/$step/pretrained_model" \
      --root "$DATA_ROOT" --repo-id "$DATASET_REPO_ID" \
      --split-file "$(basename "$SPLIT_FILE")" --device cuda \
      > "$RUN_ROOT/validation_${variant}_${step}.json"
  done
done
```

evaluator 的 `--split-file` 是 `DATA_ROOT/meta/` 下的檔名，與 launcher 的完整 split 路徑不同。
不要用 `--max-batches` 的部分結果選模型；正式應包含全部 20 episodes / 2,131 frames。
`--flow-only` 可省去 action sampling，但正式交接也應保留 first-action / gripper diagnostics。
EMA、center crop 與 seeded flow sampling 保持一致；action 評估是 teacher-forced previous action，
不是 policy 自己連續 rollout 的結果。

依每個 horizon 的最低完整 held-out EMA flow loss 選 checkpoint，同時查看 first-action MAE、
gripper-close error、premature closing 與各來源結果。flow loss 的尺度受 horizon 影響，不能直接
用 8/1 與 16/1 的數值決定哪個真機更好。若用同一批 validation 選模型，就需要另留獨立 final test
或後續真機測試，才能報告泛化/task success。

最新原始實驗已完成兩個 20k 訓練；交付選擇如下：

| Dataset | Variant | Selected step | EMA flow loss | First-action normalized MAE | Gripper-close normalized MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| screw_gripper25_20261005 | 8/1 | 20,000 | 0.075691 | 0.039477 | 0.165951 |
| screw_gripper25_20261005 | 16/1 | 15,000 | 0.165102 | 0.021225 | 0.121445 |
| screw_histbalance_20261004 | 8/1 | 20,000 | 0.076607 | 0.039971 | 0.153474 |
| screw_histbalance_20261004 | 16/1 | 15,000 | 0.164411 | 0.021363 | 0.111572 |

最新兩個 dataset 的全部 200 episodes action/state 完全相同、split 相同，但兩個 camera 的影像不同；
這組比較是在改變視覺輸入。詳見各實驗 `PRIOR_DATASET_COMPARISON.json` 與 `TRAINING_RESULT.md`。
上表是原始歷史結果，重新訓練請依自己的完整 validation 選存點。

交付時複製 **整個 `pretrained_model` 目錄**，並保留 split、資料報告、selection metrics、Git SHA：

```bash
export DELIVERY_ROOT=/absolute/path/delivery/screw_gripper25
mkdir -p "$DELIVERY_ROOT"
# 以下 selected step 是原始實驗的選擇；新訓練請換成自己的結果。
cp -a "$RUN_ROOT/8_1/checkpoints/020000/pretrained_model" "$DELIVERY_ROOT/8_1"
cp -a "$RUN_ROOT/16_1/checkpoints/015000/pretrained_model" "$DELIVERY_ROOT/16_1"
cp "$SPLIT_FILE" "$DELIVERY_ROOT/DATASET_SPLIT.json"
cp "$DATA_ROOT/PREPARE_REPORT.json" "$DELIVERY_ROOT/DATASET_REPORT.json"
cp "$RUN_ROOT"/validation_*.json "$DELIVERY_ROOT/"
git rev-parse HEAD > "$DELIVERY_ROOT/lerobot_commit.txt"
find "$DELIVERY_ROOT" -type f -name '*.safetensors' -exec sha256sum {} + \
  > "$DELIVERY_ROOT/WEIGHT_SHA256.txt"
python scripts/bench_screw_hover5_streaming_flow_v2_inference.py "$DELIVERY_ROOT/8_1" \
  --root "$DATA_ROOT" --repo-id "$DATASET_REPO_ID" --device cuda
zip -r -0 "${DELIVERY_ROOT}.zip" "$DELIVERY_ROOT"
```

delivery 目錄必須是新的，避免 `cp` 把新 checkpoint 放進舊目錄。
每個 variant 至少保留七個檔案：`model.safetensors`、`config.json`、`train_config.json`、
`policy_preprocessor.json`、`policy_preprocessor_step_4_normalizer_processor.safetensors`、
`policy_postprocessor.json`、`policy_postprocessor_step_0_unnormalizer_processor.safetensors`。
本模型原生是 safetensors，不需要轉成 `.pt`。交付 inference 目錄不含 optimizer state。

要繼續未完成的訓練，另外複製完整 `checkpoints/<step>/`，包含 `training_state`，使用原本 horizon
與 training contract；不要把這裡的 JSON 快照誤當成可 resume 的 checkpoint：

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path=/absolute/path/full_checkpoint/010000/pretrained_model/train_config.json \
  --resume=true --dataset.root="$DATA_ROOT" --output_dir="$RUN_ROOT/8_1"
```

這個例子恢復原本 20k 計畫的 10k 存點。若要延長總 steps，需另外記錄 scheduler 與新的實驗設定，
不應當作原結果的逐位元重現。可換機器，但資料、split 與 processor contract 必須一致。

## LeRobot 真機 inference 交接

新機器安裝這個 fork，加上實際 robot/camera 需要的 hardware extras。
`lerobot-eval --env.type=libero` 是 simulator 指令；這批 SO motor-position checkpoint
不能套用 LIBERO 的 observation/action contract。
請用本 repo 的 `lerobot-rollout` 或自訂 loop 載入完整 checkpoint 目錄。

載入時使用保存的 pre/postprocessor：原圖 RGB 轉成 CHW float `[0,1]`，保留六維 `.pos` state，
再讓模型做 resize、center crop 與 ImageNet normalization。
`select_action()` 回傳 normalized action，經 saved postprocessor 轉回實際馬達目標後再送出。
不要重複做 MIN_MAX / ImageNet normalization。
每個新 episode 呼叫 `policy.reset()`；首次 previous action 來自 observed state，之後跟隨最後執行的 command。
若硬體安全層改寫或 clip action，需要把真正送出的 action 用 **保存的 action bounds** 轉回 normalized
座標，再更新 `policy.set_executed_action_state()`，不能直接把 raw motor positions 傳進去。

下面僅是接手者確認機器 readiness 後使用的模板，填入真實 calibration ID、port 與 camera 對應。
camera key 固定為 front/side，解析度 640×480、30 FPS；robot type 與 `use_degrees` 要對照錄製設定。

```bash
lerobot-rollout \
  --strategy.type=base --inference.type=sync \
  --policy.path=/absolute/path/delivery/screw_gripper25/8_1 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=REPLACE_WITH_COLLECTION_CALIBRATION_ID \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \
  --device=cuda --fps=30 --interpolation_multiplier=1 \
  --task='Pull the screw' --duration=30
```

原機器 RTX5090 的 saved processor → policy → postprocessor median latency 是
8/1 約 19.4 ms、16/1 約 35.4 ms（p95 約 20.3 / 35.5 ms），不含 camera capture 或 motor I/O。
30 Hz 一個週期只有 33.3 ms；16/1 在原機器的同步重規劃已超過此預算。
接手機器需量測整個 loop；降低控制 FPS 會改變 demonstration 的時間語義，不能視為同一設定。
兩個 checkpoint 都只做過 offline reload/action 檢查，沒有量測真機成功率。

## 其他保存的實驗與舊腳本

| 設定快照 | 與最新預設的差異 |
| --- | --- |
| `screw_histbalance_20261004` | 相同 action/state、split；camera image 不同；8/1、16/1 各 20k |
| `screw_hover5_20260930` | 較早的 hover5/gripper10 資料；ordinary 與 `_close_weighted` 分開保存，請查看 JSON 的 `sample_weighting` |
| `push_button_20260914` | 8/1 與 **16/8**，initial action 為 constant；計畫 50k，實際停在 40k checkpoint，選 20k；舊 normalization stats 含 validation，存在 feature-statistics leakage |

舊實驗的 JSON 保留當時的完整設定，不要用新 screw split 代替 button/hover split。
舊 button 的採集者確認兩批校正相同，但 gripper 數值分布與相機視角仍有差異。
詳見 [button 說明](../../scripts/README_push_button_streaming_flow_v2.md)。

`train_screw_*`、`train_push_button_*`、`run_screw_gripper25_training_job.sh`、
`finish_screw_gripper25_experiment.sh` 與 `finalize_screw_*` 是保留的原機器歷史入口，含固定路徑與日期。
換機器請使用本文件的 portable launcher、參數化前處理/evaluator 與交付步驟。
optional gripper-close weighting 的實作一起保留，但最新 histogram/gripper25 recipes 的 `sample_weighting=null`。

要在新機器重現 button 的 16/8，可直接使用其完整 JSON；此指令重新訓練，與部署既有 checkpoint 不同：

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path=examples/streaming_flow_v2/experiments/push_button_20260914/16_8/train_config.json \
  --dataset.root=/absolute/path/push_button_merged_260914 \
  --output_dir=/absolute/path/runs/button_16_8
```

該 JSON 已包含當時的 train episodes；只有相同 episode 編號的原 merged dataset 才能直接使用。
新的 button 資料要重新核對 split 與 train-only stats，不能照抄 episode 編號。
