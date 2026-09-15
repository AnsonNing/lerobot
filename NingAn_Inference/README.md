# NingAn checkpoint → SO-101 follower

這兩份 checkpoint 是 `streaming_flow_v2`，不是 SmolVLA。請使用這個 repo
內的 `lerobot-rollout`，不要用 `lerobot-record`；後者只接受 leader/keyboard
teleoperator，沒有 policy inference 路徑。

## 執行前檢查

1. 此 inference launcher 已相容 Python 3.10 以上。它透過 `PYTHONPATH` 直接使用
   repo source，因此不需要把整個（metadata 仍標示 Python 3.12）的 LeRobot
   package 重新安裝進 Python 3.10 環境。
2. 確認 `/dev/ttyACM0`、camera index 4 與 6 是正確裝置。
3. 必須使用蒐集訓練資料時同一套 SO-101 calibration。
4. 先觀察 follower 回報的第六維 gripper position。checkpoint 的訓練範圍約為
   `1.61..3.05`；如果目前回報的是 `0..100` 百分比尺度，不可以直接執行，
   需先找回訓練資料使用的 gripper calibration/表示方式。
5. 清空手臂周圍，準備好斷電或急停。第一次只跑 10 秒。

## 選擇或更換 calibration JSON

這裡使用的是馬達 calibration JSON，不是 checkpoint 的模型 `config.json`。
若不指定檔案，LeRobot 會依 `--robot-id` 從下列位置載入：

```text
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot-id>.json
```

要明確換成另一個 calibration 檔，先用 inspect mode 驗證：

```bash
PYTHON_BIN=/home/allenchou0708/miniconda3/envs/NingAn/bin/python \
DEVICE=cpu ./run_ningan_inference.sh \
  --inspect-only \
  --calibration-file=/完整路徑/my_right_arm.json
```

程式會自動把 `my_right_arm.json` 的檔名 `my_right_arm` 當成 robot ID。
連線時若出現 calibration 提示：

- 按 `Enter`：把選定 JSON 的 calibration 寫入馬達並使用它。
- 輸入 `c` 再按 `Enter`：重新做 calibration，完成後會覆寫選定 JSON。

建議要保留舊設定時，先備份 JSON，再執行重新 calibration。

只連接硬體、讀取 joint positions、不載入 policy 且不送 action：

```bash
PYTHON_BIN=/path/to/python3.10 DEVICE=cpu ./run_ningan_inference.sh --inspect-only
```

## 執行

啟動腳本預設使用 `pretrained_model_8_1`、CUDA、10 秒，以及每個 control tick
最多移動 5 units 的安全限制：

```bash
cd /path/to/NingAn/lerobot/NingAn_Inference
./run_ningan_inference.sh
```

這台電腦現有的 Python 3.10 NingAn environment 可明確指定為：

```bash
PYTHON_BIN=/home/allenchou0708/miniconda3/envs/NingAn/bin/python \
DEVICE=cuda ./run_ningan_inference.sh
```

目前若 console command 尚未安裝，腳本會直接以 repo source 和 typed config
執行，因此不需要 `lerobot-rollout` 出現在 `PATH`，也不受舊版 draccus 的
polymorphic CLI parsing 問題影響。

切換到另一份 checkpoint：

```bash
CHECKPOINT=../../checkpoint/pretrained_model_16_8 \
  ./run_ningan_inference.sh
```

常用覆寫：

```bash
PYTHON_BIN=/path/to/python3.10 DEVICE=cuda DURATION=30 \
MAX_RELATIVE_TARGET=3 ROBOT_PORT=/dev/ttyACM0 \
FRONT_CAMERA=4 SIDE_CAMERA=6 ./run_ningan_inference.sh
```

`pretrained_model_8_1` 預測 8 steps、每 1 step 重規劃；
`pretrained_model_16_8` 預測 16 steps、每執行 8 steps 重規劃。

這個 V2 checkpoint 沒有 text conditioning，所以 `--task="Push the button"`
不會改變模型輸出；保留它是為了 rollout metadata 與未來相容性。

## 只推論，不錄影

預設的 `strategy.type=base` 只把 action 送至 follower，不建立 dataset。
若要同時記錄 rollout，應改用 `strategy.type=sentry`，而且
`dataset.repo_id` 的 dataset 名稱必須以 `rollout_` 開頭；本機目錄要放在
`dataset.root`，不能把絕對路徑塞進 `dataset.repo_id`。
