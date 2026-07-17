# Piper VLA Guard

AgileX Piper アーム上で VLA / OpenPI / pi0 形式のアクションチャンクを `piper_sdk` 経由で実行する前に、人間が確認できるようにする安全ラッパー UI です。

想定している実行経路は次のとおりです。

```text
OpenPI/pi0 action chunk
  -> parse + unit normalization
  -> one-step and workspace safety checks
  -> trajectory preview in UI
  -> human approval
  -> piper_sdk EndPoseCtrl / JointCtrl / GripperCtrl
  -> JSONL execution log
```

これはデプロイ時のガードであり、認証済みの安全コントローラではありません。衝突の完全回避、自己衝突の回避、力制御上の安全性、タスク成功を証明するものではありません。最初は必ず `dry_run=true`、ごく小さいステップ制限、そしてハードウェア非常停止を使えるオペレータがいる状態で始めてください。

## なぜこの形なのか

公式 Piper SDK は、ファームウェア V1.5-2 以降向けに `C_PiperInterface_V2` を提供しています。CAN 接続管理、非常停止、モード制御、モータ有効化、エンドエフェクタ姿勢制御、関節制御、グリッパ制御、アーム状態、エンド姿勢フィードバック、関節フィードバック、順運動学フィードバックを扱えます。

このプロジェクトで使う重要な Piper SDK の単位は次のとおりです。

- `EndPoseCtrl(X, Y, Z, RX, RY, RZ)` は、位置を `0.001 mm`、回転を `0.001 deg` で受け取ります。
- `JointCtrl(j1..j6)` は、関節角を `0.001 deg` で受け取ります。
- `GripperCtrl(gripper_angle, gripper_effort, ...)` は、グリッパ移動量を `0.001 mm`、力を `0.001 N/m` で受け取ります。
- `GetArmEndPoseMsgs()` と `GetArmJointMsgs()` は同じ SDK 単位を返すため、手動で変換する必要があります。

OpenPI の公式リモート推論フローでは、ポリシーをホストとポートで配信し、ロボット側クライアントが `actions` チャンクを受け取ります。このプロジェクトでは、その出力が実行前にシンプルなアクションスキーマへ変換されている前提です。

## インストール

### UI のみ / ドライラン

```bash
cd piper_vla
uv venv --python 3.11
uv sync --group dev --extra vision
./scripts/run_guard_uv.sh
```

### 実機 Piper ハードウェア

ロボット PC 上で、公式 Piper SDK の手順に従って先に CAN を有効化してください。その後、次を実行します。

```bash
cd piper_vla
uv sync --extra piper --extra vision --group dev
uv run piper-vla-guard-ui --config configs/safety.example.yaml --can can0 --real-hardware --dry-run
```

最初の確認では、設定ファイル内の `dry_run: true` を維持してください。UI でドライランをオフにするのは、状態読み取り、座標方向、ワークスペース制限を確認した後だけにしてください。

### OpenPI ポリシーサーバ

OpenPI リポジトリで、ファインチューニング済みポリシーサーバを起動します。例:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<your_config_name> \
  --policy.dir=<your_checkpoint_dir>
```

このリポジトリ内の Piper 用サーバは次でも起動できます。

```bash
cd ../docker_vla_share_clean/workspace/openpi_vla_proj
uv venv --python 3.11
uv sync
PIPER_POLICY_CONFIG=custom_scripts_piper/piper_training_params.yaml \
PIPER_POLICY_STEP=99999 \
./custom_scripts_piper/run_server_uv.sh
```

次に、ロボット環境へ OpenPI の軽量クライアントをインストールします。

```bash
cd piper_vla
./scripts/install_openpi_client_uv.sh
```

UI の OpenPI タブから `host:port` を指定して問い合わせできます。OpenPI 連携を使わず、アクション JSON を手動で貼り付けることもできます。

## アクションスキーマ

### OpenPI / robosuite OSC_POSE action

OpenPI の PiPER checkpoint から返る生 action には、`robosuite_osc_pose`
モードを使います。これは学習時の `env.step(action)` と同じ意味に寄せる
ためのモードです。

各行は次の形式です。

```json
[x, y, z, rx, ry, rz, gripper]
```

先頭 6 成分は robosuite `OSC_POSE` controller の正規化入力です。
`custom_scripts_piper/piper_assets/osc_piper.json` と同じく、位置成分は
`[-1, 1]` を `[-0.05m, 0.05m]` に変換してから安全チェックします。
回転成分はbase/world座標のaxis-angleで、`[-0.05rad, 0.05rad]` 相当です。
学習時のrobosuiteと同じく `R_target = R_delta @ R_current` で合成し、Euler角の
単純加算は行いません。

`gripper` は学習時の PiPER gripper actionと同じく `-1.0` が全開です。
白円柱checkpointは完全閉鎖へ叩き込まず、通常は `-0.1` 付近で円柱幅を
保持します。guardは `[-1.0, 0.0]` をsimの指先幅へ変換します。

### ベース座標系の Cartesian delta action

手書きの実機向け delta を直接与える場合は `delta_base_m_deg` を使います。

各行は次の形式です。

```json
[dx_m, dy_m, dz_m, droll_deg, dpitch_deg, dyaw_deg, gripper]
```

回転を省略して、次の形式も使えます。

```json
[dx_m, dy_m, dz_m, gripper]
```

`gripper` の値は正規化されています。`0.0` は開、`1.0` は閉を意味します。

例:

```json
{
  "actions": [
    [0.002, 0.000, 0.000, 0.0],
    [0.002, 0.000, 0.000, 0.0],
    [0.000, 0.000, -0.001, 1.0]
  ]
}
```

### エンドエフェクタの絶対目標

モードは `absolute_ee_m_deg` です。

各行は次の形式です。

```json
[x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg, gripper]
```

### 関節差分アクション

モードは `joint_delta_deg` です。

各行は次の形式です。

```json
[dj1_deg, dj2_deg, dj3_deg, dj4_deg, dj5_deg, dj6_deg, gripper]
```

このモードでは `JointCtrl` を使うため、コマンド送信前に目標関節角の制限を確認できます。

## 実装済みの安全チェック

- 1 ステップあたりの Cartesian delta 制限。
- 1 ステップあたりの回転 delta 制限。
- Piper/base 座標系でのワークスペース箱制限。
- 最小 z / テーブルクリアランス制限。
- Piper/base 座標系での任意安全平面制限。
- VLAループ開始時の実測姿勢からの最大合計移動量。
- グリッパ移動量制限。
- 関節アクションモードでの関節制限と関節 delta チェック。
- 関節フィードバックが取得できる場合の現在関節角マージン警告。
- 実行前のアーム状態 fault チェック。
- 実行直前の現在姿勢 / 関節状態の再確認。
- 実行前の手動承認チェックボックス。
- ロボットへコマンドを送らないドライランモード。
- チェック済みプランと実行ステップの JSONL ログ。
- 白円柱checkpointのdataset ID / step / prompt / fps / norm-stats hash照合。
- 全面単色、同一反復、取得失敗camera frameの推論前拒否。
- 校正済みhomographyによる224x224机ROIへの射影。
- 20 Hzの速度・加速度制限と、実測EE追従誤差によるhold / E-stop。
- EEだけでなく校正済みtool pointsに対する机面clearance確認。
- ready関節姿勢ゲートと、純VLA解放条件ファイル。
- 校正済み関節waypoint、全補間点FK、安全平面、高さ、関節速度、追従timeoutを使う`--auto-ready`。
- 手動承認付きhybrid pick（vision align、段階close、test lift、full lift）。

## 白円柱 real loop

白円柱用serverは次のidentityに固定されます。

```text
dataset_repo_id: local/piper_topdown_lift
checkpoint: pi0_piper_stack_lora/my_finetune_PBL3/30000
prompt: lift the white cylinder 10cm
fps: 20
```

`configs/pick_calibration.example.yaml` を実測値で埋めるまで、`hybrid` と
`pure-vla` の実機実行は拒否されます。通常は `observe`、`hybrid`、
`pure-vla` の順で進めます。`pure-vla` には画像感度、dry-run 20回、hybrid
8/10成功を記録したgate JSONが必要です。書式は
`configs/pure_vla_gate.example.json` を使い、各試験の実測結果だけを記録します。

policy serverは `(20, 7)` のaction horizonを返し、sim評価設定と同じくreal loopは既定で
先頭5 stepを実行してから2カメラを再取得・再推論します。各stepは直前の50 ms実行後の
実測EE/関節状態へdeltaをrebaseし、再度IKと安全検査を行います。JointCtrl補間も1 actionの
50 ms内に収め、必要関節速度が上限を超えるactionは遅延実行せず拒否します。

自動ready復帰は`--auto-ready`を明示した場合だけ有効です。PiPERには固定の電源投入姿勢が
ないため、実行時の実測関節角を経路の暗黙の始点にします。`pick_calibration.yaml`の
`ready_path_joints_deg`は通常は空のままでよく、その場合は現在の実測関節角から
`ready_joints_deg`へ直接移動します。直線的な関節補間が物理環境に適さない場合だけ、安全確認済み
の経由点を追加し、最後を学習readyにします。現在姿勢からreadyまでの全補間点についてPiper SDK
FKがworkspace、高さ、安全平面、tool/table envelopeを通る場合だけ、`--auto-ready --execute --yes`
を明示した実行で低速移動します。FKと実測が20 mm超不一致、または設定上のready座標がworkspace外なら、enable前に
拒否します。実機固有の障害物や自己衝突までは証明しないため、直接経路に物がないことは実機で
確認してください。

起動時がTeaching mode、E-stop、または1軸でもenable済みの場合、自動CAN takeoverは行いません。
enabled状態で制御modeを切り替えるとPiper内の保持済みJointCtrl targetが適用される可能性があるためです。
このアプリケーションはTeaching解除、E-stop resume、resetを自動送信しません。これらの復旧は
アームを物理的に支持したうえでメーカー手順に従って行い、`NORMAL`状態を確認してから開始します。
全軸disableかつSTANDBY状態から、公式SDK例と同じくまずSTANDBYのまま全軸enableし、0.5秒以上
無移動を確認します。その後、実測targetをseedしてからCAN MoveJとJointCtrlを待ち時間なしで連続送信し、
さらに1秒以上holdを監視します。EE/関節driftが設定値を超えた場合はready軌道へ進みません。

`arm-test` はZ・回転・gripperを抑止したXY最大0.5 mmの1 actionを、端末で`MOVE`と承認した
場合だけ実行します。停止は3種類を混同しません。

- 一時停止・手動確認・VLA異常終了: VLA actionを破棄し、実測関節角を`JointCtrl`目標として保持します。
  `MotionCtrl_1`、`EmergencyStop`、reset、motor disableは送りません。
- 通常終了: `shutdown_joints_deg`が設定され、その姿勢を実測確認できた場合だけ`DisablePiper()`を許可します。
  未設定時や姿勢不一致時は失電を拒否します。通常のreal loop終了時はdisableせずholdを維持します。
- 本当の危険: 人が明示操作したときだけ`EmergencyStop(0x01)`を送ります。Piperはダンピング下降する
  可能性があるため、UIにreset/resumeボタンは置きません。

保存観測の背景・画像感度比較には次を使います。

```bash
uv run piper-vla-replay \
  --observation-json logs/real_vla_debug/cycle_0005.json \
  --overhead-image logs/real_vla_debug/cycle_0005_overhead.png \
  --wrist-image logs/real_vla_debug/cycle_0005_wrist.png \
  --calibration configs/pick_calibration.yaml
```

## UI の流れ

1. ドライランを有効にした状態で UI を起動します。
2. Piper に接続するか、モックモードを使います。
3. 現在の姿勢と関節状態を読み取ります。
4. アクションチャンクを貼り付けるか、OpenPI に問い合わせます。
5. `Check trajectory` をクリックします。
6. テーブルと 3D 軌道プレビューを確認します。
7. 安全レポートが緑であれば、承認チェックを入れます。
8. 承認済みプランを 1 つ実行します。
9. UI の非常停止ボタンと物理 E-stop を常に使える状態にしておきます。

## 設定項目

まず `configs/safety.example.yaml` を編集してください。特に重要な値は次のとおりです。

```yaml
workspace_m:
  x: [0.20, 0.40]
  y: [-0.15, 0.15]
  z: [0.10, 0.30]
min_z_m: 0.10
safety_planes: []
max_step_xyz_m: [0.003, 0.003, 0.003]
max_step_rpy_deg: [1.0, 1.0, 1.0]
max_start_pose_drift_m: 0.002
max_start_rpy_drift_deg: 0.5
max_start_joint_drift_deg: 1.0
cartesian_execution_mode: joint_ik
workspace_floor_corners_m:
  - [0.20, -0.15, 0.10]
  - [0.40, -0.15, 0.10]
  - [0.40,  0.15, 0.10]
  - [0.20,  0.15, 0.10]
workspace_floor_margin_m: 0.005
dry_run: true
```

最初は非常に小さいワークスペースと小さいステップ制限から始めてください。ログと実機の移動方向を確認してから、少しずつ広げてください。

`safety_planes` は、実機の机・治具・侵入禁止領域が測定できてから追加します。各平面は Piper/base 座標系の半空間として扱われ、`dot(normal, target - point) >= margin_m` を満たす側だけを許可します。テーブル面だけなら、まずは `min_z_m` と `workspace_m.z` のほうを優先して狭く設定してください。

`workspace_floor_corners_m` はPiper/base座標で作業面の4隅を、外周に沿った順番で指定します。水平な安全平面として運用する場合は4点すべてに同じZを指定します。必要なら4点の異なるZから最小二乗で傾斜面を作ることもできます。4点のXY投影より外側、および`workspace_floor_margin_m`より低いEEと全tool pointを拒否します。4点の平面fit誤差が`workspace_floor_max_fit_error_m`を超えた場合も実行しません。

### 実機geometryの測定順序

未知の指先で机を触って指先offsetと机面を同時に推定すると循環するため、最初はEE原点からの
XYZ offsetをノギス/CADで確認できる固定プローブを使います。Teaching modeかつread-only feedbackで
4隅へ接触し、各`ee_pose_m_deg`と同じ物理点の`pixel_xy`を
`configs/calibration_samples.example.yaml`へ記録します。この記録中にarm enable、MoveJ、MoveLを
送ってはいけません。

次に、base座標が分かった固定点へ指先パッド中心を3姿勢以上で合わせます。最後に、開閉全域の
両指とgripper bodyを覆うEE-local XYZ bounding boxをノギスまたは検証済みCADで採寸します。
オフライン計算は次で行います（CAN接続やmotion commandはありません）。

```bash
uv run python -m piper_vla_guard.calibration_calculator \
  --samples configs/calibration_samples.yaml \
  --output logs/calibration_candidate.yaml
```

出力の`workspace_floor_corners_m`、`overhead`、`finger_center_offset_m`、`tool_points_m`を
`pick_calibration.yaml`へ転記します。`floor_max_fit_error_m <= 0.003`、
`finger_center_max_residual_m <= 0.002`を確認し、4点とは別の検証点でcamera-to-base誤差5 mm以下を
確認するまで`complete: false`を維持します。

## ROSを使わない白円柱pick

電源投入後のfeedbackが`NORMAL / TEACHING_MODE`の場合だけ、先輩コード
`test_ctrlPiperJoint_can0_2.py`と同じ初回遷移を明示的に使えます。

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/piper-vla-real-loop \
  --transport sdk \
  --attach-enabled-can \
  --vendor-teaching-bootstrap \
  --mode arm-test \
  --auto-ready \
  --execute --yes \
  --config configs/safety.example.yaml \
  --calibration configs/pick_calibration.yaml \
  --policy-host 127.0.0.1 --policy-port 8000 \
  --overhead-camera-source 4 --wrist-camera-source 6 \
  --save-observation-dir logs/real_arm_test_execute \
  --max-cycles 1 --chunk-size 1 --speed-pct 2
```

`BOOTSTRAP`、`READY`、`MOVE`の順に現場承認を要求します。bootstrapは
`EmergencyStop(0x01)`後に腕がダンピング下降し、先輩コードの関節窓
（`|J2|,|J3|<10deg`、`12deg<J5<45deg`）へ入ってから
`DisablePiper()`、`EmergencyStop(0x02)`、CAN/MoveJ、全軸enable、現在関節保持を
行います。`NORMAL / TEACHING_MODE`以外ではこの処理を実行しません。タイムアウト時は
disable/resumeへ進まずE-stop/damping状態に残すため、腕を支持してvendor手順で復旧してください。

すでに`NORMAL / CAN_CTRL / MOVE_J`で保持できている2回目以降は
`--vendor-teaching-bootstrap`を外し、`--attach-enabled-can`だけを使います。

`configs/pick_calibration.yaml` のピクセル4点、床面4隅、全tool point、指先中心offsetを実測し、最後に `complete: true` とします。現在のファイルは仮値を含み `complete: false` なので、実機pickは意図的に拒否されます。

まずモータへ送信しない全経路確認を行います。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra piper --extra vision --extra openpi \
  piper-vla-real-loop \
  --transport sdk \
  --mode hybrid \
  --calibration configs/pick_calibration.yaml \
  --auto-ready \
  --overhead-camera-source 0 \
  --wrist-camera-source 2 \
  --save-observation-dir logs/white_cylinder_dryrun \
  --max-cycles 200
```

校正値、保存画像、全IK目標、4隅面、移動方向を確認した後だけ、同じコマンドへ `--execute --yes` を追加します。promptは既定で `lift the white cylinder 10cm` に固定され、policy serverのdataset/checkpoint/prompt/fps/norm-stats identityが一致しなければ開始しません。hybrid modeはVLAで接近後にvisionで位置合わせし、グリッパclose直前に端末で `CLOSE` の承認を要求し、15 mm test liftと画像確認を経て合計100 mm liftします。

従来のUI経路でも、`cartesian_execution_mode: joint_ik` の場合は `Check trajectory` 時にIK関節列が表示され、承認後は `EndPoseCtrl` ではなくその `JointCtrl` 列を実行します。実機UIは `scripts/run_real_guard_uv.sh` で常にdry-runから開始します。

## 制限事項

- `joint_ik` は現在関節角をseedにしたPiper SDK FKベースの数値IKです。各目標と補間点を検査しますが、自己衝突や外部障害物の完全な幾何モデルではありません。`end_pose` compatibility modeを選ぶと従来どおり `EndPoseCtrl` になり、実行前の関節解検査はできません。
- カメラhomographyとtool geometryの機構は含まれますが、実機固有の測定値は自動生成されません。example値をそのまま実機へ使わないでください。
- 自己衝突モデルは含まれていません。より強い保証が必要な場合は、Piper URDF とシーン形状を使って MoveIt / Pinocchio / MuJoCo の衝突チェッカを追加してください。
- 力覚 / トルク接触安全は実装されていません。

## ファイル構成

```text
piper_vla_guard/
  actions.py          OpenPI / 手動アクションチャンクをパースする。
  config.py           YAML 安全設定を読み込む。
  safety.py           軌道プランを構築し、検証する。
  kinematics.py       Piper SDK FKとseed付き数値IK。
  piper_adapter.py    モックおよび実 piper_sdk アダプタ。
  policy_adapter.py   任意の OpenPI websocket クライアントラッパー。
  executor.py         ドライランおよび実行経路。
  ui_app.py           Gradio UI。
  logging_utils.py    JSONL ログ。
  pick_calibration.py camera/base/tool校正と画像射影。
  hybrid_pick.py      白円柱検出と手動承認付きhybrid pick。
  observation_replay.py 保存観測の画像感度比較。
configs/
  safety.example.yaml
  pick_calibration.example.yaml
examples/
  sample_actions.json
```
