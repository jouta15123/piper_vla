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
回転成分は `[-0.05rad, 0.05rad]` 相当として度に変換します。

`gripper` は学習時の PiPER gripper action と同じく `-1.0` が開、
`1.0` が閉です。

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
- 初期姿勢からの最大合計移動量。
- グリッパ移動量制限。
- 関節アクションモードでの関節制限と関節 delta チェック。
- 関節フィードバックが取得できる場合の現在関節角マージン警告。
- 実行前のアーム状態 fault チェック。
- 実行直前の現在姿勢 / 関節状態の再確認。
- 実行前の手動承認チェックボックス。
- ロボットへコマンドを送らないドライランモード。
- チェック済みプランと実行ステップの JSONL ログ。

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
dry_run: true
```

最初は非常に小さいワークスペースと小さいステップ制限から始めてください。ログと実機の移動方向を確認してから、少しずつ広げてください。

`safety_planes` は、実機の机・治具・侵入禁止領域が測定できてから追加します。各平面は Piper/base 座標系の半空間として扱われ、`dot(normal, target - point) >= margin_m` を満たす側だけを許可します。テーブル面だけなら、まずは `min_z_m` と `workspace_m.z` のほうを優先して狭く設定してください。

## 制限事項

- Cartesian の `EndPoseCtrl` コマンドでは、実行前に目標 IK 解を取得できません。そのため、関節制限の厳密な検証ができるのは `joint_delta_deg` モードだけです。Cartesian モードでは、ワークスペース、ステップサイズ、現在関節角マージン、Piper アーム状態をチェックします。
- camera-to-base キャリブレーションは含まれていません。VLA がカメラ座標系のアクションを出力する場合は、`SafetyChecker.build_plan()` の前に変換を追加してください。
- 自己衝突モデルは含まれていません。より強い保証が必要な場合は、Piper URDF とシーン形状を使って MoveIt / Pinocchio / MuJoCo の衝突チェッカを追加してください。
- 力覚 / トルク接触安全は実装されていません。

## ファイル構成

```text
piper_vla_guard/
  actions.py          OpenPI / 手動アクションチャンクをパースする。
  config.py           YAML 安全設定を読み込む。
  safety.py           軌道プランを構築し、検証する。
  piper_adapter.py    モックおよび実 piper_sdk アダプタ。
  policy_adapter.py   任意の OpenPI websocket クライアントラッパー。
  executor.py         ドライランおよび実行経路。
  ui_app.py           Gradio UI。
  logging_utils.py    JSONL ログ。
configs/
  safety.example.yaml
examples/
  sample_actions.json
```
