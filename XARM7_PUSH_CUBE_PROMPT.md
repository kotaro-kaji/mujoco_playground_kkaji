# タスク: XArm7 PushCube 環境の段階的追加（日本語）


## ユーザーによる最も重要な追記：絶対にテスト等の実行はあなたは行ってはいけません。pythonの仮想環境等を本pcでは用意していませんし、用意するつもりもありません。各実装段階のテストはすべてユーザーに任せてください。


前提メモ
- 対象リポジトリ: このプロジェクト（mujoco_playground_kkaji）
- 既存基盤: `mujoco_playground`（JAX/MJX、PPO 学習パイプライン有）
- 参考環境: Panda PushCube
  - 実装: `mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/push_cube.py`
  - ベース: `mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/panda_robotiq.py`
- 新ロボット: XArm7（純正グリッパーを使用予定）
- 追加アセット（クローン済み）: `RoboManipBaselines`
  - 参照XML: `RoboManipBaselines/robo_manip_baselines/envs/assets/mujoco/envs/xarm7/env_xarm7_cable.xml`

進め方（段階分割）

ステージ0: 独立環境のスキャフォールド（XMLはまだPandaのまま）
- 目的: Panda PushCube をコピーして、新しい環境名 `XArm7PushCube` を作成し、レジストリや設定が分離されていることを確認する。
- この段階では Panda のシーンXMLを参照したままでOK（配線確認のため）。

実装詳細（ステージ0）
- 新規モジュール作成: `mujoco_playground/_src/manipulation/xarm7/`
  - `xarm7_base.py`: 暫定ベース。まずは Panda のベース（`PandaRobotiqBase`）を継承して最小差分でOK。配置は新モジュール配下にして、後続の置換に備える。
  - `push_cube.py`: Panda版 `push_cube.py` をコピーし、クラス名を `XArm7PushCube` に変更。XMLパスは当面 Panda のシーンを参照。
  - `xmls/` ディレクトリ: ひとまず空で作成（ステージ1で中身を入れる）。
- レジストリ登録: `mujoco_playground/_src/manipulation/__init__.py`
  - `_envs["XArm7PushCube"] = xarm7.push_cube.XArm7PushCube`
  - `_cfgs["XArm7PushCube"] = xarm7.push_cube.default_config`
- Pandaクラスと名前が衝突しないよう、クラス名/環境名は固有にする。

受け入れ確認（ステージ0）
- REPLでの生成確認:
  ```py
  from mujoco_playground import registry
  cfg = registry.get_default_config("XArm7PushCube")
  env = registry.load("XArm7PushCube", config=cfg)
  print("action_size:", env.action_size)
  print("obs_size:", env.observation_size)
  ```
- 学習スクリプトが環境を認識:
  ```bash
  python learning/train_jax_ppo.py --env_name XArm7PushCube --num_timesteps 0 --run_evals False
  ```
  - コンフィグが表示され、エラー無く終了すればOK。

注意（ステージ0）
- 報酬/観測ロジックはPanda版のままで良い。目的は「独立した環境が選択できること」の確認。
- 変更は新規モジュールとレジストリ追加に限定し、最小差分に留める。

ステージ1: XArm7ベースのMJCFシーンを作成してビジュアル検証
- 目的: PandaのXMLをXArm7のシーンXMLに差し替え、`mujoco.viewer` で見た目と初期化を確認する。
  - アームが表示され、MJCFパースエラーが無いこと
  - 机/床上にプッシュ用のキューブ（`box`）が存在すること
  - コントローラ範囲が妥当で、初期状態で不安定化（NaN/爆発）しないこと

実装詳細（ステージ1）
- 参照開始点: `RoboManipBaselines/robo_manip_baselines/envs/assets/mujoco/envs/xarm7/env_xarm7_cable.xml`
  - include/参照メッシュ/テクスチャを確認し、必要ファイルを `mujoco_playground/_src/manipulation/xarm7/xmls` および `.../assets` 配下にコピー。
  - 完全な環境XMLの場合は、Panda PushCube 相当の構成になるようにシーンを組む：
    - 静的ワールド（床/テーブル、必要なら壁）
    - プッシュ対象 `box`（可能ならPandaと同じ命名）
    - 目標姿勢用 `mocap_target` ボディ
    - 後段で使うため、EEFサイト名と指ジオム名を明確に（衝突判定用）
- 新シーンXML候補パス:
  - `mujoco_playground/_src/manipulation/xarm7/xmls/scene_xarm7_push_cube.xml`
- `xarm7_base.py`（暫定ベース）の更新:
  - 当面はPandaベース継承のまま、`self._xml_path` を上記シーンに向ける。
  - Panda同様に `get_assets()` を用意し、`xmls`/`assets` を `MjModel.from_xml_string` に渡す。
  - `self._mj_model.opt.timestep = self.sim_dt` を忘れずに。

ビューア検証コマンド
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python -m mujoco.viewer \
  --mjcf=mujoco_playground/_src/manipulation/xarm7/xmls/scene_xarm7_push_cube.xml \
  2>&1 | tee mj_error.log
```
- 期待結果: ビューアが起動し、XArm7 とプッシュ対象オブジェクトが表示。`mj_error.log` に致命的エラーが無い。

受け入れ確認（ステージ1）
- MJCF検証で Fatal/ERROR が出ない。
- シーンが即座に不安定化しない（初期姿勢が安定）。

スコープ外（ステージ2の計画のみ）
- Panda依存を脱却した `XArm7Base` の実装：
  - XArm7 の関節名、EEFサイト名、指ジオム名、ギア/トルク上限、関節位置/速度範囲を定義。
  - 報酬/終了条件の衝突ジオムIDを対応させる。
  - 観測ベクトル（関節インデックス、EEFサイト）をXArm7仕様に更新。
- 既存のPushCube設計に沿って、報酬/観測は極力互換維持。

実装ガイドライン
- 変更は最小限・局所化。無関係なリファクタは避ける。
- 命名は可能な限り既存に合わせる（`box`、`mocap_target`、`gripper` など）ことでコード変更を減らす。
- ランタイムとビューアでパスが異なる問題は、`get_assets()`+`from_xml_string` 埋め込み方式で回避。

PRチェックリスト（担当者向け）
- [ ] 新モジュール `xarm7` がステージ0のスキャフォールドを含む
- [ ] 環境が `XArm7PushCube` 名でレジストリ登録されている
- [ ] ステージ0のREPL生成/学習スクリプトが通る
- [ ] ステージ1のXMLを `mujoco.viewer` で読み込み、`mj_error.log` に致命的エラーが無い
- [ ] スタイル/ヘッダ類は既存に準拠（無関係な変更なし）

参考（参照のみ）
- Panda PushCube: `mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/push_cube.py`
- Panda Base: `mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/panda_robotiq.py`
- Registry: `mujoco_playground/_src/manipulation/__init__.py`
- MJX Core: `mujoco_playground/_src/mjx_env.py`
- XArm7 XML（外部）: `RoboManipBaselines/robo_manip_baselines/envs/assets/mujoco/envs/xarm7/env_xarm7_cable.xml`
