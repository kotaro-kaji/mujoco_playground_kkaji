my_leap_hand/xmls overview
================================================

このディレクトリは、LEAP Hand を単体/他モデルと組み合わせて使うための MJCF (MuJoCo XML) を置いています。
最終的に推奨する“メイン”シーンは `leap_hand_6dof_ctrl_scene.xml` です。

ファイル一覧（要点）
--------------------------------

- `leap_hand_6dof_ctrl_scene.xml` 〔メイン/完成形〕
  - 6DoF基底（3スライダ+3ヒンジ）を `position` アクチュエータで制御し、LEAP Hand 全体をPD相当で高剛性に駆動。
  - 床はグリッド表示（非接触）と透明の物理平面を分離。`reorientation_cube.xml` を取り込み、落下・把持テストが可能。
  - 推奨の学習/制御用シーン。RLの action→ctrl に直接マッピング可能。

- `leap_rh_mjx.xml`
  - LEAP Hand 単体モデル（標準MJCF）。他シーンに include して再利用するベース。
  - `compiler meshdir=""` で外部の `meshdir` 影響を遮断。指は `position` アクチュエータ。

- `leap_rh_mjx_direct.xml`
  - LEAP の root（`leap_mount`）自体に 6DoF の関節とアクチュエータを与え、直接制御する構成。
  - 実験用の“直結6DoF”モデル。モジュール性より単体検証を優先したいときに使用。

- `ur5e_leap_menagerie_scene.xml`
  - Menagerie の UR5e と LEAP を剛結（`equality/weld`）。UR5e は位置制御（ゲイン控えめ）。
  - シーン/ソルバ設定を安定化し、拘束をやや過減衰にチューニング。

- `ur5e_local.xml`
  - UR5e のローカルアセット参照やアクチュエータ設定をまとめたラッパー。

- `reorientation_cube.xml`
  - 共有の把持用キューブ。テクスチャ付きメッシュ+物理ボックスを同居。
  - 他シーンから include して使用（初期高さは `pos` で調整）。

- `my_scene_mjx_cube.xml`
  - ローカル実験用（LEAP+キューブ等の組み合わせ例）。将来整理予定。

- `legacy/`
  - 旧シーンや一時退避。後方互換や参考用に保持。

運用のヒント
----------------

- 6DoFの“PD化”は、`position` アクチュエータの `kp`（P相当）と、各関節の `damping`+`armature`（D/反射慣性相当）で実現。
- 過減衰にして揺れを抑え、安定化後に `kp` を段階的に上げて剛性を上げると扱いやすい。
- 床は見た目用（非接触）と物理用（透明）を分離。接触条件は `ground_physical` の属性で調整。

メインにすべき理由（`leap_hand_6dof_ctrl_scene.xml`）
-----------------------------------------------

- RL/制御でそのまま `ctrl` にアクションを流せる（mocap不要）。
- モジュール切替やUR5e等との結合に依存せず、LEAP単体で完結。
- 数値的に安定で、チューニングが読みやすい（`kp`, `damping`, `armature` を明示）。


