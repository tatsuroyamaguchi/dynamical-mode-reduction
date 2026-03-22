# Dynamical Mode Reduction (DMR)

> 物理・力学系からのアナロジーに基づく、新しい次元削減アルゴリズム

---

## 概要

**DMR (Dynamical Mode Reduction)** は、高次元データを**力学系（質点＋バネ）**として解釈し、その**固有振動モード（法線モード）**を低次元表現として用いる次元削減手法です。

PCA の線形性の限界・t-SNE の計算コスト・UMAP の局所構造偏重・既存手法全般の解釈性の低さ、という4つの課題を同時に解決することを目指して設計されました。

---

## 核心アイデア

高次元空間の各データ点を**質点**、近傍関係を**バネ**として接続した力学系を構成します。その系を支配する一般化固有値問題

```
L v = λ M v
```

を解き、最低振動数のモード（遅い振動 = 大域構造）を低次元埋め込みとして採用します。

| 記号 | 意味 |
|------|------|
| `L` | 剛性行列（幾何適応バネのグラフラプラシアン） |
| `M` | **質量行列（★ 核心の新規性）** — 局所密度の逆数から計算 |
| `λ` | 一般化固有値 = 角振動数の2乗 ω² |
| `v` | 固有モード = 低次元埋め込みの各軸 |

### 質量行列の物理的意味

```
密な領域（ρ大） → 重い質点 → 動きにくい → 大域トポロジーのアンカー
疎な領域（ρ小） → 軽い質点 → 動きやすい → 局所変形に柔軟
```

密度ρは「接続バネ強度の総和」で推定し、`m_i = ρ_i^(−mass_power)` で質量に変換します。この一行が、大域構造保持と局所柔軟性を同時に達成するカギです。

### バネ定数（幾何適応）

```
k_ij = exp(−d² / 2σ_ij²)  ×  d^(−α)
        └─ Gaussian（近傍選択）─┘   └─ 剛性冪乗（近いほど剛い）─┘
```

- **Gaussian項**：局所性を担保し、遠距離の影響を自然にカットオフ
- **冪乗項**：近傍ほど強いバネ → 非線形構造を捉える

---

## 既存手法との比較

| 手法 | 解くべき問題 | 線形 | 計算速度 | 大域構造 | 解釈性 |
|------|------------|:----:|:-------:|:-------:|:------:|
| PCA | `X^T X` の固有値問題 | ✗ | ◎ | △ | △ |
| Laplacian EM | `L v = λ D v` | ○ | ○ | △ | △ |
| t-SNE | KL\[P‖Q\] の勾配降下 | ○ | ✗ | ✗ | ✗ |
| UMAP | 局所ファジー単体複体の最適化 | ○ | △ | △ | ✗ |
| **DMR** | **`L v = λ M v`** | **○** | **○** | **○** | **○** |

### 計測結果（N=600, sklearn ベンチマークデータ）

| 手法 | 平均実行時間 |
|------|------------|
| PCA | 0.00s |
| **DMR** | **0.08s** |
| UMAP | 0.80s |
| t-SNE | 3.08s |

---

## インストール

```bash
pip install numpy scipy scikit-learn matplotlib
# UMAP（任意）
pip install umap-learn
```

Python 3.10 以上を推奨します（型ヒント `float | str` を使用）。

---

## クイックスタート

```python
from dmr import DynamicalModeReduction
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler

X, color = make_swiss_roll(1000, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

dmr = DynamicalModeReduction(n_components=2)
embedding = dmr.fit_transform(X)
# embedding.shape → (1000, 2)
```

scikit-learn 互換の API なので、パイプラインにも組み込めます。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('dmr', DynamicalModeReduction(n_components=2)),
])
embedding = pipe.fit_transform(X)
```

---

## パラメータ

```python
DynamicalModeReduction(
    n_components = 2,      # 埋め込み次元数（保持するモード数）
    n_neighbors  = 15,     # グラフ構築に用いる最近傍数
    sigma        = 'auto', # ガウスカーネルのバンド幅。'auto' で点ごとに適応
    alpha        = 1.0,    # バネ剛性の距離冪乗指数
    mass_power   = 0.5,    # 密度→質量変換の指数
    freq_scale   = True,   # 振動数による次元スケーリング
    random_state = 42,
)
```

### `alpha` — バネ剛性の指数

| 値 | 挙動 |
|----|------|
| `0.0` | 距離に依存しない剛性（Laplacian Eigenmaps と等価） |
| `1.0` | 近いほど剛いバネ（デフォルト・推奨） |
| `2.0` | 強い局所剛性（緻密な局所構造を重視） |

### `mass_power` — 密度適応の強さ

| 値 | 挙動 |
|----|------|
| `0.0` | 均一質量（Laplacian Eigenmaps と等価） |
| `0.5` | 中程度のアンカー効果（デフォルト） |
| `1.0` | 強い密度適応（不均一データに有効） |

---

## 解釈性 API

DMR の最大の特徴のひとつが、埋め込み後の**物理的解釈**を取り出せることです。

### 固有モードの解釈レポート

```python
for r in dmr.interpretability_report():
    print(f"次元 {r['dimension']}: λ={r['eigenvalue']:.4f}  ω={r['frequency']:.4f}")
    print(f"  → {r['mode_class'].upper()} — {r['interpretation']}")
```

出力例（Swiss Roll）：

```
次元 1: λ=0.0736  ω=0.271  T=23.2s
  → GLOBAL — 大域モード：クラスター間分離・全体トポロジーを表現
次元 2: λ=0.163   ω=0.404  T=15.6s
  → MESO   — 中間モード：中規模の構造（サブクラスター等）を表現
```

| フィールド | 内容 |
|-----------|------|
| `eigenvalue` | λ（角振動数の2乗） |
| `frequency` | ω = √λ（固有振動数）|
| `period` | T = 2π/ω（振動周期） |
| `mode_class` | `global` / `meso` / `local` |

### 質量分布レポート

```python
print(dmr.mass_report())
# → {'min_mass': 0.60, 'max_mass': 1.78, 'mass_ratio': 2.9, ...}
```

`mass_ratio` が大きいほど、データの密度が不均一であることを示します。

---

## 属性（fit_transform 後）

| 属性 | 形状 | 内容 |
|------|------|------|
| `embedding_` | `(n_samples, n_components)` | 低次元埋め込み |
| `eigenvalues_` | `(n_components,)` | 一般化固有値 λ |
| `frequencies_` | `(n_components,)` | 固有振動数 ω = √λ |
| `masses_` | `(n_samples,)` | 各データ点の質量 |
| `spring_matrix_` | sparse matrix | バネ定数行列 K |

---

## デモの実行

```bash
python demo.py
```

以下の3つの可視化画像がスクリプトと同じディレクトリに出力されます。

| ファイル | 内容 |
|---------|------|
| `comparison.png` | 4データセット × 4手法の埋め込み比較 |
| `modes.png` | Swiss Roll の固有振動モード（4次元分） |
| `mass_effect.png` | 質量分布と密度不均一クラスターの分離 |

---

## アルゴリズムの流れ

```
入力: X ∈ R^(N×D)
  │
  ① k最近傍グラフの構築
  │   NearestNeighbors で各点の近傍を取得
  │
  ② バネ定数行列 K の計算
  │   k_ij = exp(-d²/2σ²) × d^(-α)  で幾何適応的に設定
  │
  ③ 質量行列 M の計算  ← ★ 核心
  │   ρ_i = Σ_j k_ij  （局所密度の推定）
  │   m_i = ρ_i^(-mass_power)
  │
  ④ 剛性行列 L の計算
  │   L = D - K  （グラフラプラシアン）
  │
  ⑤ 一般化固有値問題を解く
  │   L v = λ M v
  │   対称変換: M^{-1/2} L M^{-1/2} u = λ u
  │
  ⑥ 低振動数モードを採用
  │   零固有値（剛体並進）をスキップし、
  │   小さい λ から n_components 個を抽出
  │
出力: embedding ∈ R^(N×n_components)
```

---

## ファイル構成

```
.
├── dmr.py        # DynamicalModeReduction クラス本体
├── demo.py       # 比較デモスクリプト
└── README.md     # このファイル
```

---

## 今後の展望

- **逆変換（再構成）の実装** — 埋め込みから元空間への近似写像
- **Out-of-sample 拡張** — 新規データ点の変換（Nyström 近似）
- **オンライン学習** — ストリームデータへの対応
- **パラメータ自動調整** — グリッドサーチ・ベイズ最適化との統合
- **GPU 対応** — CuPy / RAPIDS による大規模データへのスケールアップ

---

## 参考文献

本手法は以下のアイデアを出発点とし、新たな組み合わせで構成されています。

- Belkin & Niyogi (2003). *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*
- McInnes et al. (2018). *UMAP: Uniform Manifold Approximation and Projection*
- Coifman & Lafon (2006). *Diffusion maps*
- Wilson & Deane (古典力学). *Normal mode analysis of molecular systems*
