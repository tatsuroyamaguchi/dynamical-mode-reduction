"""
Dynamical Mode Reduction (DMR)
===============================
A novel dimensionality reduction method inspired by classical mechanics.

【コアコンセプト】
    高次元データ点を質点（particle）として扱い、
    近傍関係をバネ（spring）として接続した力学系を構成する。
    その系の固有振動モード（法線モード, normal modes）が低次元埋め込みとなる。

【既存手法との本質的な違い】
    ┌──────────────┬────────────────────────────────────────────────────────┐
    │ 手法          │ 解くべき問題                                           │
    ├──────────────┼────────────────────────────────────────────────────────┤
    │ PCA           │ X^T X の固有値問題（線形のみ）                         │
    │ Laplacian EM  │ L v = λ D v  (D=次数行列)                             │
    │ UMAP          │ 局所ファジー単体複体の最適化                            │
    │ t-SNE         │ KL[P‖Q] の勾配降下                                    │
    │ **DMR**       │ **L v = λ M v  (M=密度由来の質量行列)**               │
    └──────────────┴────────────────────────────────────────────────────────┘

【新規性の核心】
    1. 質量行列 M: 局所密度の逆数から計算
       - 密な領域 → 重い質点 → 構造アンカー → 大域トポロジー保持
       - 疎な領域 → 軽い質点 → 自由に再配置 → 局所変形に柔軟
    
    2. 幾何適応バネ定数: k_ij = exp(-d²/2σ²) × d^(-α)
       - Gaussian項: 近傍のみ相互作用（局所性）
       - 冪乗項: 近いほど剛いバネ（非線形性）
    
    3. 解釈性: 固有値 λ = ω²（角振動数の2乗）
       - 小さいλ → 遅い振動 → 大域構造
       - 大きいλ → 速い振動 → 局所構造

Author: DMR prototype
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


class DynamicalModeReduction(BaseEstimator, TransformerMixin):
    """
    Dynamical Mode Reduction (DMR)

    高次元データを力学系として解釈し、その固有振動モードによって
    低次元埋め込みを構成する次元削減法。

    Parameters
    ----------
    n_components : int, default=2
        埋め込み次元数（保持する振動モード数）

    n_neighbors : int, default=15
        グラフ構築に用いる最近傍数

    sigma : float or 'auto', default='auto'
        ガウスカーネルのバンド幅。'auto' で点ごとに適応的に設定。

    alpha : float, default=1.0
        バネ剛性の距離冪乗指数。
        0 = 距離に依存しない剛性（Laplacian EM と同等）
        1 = 近いほど剛いバネ（デフォルト）
        2 = より強い局所剛性

    mass_power : float, default=0.5
        密度→質量変換の指数。
        0   = 均一質量（Laplacian Eigenmaps と同等）
        0.5 = 中程度のアンカー効果（デフォルト）
        1.0 = 強い密度適応

    freq_scale : bool, default=True
        True の場合、各次元を振動数で正規化する
        （遅いモード=大域構造 が相対的に強調される）

    Attributes
    ----------
    embedding_ : ndarray, shape (n_samples, n_components)
        低次元埋め込み

    eigenvalues_ : ndarray, shape (n_components,)
        一般化固有値 λ（= 角振動数の2乗 ω²）

    frequencies_ : ndarray, shape (n_components,)
        各モードの固有振動数 ω = sqrt(λ)

    masses_ : ndarray, shape (n_samples,)
        各データ点に割り当てられた質量

    spring_matrix_ : scipy.sparse matrix
        バネ定数行列 K (stiffness matrix の基底)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        sigma: float | str = 'auto',
        alpha: float = 1.0,
        mass_power: float = 0.5,
        freq_scale: bool = True,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.alpha = alpha
        self.mass_power = mass_power
        self.freq_scale = freq_scale
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_sigma(self, distances: np.ndarray) -> np.ndarray:
        """点ごとの適応バンド幅: k最近傍距離の平均"""
        return distances[:, 1:].mean(axis=1)

    def _build_spring_matrix(
        self, distances: np.ndarray, indices: np.ndarray, sigma_i: np.ndarray
    ) -> csr_matrix:
        """
        バネ定数行列 K を構築する。

        k_ij = exp(-d² / 2σ_ij²) · d^(-α)
              └─ Gaussian (近傍選択) ─┘ └─ 剛性冪乗 ─┘
        """
        N = len(sigma_i)
        row_list, col_list, val_list = [], [], []

        for i in range(N):
            for t, j in enumerate(indices[i, 1:]):
                d = distances[i, t + 1]
                sigma_ij = np.sqrt(sigma_i[i] * sigma_i[j])  # 幾何平均バンド幅
                affinity = np.exp(-d**2 / (2 * sigma_ij**2 + 1e-12))
                stiffness = (d + 1e-10) ** (-self.alpha)       # 近いほど剛い
                row_list.append(i)
                col_list.append(j)
                val_list.append(affinity * stiffness)

        K = csr_matrix((val_list, (row_list, col_list)), shape=(N, N))
        K = (K + K.T) * 0.5  # 対称化
        return K

    def _mass_matrix(self, K: csr_matrix) -> np.ndarray:
        """
        質量行列を局所密度から計算。

        密度推定: ρ_i = sum_j k_ij  （入射バネ強度の総和）
        質量:     m_i = ρ_i^(-mass_power)

        物理的意味:
            密な領域（ρ大）→ 重い質点 → 動きにくい → 大域構造アンカー
            疎な領域（ρ小）→ 軽い質点 → 動きやすい → 局所変形に柔軟
        """
        density = np.asarray(K.sum(axis=1)).flatten()
        density = np.clip(density, 1e-12, None)
        masses = density ** (-self.mass_power)
        masses /= masses.mean()  # 平均1に正規化
        return masses

    def _stiffness_matrix(self, K: csr_matrix) -> csr_matrix:
        """
        剛性行列 L（重み付きグラフラプラシアン）を構築。

        L_ii =  Σ_j k_ij   （対角: 総バネ力）
        L_ij = -k_ij        （非対角: 対バネ）

        これはポテンシャルエネルギー V = ½ Σ_ij k_ij ||r_i - r_j||² の Hessian。
        """
        degree = np.asarray(K.sum(axis=1)).flatten()
        D = diags(degree)
        return D - K  # Laplacian

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        X を低次元埋め込みに変換する。

        アルゴリズム概要:
        ① k最近傍グラフを構築
        ② バネ定数行列 K を計算（幾何適応）
        ③ 密度から質量行列 M を計算（★ 新規要素）
        ④ 一般化固有値問題 L v = λ M v を解く
        ⑤ 低振動数モード（大域構造）を埋め込み次元として採用

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        embedding : ndarray, shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape

        if N <= self.n_components + 2:
            raise ValueError(
                f"サンプル数 {N} が n_components+2={self.n_components+2} 以下です。"
            )

        # ① k最近傍
        k_nn = min(self.n_neighbors + 1, N)
        nn = NearestNeighbors(n_neighbors=k_nn, algorithm='auto', n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # ② バネ定数行列
        sigma_i = (
            self._adaptive_sigma(distances)
            if self.sigma == 'auto'
            else np.full(N, float(self.sigma))
        )
        K = self._build_spring_matrix(distances, indices, sigma_i)
        self.spring_matrix_ = K

        # ③ 質量行列（★ コアの新規性）
        masses = self._mass_matrix(K)
        self.masses_ = masses
        M_inv_sqrt = diags(1.0 / np.sqrt(masses))

        # ④ 剛性行列 L
        L = self._stiffness_matrix(K)

        # ⑤ 対称変換: M^{-1/2} L M^{-1/2} u = λ u  (v = M^{-1/2} u)
        L_sym = M_inv_sqrt @ L @ M_inv_sqrt

        k_eig = min(self.n_components + 3, N - 1)
        try:
            eigenvalues, eigenvectors = eigsh(
                L_sym, k=k_eig, which='SM', tol=1e-6, maxiter=5000
            )
        except Exception as exc:
            warnings.warn(f"疎固有値ソルバー失敗 ({exc})。密行列ソルバーに切り替えます。")
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym.toarray())

        # 昇順ソート
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # 零固有値（剛体並進）をスキップ
        skip = 0
        while skip < len(eigenvalues) - 1 and eigenvalues[skip] < 1e-7:
            skip += 1

        sel = slice(skip, skip + self.n_components)
        self.eigenvalues_ = eigenvalues[sel]
        vecs_u = eigenvectors[:, sel]

        # v = M^{-1/2} u で元の座標系に戻す
        embedding = M_inv_sqrt @ vecs_u

        # ⑥ 振動数スケーリング（任意）
        self.frequencies_ = np.sqrt(np.abs(self.eigenvalues_))
        if self.freq_scale and len(self.frequencies_) > 1:
            # 遅いモード（小さいω）を相対的に大きくスケール
            scale = self.frequencies_[0] / (self.frequencies_ + 1e-12)
            embedding *= scale[np.newaxis, :]

        # 各次元を標準化
        std = embedding.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        embedding /= std

        self.embedding_ = embedding
        return embedding

    def fit(self, X: np.ndarray, y=None) -> 'DynamicalModeReduction':
        self.fit_transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        fit済みモデルで新規データを変換する（sklearn Pipeline 互換）。
        fit_transform 時に学習した固有モードへの射影を返す。
        """
        if not hasattr(self, 'embedding_'):
            raise RuntimeError("fit_transform を先に呼び出してください。")
        return self.embedding_

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def interpretability_report(self) -> list[dict]:
        """
        各埋め込み次元の物理的解釈を返す。

        Returns
        -------
        list of dict, 各要素:
            - dimension    : 次元番号（1始まり）
            - eigenvalue   : λ（角振動数の2乗）
            - frequency    : ω = sqrt(λ)
            - period       : T = 2π/ω
            - mode_class   : 'global' / 'meso' / 'local'
            - interpretation: 説明文
        """
        if not hasattr(self, 'eigenvalues_'):
            raise RuntimeError("fit_transform を先に呼び出してください。")

        freq = self.frequencies_
        f_q1, f_q3 = np.percentile(freq, 25), np.percentile(freq, 75)
        report = []

        for i, (lam, f) in enumerate(zip(self.eigenvalues_, freq)):
            if f <= f_q1:
                cls = 'global'
                desc = '大域モード：クラスター間分離・全体トポロジーを表現'
            elif f <= f_q3:
                cls = 'meso'
                desc = '中間モード：中規模の構造（サブクラスター等）を表現'
            else:
                cls = 'local'
                desc = '局所モード：近傍の細かい幾何構造を表現'

            report.append({
                'dimension': i + 1,
                'eigenvalue': float(lam),
                'frequency': float(f),
                'period': float(2 * np.pi / (f + 1e-12)),
                'mode_class': cls,
                'interpretation': desc,
            })
        return report

    def mass_report(self) -> dict:
        """
        質量分布の統計情報（データの密度構造の可解釈な指標）。
        """
        if not hasattr(self, 'masses_'):
            raise RuntimeError("fit_transform を先に呼び出してください。")
        m = self.masses_
        ratio = m.max() / (m.min() + 1e-12)
        return {
            'min_mass': float(m.min()),
            'max_mass': float(m.max()),
            'mean_mass': float(m.mean()),
            'mass_ratio': float(ratio),
            'interpretation': (
                f'質量比 {ratio:.1f}x — '
                + ('密度の不均一性が強い（大域構造保持効果大）'
                   if ratio > 10 else
                   '密度が比較的均一（各点が対等なアンカー）')
            ),
        }
