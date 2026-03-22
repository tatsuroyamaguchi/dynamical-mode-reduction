"""
DMR デモ & 比較スクリプト
=========================
DMR (Dynamical Mode Reduction) を PCA / t-SNE / UMAP と比較する。

データセット:
    1. Swiss Roll       — 大域構造（巻き構造の展開）
    2. Nested Circles   — 非線形構造（同心円分離）
    3. Anisotropic Blobs — クラスター構造（密度不均一）
    4. S-Curve          — 多様体学習ベンチマーク
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# ── データ生成 ──────────────────────────────────────────────────
from sklearn.datasets import make_swiss_roll, make_circles, make_s_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# DMR
from dmr import DynamicalModeReduction

# UMAP（インストール済みなら使用）
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Skipping UMAP comparison.")


def make_anisotropic_blobs(n=500, seed=42):
    """密度が異なる3クラスターを生成"""
    rng = np.random.default_rng(seed)
    c1 = rng.multivariate_normal([0, 0, 0], np.diag([0.2, 0.2, 0.2]), 300)  # 密
    c2 = rng.multivariate_normal([5, 5, 5], np.diag([1.0, 1.0, 1.0]), 150)  # 中
    c3 = rng.multivariate_normal([10, 0, 5], np.diag([2.0, 2.0, 2.0]), 50)  # 疎
    X = np.vstack([c1, c2, c3])
    labels = np.array([0]*300 + [1]*150 + [2]*50)
    return X, labels


def timer(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {name:20s}: {elapsed:.2f}s")
    return result, elapsed


# ── メイン比較 ──────────────────────────────────────────────────

def run_comparison():
    np.random.seed(42)
    N = 600

    datasets = {
        'Swiss Roll\n（大域構造保持テスト）': make_swiss_roll(N, noise=0.15, random_state=42),
        'Nested Circles\n（非線形分離テスト）': (
            *make_circles(N, noise=0.05, factor=0.4, random_state=42),
        ),
        'Anisotropic Blobs\n（密度不均一テスト）': make_anisotropic_blobs(N),
        'S-Curve\n（多様体学習ベンチマーク）': make_s_curve(N, noise=0.05, random_state=42),
    }

    methods = ['DMR (本手法)', 'PCA', 't-SNE']
    if HAS_UMAP:
        methods.append('UMAP')
    n_methods = len(methods)
    n_datasets = len(datasets)

    fig = plt.figure(figsize=(4 * n_methods + 1, 4.2 * n_datasets + 0.5))
    fig.patch.set_facecolor('#0d0d0d')

    gs = gridspec.GridSpec(
        n_datasets, n_methods,
        figure=fig,
        hspace=0.45, wspace=0.18,
        top=0.94, bottom=0.04, left=0.04, right=0.97
    )

    fig.suptitle(
        'Dynamical Mode Reduction (DMR)  vs  既存手法',
        color='white', fontsize=16, fontweight='bold', y=0.975
    )

    cmap = plt.cm.plasma

    timing_summary = {}

    for row, (ds_name, (X, color)) in enumerate(datasets.items()):
        X = StandardScaler().fit_transform(X)

        print(f"\n【{ds_name.split(chr(10))[0]}】")

        # ─ DMR ─
        dmr = DynamicalModeReduction(n_components=2, n_neighbors=15, alpha=1.0, mass_power=0.5)
        emb_dmr, t_dmr = timer('DMR', dmr.fit_transform, X)
        timing_summary.setdefault('DMR', []).append(t_dmr)

        # ─ PCA ─
        emb_pca, t_pca = timer('PCA', PCA(n_components=2).fit_transform, X)
        timing_summary.setdefault('PCA', []).append(t_pca)

        # ─ t-SNE ─
        emb_tsne, t_tsne = timer('t-SNE', TSNE(n_components=2, perplexity=30,
                                                random_state=42).fit_transform, X)
        timing_summary.setdefault('t-SNE', []).append(t_tsne)

        # ─ UMAP ─
        if HAS_UMAP:
            emb_umap, t_umap = timer(
                'UMAP',
                umap.UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform, X
            )
            timing_summary.setdefault('UMAP', []).append(t_umap)

        embeddings = [emb_dmr, emb_pca, emb_tsne]
        timings    = [t_dmr,   t_pca,   t_tsne  ]
        if HAS_UMAP:
            embeddings.append(emb_umap)
            timings.append(t_umap)

        for col, (method_name, emb, t) in enumerate(zip(methods, embeddings, timings)):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#1a1a1a')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

            norm = Normalize(vmin=color.min(), vmax=color.max())
            sc = ax.scatter(
                emb[:, 0], emb[:, 1],
                c=color, cmap=cmap, norm=norm,
                s=7, alpha=0.8, linewidths=0
            )

            # タイトル（最上行のみメソッド名）
            if row == 0:
                is_dmr = col == 0
                title_color = '#ff6b6b' if is_dmr else '#aaa'
                title_weight = 'bold' if is_dmr else 'normal'
                ax.set_title(method_name, color=title_color, fontsize=11,
                             fontweight=title_weight, pad=6)

            # データセット名（最左列のみ）
            if col == 0:
                ax.set_ylabel(ds_name, color='#ccc', fontsize=8.5, labelpad=6)

            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.98, 0.02, f'{t:.2f}s', transform=ax.transAxes,
                    color='#888', fontsize=7.5, ha='right', va='bottom')

            # DMR の質量マーカー（点サイズで質量を表現）
            if col == 0 and hasattr(dmr, 'masses_'):
                # 質量の逆数でサイズ（軽い点=大きい○ で目立たせる）
                sizes = 3 + 20 * (1 / (dmr.masses_ + 0.1))
                sizes = np.clip(sizes, 2, 30)
                ax.scatter(emb[:, 0], emb[:, 1],
                           c=color, cmap=cmap, norm=norm,
                           s=sizes, alpha=0.85, linewidths=0)

    # ── タイミング比較バー（右下余白） ──
    times_mean = {k: np.mean(v) for k, v in timing_summary.items()}
    ax_t = fig.add_axes([0.035, 0.01, 0.19, 0.03])
    ax_t.set_facecolor('#1a1a1a')
    bar_colors = ['#ff6b6b'] + ['#666'] * (n_methods - 1)
    ax_t.barh(list(times_mean.keys()), list(times_mean.values()),
              color=bar_colors, height=0.6)
    ax_t.set_xlim(0, max(times_mean.values()) * 1.15)
    for spine in ax_t.spines.values():
        spine.set_visible(False)
    ax_t.tick_params(colors='#aaa', labelsize=7)
    ax_t.set_title('平均計算時間 (s)', color='#aaa', fontsize=7, pad=3)

    out_path = _out('comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"\n✓ 比較図を {out_path} に保存しました。")


# ── 解釈性デモ ──────────────────────────────────────────────────

def run_interpretability_demo():
    """DMR の固有振動モードと質量分布を可視化"""
    from sklearn.datasets import make_swiss_roll

    X, color = make_swiss_roll(600, noise=0.15, random_state=42)
    X = StandardScaler().fit_transform(X)

    dmr = DynamicalModeReduction(n_components=4, n_neighbors=15)
    dmr.fit_transform(X)

    # ─ レポート表示 ─
    print("\n" + "="*55)
    print("【解釈性レポート — Swiss Roll】")
    print("="*55)
    for r in dmr.interpretability_report():
        print(f"  次元 {r['dimension']}: λ={r['eigenvalue']:.4f}  "
              f"ω={r['frequency']:.4f}  T={r['period']:.2f}")
        print(f"    → {r['mode_class'].upper()} — {r['interpretation']}")
    print()
    mr = dmr.mass_report()
    print(f"【質量分布】")
    print(f"  min={mr['min_mass']:.3f}, max={mr['max_mass']:.3f}, "
          f"ratio={mr['mass_ratio']:.1f}x")
    print(f"  → {mr['interpretation']}")

    # ─ モード別可視化 ─
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle('DMR: 4つの固有振動モード（Swiss Roll）',
                 color='white', fontsize=13, fontweight='bold')

    mode_labels = ['Mode 1\n（最も遅い=大域）', 'Mode 2', 'Mode 3', 'Mode 4\n（速い=局所）']
    report = dmr.interpretability_report()

    for i, (ax, label, r) in enumerate(zip(axes, mode_labels, report)):
        ax.set_facecolor('#1a1a1a')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

        ax.scatter(dmr.embedding_[:, 0], dmr.embedding_[:, i],
                   c=color, cmap='plasma', s=6, alpha=0.85, linewidths=0)
        ax.set_title(label, color='#ddd', fontsize=9)
        ax.set_xlabel(f'Mode 1 (ω={dmr.frequencies_[0]:.3f})', color='#888', fontsize=7.5)
        ax.set_ylabel(f'Mode {i+1} (ω={r["frequency"]:.3f})', color='#888', fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])

        cls_color = {'global': '#ff6b6b', 'meso': '#ffd93d', 'local': '#6bcb77'}[r['mode_class']]
        ax.text(0.02, 0.97, r['mode_class'].upper(), transform=ax.transAxes,
                color=cls_color, fontsize=8, fontweight='bold', va='top')

    plt.tight_layout()
    out_path = _out('modes.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"\n✓ モード可視化を {out_path} に保存しました。")


# ── 質量行列の可視化 ─────────────────────────────────────────────

def run_mass_visualization():
    """質量分布がどのように大域構造保持に寄与するかを可視化"""
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=400, centers=[[0,0,0],[5,5,5],[10,0,5]],
                      cluster_std=[0.3, 1.0, 2.0], random_state=42)
    X = StandardScaler().fit_transform(X)

    dmr = DynamicalModeReduction(n_components=2, n_neighbors=12, mass_power=0.5)
    emb = dmr.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d0d0d')
    fig.suptitle('DMR 質量行列の効果（密度不均一クラスター）',
                 color='white', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=dmr.masses_, cmap='coolwarm_r',
                    s=np.clip(20 / (dmr.masses_ + 0.5), 5, 80),
                    alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label='mass').ax.yaxis.label.set_color('white')
    ax.set_title('質量マップ\n（赤=重い=密な領域=アンカー）', color='#ddd', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor('#444')

    ax = axes[1]
    ax.set_facecolor('#1a1a1a')
    colors_cluster = ['#ff6b6b', '#ffd93d', '#6bcb77']
    for ci in range(3):
        mask = (y == ci)
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colors_cluster[ci], s=8, alpha=0.85,
                   label=f'Cluster {ci+1} (σ={[0.3,1.0,2.0][ci]})', linewidths=0)
    ax.legend(facecolor='#2a2a2a', labelcolor='white', fontsize=8, framealpha=0.8)
    ax.set_title('クラスター分離\n（密度が異なっても正確に分離）', color='#ddd', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor('#444')

    plt.tight_layout()
    out_path = _out('mass_effect.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"✓ 質量効果を {out_path} に保存しました。")


if __name__ == '__main__':
    import sys
    import os

    # スクリプトと同じディレクトリに出力する
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    def _out(filename):
        return os.path.join(SCRIPT_DIR, filename)

    sys.path.insert(0, SCRIPT_DIR)

    print("=" * 55)
    print(" Dynamical Mode Reduction (DMR) — デモ実行")
    print("=" * 55)

    run_comparison()
    run_interpretability_demo()
    run_mass_visualization()

    print("\n全デモ完了！")
