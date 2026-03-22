#!/usr/bin/env bash
# =============================================================
# DMR — GitHub セットアップスクリプト
# 使い方:
#   1. このファイルをプロジェクトルートに置く
#   2. GITHUB_USERNAME を自分のユーザー名に書き換える
#   3. bash setup_github.sh を実行する
# =============================================================

set -e  # エラーで即停止

# ──────────────────────────────────────────────
# ★ ここを自分の情報に書き換えてください ★
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
REPO_NAME="dynamical-mode-reduction"
REPO_DESC="Dynamical Mode Reduction: dimensionality reduction via normal modes of a mass-spring system"
# ──────────────────────────────────────────────

echo "=== DMR GitHub セットアップ ==="
echo "ユーザー: $GITHUB_USERNAME"
echo "リポジトリ: $REPO_NAME"
echo ""

# pyproject.toml / README の YOUR_USERNAME を実際のユーザー名に置換
sed -i "s/YOUR_USERNAME/$GITHUB_USERNAME/g" pyproject.toml
echo "✓ pyproject.toml を更新しました"

sed -i "s/YOUR_USERNAME/$GITHUB_USERNAME/g" README.md
echo "✓ README.md を更新しました"

# git 初期化
git init
git add .
git commit -m "feat: initial release of Dynamical Mode Reduction (DMR) v0.1.0"
echo "✓ git コミット完了"

# GitHub リポジトリ作成（gh CLI が必要）
if command -v gh &> /dev/null; then
    gh repo create "$REPO_NAME" \
        --public \
        --description "$REPO_DESC" \
        --source=. \
        --remote=origin \
        --push
    echo ""
    echo "✅ アップロード完了！"
    echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "pip でインストールする方法:"
    echo "   pip install git+https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
else
    echo ""
    echo "⚠️  gh CLI が見つかりません。以下を手動で実行してください:"
    echo ""
    echo "  1. https://github.com/new でリポジトリ '$REPO_NAME' を作成"
    echo "  2. 以下のコマンドを実行:"
    echo ""
    echo "     git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "     git branch -M main"
    echo "     git push -u origin main"
    echo ""
    echo "アップロード後のインストール:"
    echo "   pip install git+https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
fi
