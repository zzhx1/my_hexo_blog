#!/bin/bash
# 一键发布：清理 → 生成 → 推送到 GitHub Pages → 清理
#
# 注意：这里没有用 `hexo deploy`（hexo-deployer-git）。原因是它复制产物时会
# 跳过点文件（dotfile），导致 .nojekyll 推不上去；而 GitHub Pages 默认用 Jekyll
# 处理静态文件，Hexo/Fluid 的产物会让 Jekyll 构建失败，线上就一直停在旧版本。
# 这里直接把 public/（含 .nojekyll）推到 Pages 仓库，让 Pages 跳过 Jekyll、当
# 纯静态站点发布。
set -e

REPO="https://github.com/zzhx1/zzhx1.github.io.git"
BRANCH="main"

hexo clean
hexo generate

cd public
touch .nojekyll                      # 关闭 GitHub Pages 的 Jekyll 处理
git init -q
git add -A
git commit -q -m "Site updated: $(date '+%Y-%m-%d %H:%M:%S')"
git push -f "$REPO" HEAD:"$BRANCH"
cd ..

hexo clean
