# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal blog (zhangzihang's blog) built with **Hexo 8.1.1**, published at `https://zzhx1.github.io`. Content is in Chinese (`zh-CN`). The active theme is **butterfly** (configured in `_config.yml`); the `hexo-theme-landscape` entry in `package.json` and the empty `_config.landscape.yml` are leftovers from Hexo's default scaffold and are not rendered.

## Common Commands

```bash
npm run server      # hexo server — local preview at http://localhost:4000
npm run build       # hexo generate — render static site into public/
npm run clean       # hexo clean — remove db.json and public/
npm run deploy      # hexo deploy — push generated site per _config.yml deploy section
./deply.sh          # full release: clean → generate → deploy → clean  (filename misspelling is intentional — do not rename)
```

To scaffold content, use `npx hexo new post "<title>"` (or `page` / `draft`); files land in `source/_posts/` using `scaffolds/post.md` as the template.

## Deployment

`npm run deploy` uses `hexo-deployer-git` to push the generated `public/` to the `main` branch of `https://github.com/zzhx1/zzhx1.github.io.git` (GitHub Pages). The `.deploy_git/` directory is the working clone used by the deployer — do not hand-edit it; it is regenerated on each deploy. The commit message template is set in `_config.yml` under `deploy.message`.

## Repository Layout (non-obvious parts)

- `source/_posts/` — Markdown posts; filename becomes the slug. Post frontmatter follows `scaffolds/post.md` (`title`, `date`, `updated`, `tags`, `category`, `description`, `keywords`).
- `source/about/`, `source/photography/` — custom pages (front-matter `type` drives layout in the butterfly theme).
- `source/images/`, `source/downloads/code/` — static assets copied verbatim to `public/`. `_config.yml` sets `post_asset_folder: false`, so the installed `hexo-asset-image` plugin is effectively dormant — images are referenced centrally from `source/images/`, not from per-post folders.
- `themes/butterfly/` — vendored theme source. Theme-level config (menu, sidebar, comments, CDN, etc.) lives in `themes/butterfly/_config.yml`; site-level config lives in the root `_config.yml`. When both define a key, the root file wins only if the key is mirrored there — otherwise edit the theme's config.
- `db.json` and `public/` — Hexo's generated cache and output. Both are tracked in git currently (neither is in `.gitignore`) but are rebuilt by `hexo generate`. Run `hexo clean` before committing if you don't want cache churn in diffs.
- `permalink: :year/:month/:day/:title/` — URLs are date-based, so the `date` field in post frontmatter is load-bearing for the final URL. Also relevant: `future: true` (future-dated posts still render) and `updated_option: 'mtime'` (the `updated` field falls back to file mtime at build time — set `updated:` explicitly in frontmatter to pin it).

## Generators & Plugins in Use

Configured via `package.json` + `_config.yml`: archive, category, tag, index, sitemap (+ baidu sitemap), searchdb (search index at `search.xml`), word count, and `hexo-asset-image` for per-post image folders. Markdown is rendered by `hexo-renderer-marked`; syntax highlighting uses `highlight.js` with line numbers on.
