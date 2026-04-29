# git-blog

A minimal Jekyll blog hosted on GitHub Pages. Layout is a left sidebar
(profile + categories) plus a main column. Bilingual: English at `/`,
Korean at `/ko/`, with an EN/KO toggle in the header.

## Local development

```bash
bundle install
bundle exec jekyll serve --livereload
```

The site runs at <http://localhost:4000>. macOS users on system Ruby
should use Homebrew's Ruby — see `CLAUDE.md` for the exact PATH.

## 1. Change profile info (name, bio, links)

### Name and links — `_config.yml`

`name`, `email`, and the social handles render in the sidebar profile
card. Uncomment a line to add that link; comment it out to remove.

### Avatar

Drop a square image (PNG/JPG) at `assets/images/avatar.png`, then point
to it in `_config.yml`:

```yaml
author:
  avatar: /assets/images/avatar.png
```

Without an avatar, the sidebar shows a circle with the first letter of
the author's name.

### Bilingual bio — `_data/i18n.yml`

The sidebar bio is per-language. Edit the `bio:` strings in
`_data/i18n.yml`:

If you only set `author.bio` in `_config.yml` and skip `_data/i18n.yml`,
both languages fall back to that single string.

### Site title, browser favicon

- **Site title** (header + browser tab) — `title:` in `_config.yml`.
- **Favicon** — replace `assets/images/icon-from-plastic-donut-64.png`
  with your own PNG of the same path (or rename and update the
  `<link rel="icon">` href in `_layouts/default.html`).

### Colors and typography

CSS variables live at the top of `assets/css/style.scss`. The accent
palette is currently green; swap the three `--color-accent-*` lines to
re-theme.

## 2. Add a post (EN and KO)

### English posts → `_posts/`

Create `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Post title"
date: 2026-04-29 10:00:00 +0900
categories: [code]
tags: [jekyll, markdown]
excerpt: "Used by jekyll-seo-tag for meta tags only."
---

Your post body in markdown.
```

The post will publish at `/2026/04/29/slug/`. Post lists show the first
150 characters of rendered content (the `excerpt:` field is for SEO
only).

### Korean posts → `_ko_posts/`

Create `_ko_posts/YYYY-MM-DD-slug.md` with the same shape; it publishes
under `/ko/2026/04/29/slug/`. The `lang: ko` and the `/ko/` permalink
are applied automatically.

### Linking translations

To make the EN/KO toggle jump between a post pair, set the same
`translation_key:` on both files:

```yaml
# _posts/2026-04-29-foo.md
translation_key: foo

# _ko_posts/2026-04-29-foo.md
translation_key: foo
```

Posts without a translation render fine — the toggle just falls back
to `/404.html` or `/ko/404.html` ("no translation" pages). The date
and slug can differ between the two files; only `translation_key`
needs to match.

## 3. Manage categories

### Add a category

Add it to the post's front matter:

```yaml
categories: [code]
```

The category page is generated automatically at `/categories/code/`
(and `/ko/categories/code/` for Korean posts). The sidebar list and
`/categories/` index pick it up on the next build.

Multiple categories on one post work too:

```yaml
categories: [code, design]
```

That post then appears under both `/categories/code/` and
`/categories/design/`.

### Translate the category label

Category **slugs** stay in English so URLs are shared across languages
(`/categories/code/` ↔ `/ko/categories/code/`). Only the visible label
translates. Add an entry under `category_labels` in `_data/i18n.yml`:

```yaml
en:
  category_labels: {}        # English labels default to the slug

ko:
  category_labels:
    code: 코드
    design: 디자인
    meta: 메타
```

If a slug is missing from the map, the slug itself is shown.

### Remove a category

Delete it from every post's front matter. Empty categories disappear
from the sidebar and index automatically; no page is generated for
them.

## Deploying to GitHub Pages

This site uses Jekyll 4 (not the legacy `github-pages` gem) and ships
with `.github/workflows/pages.yml`, which builds and deploys on every
push to `main`.

1. Push the repo to GitHub.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Push to `main`. The workflow deploys to
   `https://<username>.github.io/<repo>/`.

For a custom domain, drop a `CNAME` file at the project root and point
your DNS at GitHub Pages.

## Layout

```
git-blog/
├── _config.yml             # site settings, author, collections
├── _data/i18n.yml          # UI strings + bio + category labels per lang
├── _layouts/               # default, home, post, category
├── _includes/              # profile, categories, recent-posts, lang-toggle
├── _plugins/
│   └── ko_categories.rb    # generates /ko/categories/<slug>/ pages
├── _posts/                 # English posts
├── _ko_posts/              # Korean posts
├── ko/                     # /ko/ home, categories index, 404
├── 404.html                # site default 404 (also "no EN translation")
├── assets/
│   ├── css/style.scss
│   ├── js/pagination.js
│   └── images/             # avatar, favicon
├── index.html              # / home
└── categories.html         # /categories/ index
```

For deeper architecture notes (toggle resolution order, why
`jekyll-archives` doesn't see `_ko_posts/`, etc.), see `CLAUDE.md`.
