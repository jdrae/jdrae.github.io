# AGENTS.md

Project context for AI assistants working in this repo.

## What this is

A personal Jekyll blog deployed to GitHub Pages via GitHub Actions. Layout is a
left sidebar (profile + categories) and a right column (post list, or post
content with recent-posts strip beneath). Visual style: GitHub Primer light
theme, system sans only, no serif fonts.

The site is bilingual: English at `/`, Korean at `/ko/`, with an EN/KO toggle
in the header. See the [Bilingual (EN/KO)](#bilingual-enko) section.

## Tech stack quirks worth knowing

**This site does NOT use the `github-pages` gem.** That gem pins Jekyll 3.9 +
Liquid 4.0.3, which calls `Object#tainted?` — a method removed in Ruby 3.2+.
Local dev on modern Ruby crashes with the legacy gem. Instead:

- `Gemfile` uses plain `jekyll ~> 4.3`
- `.github/workflows/pages.yml` builds the site with the same Jekyll 4 and
  deploys via `actions/deploy-pages` (Pages source must be set to **GitHub
  Actions** in repo settings, not "Deploy from a branch")

If you ever need to switch back to the classic Pages builder, you'll have to
downgrade Ruby to 2.7 / 3.1 *and* swap the Gemfile.

**Stdlib gems removed in Ruby 3.4+** are pinned explicitly in the Gemfile:
`csv`, `base64`, `bigdecimal`, `logger`. Jekyll 4 still requires these.
Don't remove them.

## Local dev (macOS)

System Ruby (2.6) is too old. Use Homebrew Ruby (4.x):

```bash
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH"
bundle install
bundle exec jekyll serve --livereload --port 4000
```

Open <http://127.0.0.1:4000>. The dev server rebuilds on file change.

## Architecture

```
_config.yml             site + jekyll-archives + ko_posts collection
_data/
  i18n.yml              UI strings keyed by lang (en, ko)
_layouts/
  default.html          frame: header (lang toggle) + sidebar + content + footer
  home.html             paginated post list, lang-aware (used by /index.html and /ko/index.html)
  post.html             single post (default for _posts/ and _ko_posts/)
  category.html         per-category post list (jekyll-archives for EN, _plugins/ko_categories.rb for KO)
_includes/
  profile.html          sidebar profile card (lang-aware bio)
  categories.html       sidebar category list, lang-aware
  recent-posts.html     paginated list of all posts on post detail pages, lang-aware
  lang-toggle.html      EN/KO header toggle + translation-pair resolution
  icon-link.html        inline link-icon SVG (uses currentColor)
_posts/                 EN posts (lang: en)
_ko_posts/              KO posts (lang: ko, /ko/... permalink)
_plugins/
  ko_categories.rb      mirrors jekyll-archives for _ko_posts/
ko/
  index.html            KO home
  categories.html       KO categories index
  404.html              KO "no translation" page
404.html                EN 404 / "no translation" page (also GitHub Pages' default 404)
assets/
  css/style.scss        all styles, all colors via :root vars (incl. Rouge syntax palette)
  images/               favicon + avatar live here
  js/pagination.js      client-side paginator for any [data-paginate=N] container
```

The accent palette in `style.scss` is **green** (`--color-accent-fg: #178a3d`),
not Primer's default blue. All hex/rgba literals are confined to `:root` —
including a `--syntax-*` set used by the `.highlight` block.

### Page frame
`default.html` puts the `<footer>` **outside** `<div class="frame">`, at the
body level. The frame is a max-width 1200px column; the footer is a
full-width band with its own `.site-footer__inner` max-width container. Don't
move the footer back inside the frame — it gives the footer its own bg/border
and lets it span the viewport on wide screens.

`<head>` calls `{% seo title=false %}` deliberately — `jekyll-seo-tag` would
otherwise emit a second `<title>` tag that competes with the manual one. The
manual `<title>` uses `Page · Site` formatting; `{% seo %}` still runs for
its other meta tags (og:*, twitter:*, etc.).

Sticky footer pattern: `body` is a flex column (`min-height: 100vh`), `.frame`
is `flex: 1`, and `.layout` is also `flex: 1` inside the frame. The footer
naturally sits at the bottom on short pages.

### Recent-posts placement
`default.html` only renders `recent-posts.html` when `page.layout == "post"`,
positioned **below** `<main>`. Home, categories index, category, and 404
pages do not show it.

Each row in the widget shows a small category chip (the post's first
category, label translated via `t.category_labels[slug]`) in front of the
title. The `.recent-posts__lead` flex container holds the chip + title so
the title still ellipsizes correctly when the row is narrow.

### Per-category pages
The `jekyll-archives` plugin auto-generates a page per category at
`/categories/:name/` using `_layouts/category.html`. The plugin gives the
layout `page.title` (category name) and `page.posts` (filtered list).
**Don't manually create category files** — they'll collide with the generator.

The `categories.html` page at `/categories/` is the index that links to all
category pages.

## Bilingual (EN/KO)

EN at `/`, KO at `/ko/`. Default lang is EN.

### Collections
- `_posts/` — EN posts. Default `lang: en`, permalink `/:year/:month/:day/:title/`.
- `_ko_posts/` — KO posts. Default `lang: ko`, permalink `/ko/:year/:month/:day/:title/`.

Both render through `_layouts/post.html`. Defaults live in `_config.yml`.

### Pairing translations
Set the same `translation_key:` in both files' front matter. The toggle scans
the other collection for a matching key.

```yaml
# _posts/2026-04-15-hello-world.md
translation_key: hello-world

# _ko_posts/2026-04-15-hello-world.md
translation_key: hello-world
```

Posts without a sibling render fine — the toggle falls back to `/404.html`
or `/ko/404.html`.

### UI strings
`_data/i18n.yml` is keyed by lang. Resolve in templates via:

```liquid
{% assign t = site.data.i18n[page.lang] %}
{{ t.categories }}
```

`category_labels` inside `i18n.yml` maps category slugs to display names per
lang. Slugs stay English in URLs (`/categories/code/`, `/ko/categories/code/`);
only the visible label translates.

### Language toggle (`_includes/lang-toggle.html`)
Resolves the link target in this order:

1. **Post** — sibling by `translation_key`.
2. **Category page** — same slug in the other collection, only if it has at
   least one post there.
3. **Static page** (home, categories index) — prefix-swap between `/` and `/ko/`.
4. **Fallback** — `/404.html` or `/ko/404.html`.

Buttons render in **fixed EN→KO order** regardless of `page.lang`. Only the
`is-active` class moves between them — positions stay stable so the header
doesn't shift on language change. The active one is a `<span>`, the inactive
one is an `<a>`.

### KO category pages
`jekyll-archives` only sees the `posts` collection. `_plugins/ko_categories.rb`
mirrors it for `_ko_posts/` and emits `/ko/categories/<slug>/` only for slugs
that have at least one KO post. Same `_layouts/category.html`, same paginator,
same chrome.

### 404 pages
`/404.html` (EN) and `/ko/404.html` (KO) are regular Jekyll pages with
`permalink:` and `sitemap: false`. They render through `_layouts/default.html`,
so the toggle/sidebar still work for navigating away. The EN one doubles as
the site's GitHub Pages default 404 (any unknown path lands there). The KO
one is only reached via toggle fallback or direct link.

## Conventions

### Adding a post
Create `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Post title"
date: 2026-04-29 10:00:00 +0900
categories: [code]      # drives sidebar list + auto-generates /categories/code/
tags: [jekyll, css]
translation_key: slug   # optional — set the same value on the KO sibling
excerpt: "Used by jekyll-seo-tag for meta tags."
---
```

Categories are case-insensitive in the URL — they get downcased and
hyphenated by the link templates.

`excerpt:` is consumed only by `jekyll-seo-tag` for SEO meta tags. Post lists
display the **first 150 characters of rendered content** instead
(`{{ post.content | strip_html | normalize_whitespace | truncate: 150 }}`),
so you don't need to keep the excerpt copy in sync with the body.

For a Korean version, mirror the file at `_ko_posts/YYYY-MM-DD-slug.md` with
the same `translation_key:` (date and slug can diverge if you want).

### User's usual post workflow
The user usually drafts new posts in Korean and asks the assistant to publish
them to the blog. Treat Korean as the source of truth unless the user says
otherwise.

Default workflow:

1. Add or update the Korean post in `_ko_posts/YYYY-MM-DD-slug.md`.
2. Convert Obsidian-style embeds such as `![[image.png]]` or
   `![[image.png|400]]` to standard Markdown image syntax.
3. Copy referenced images into a post-specific folder:
   `assets/images/posts/YYYY-MM-DD-slug/`.
4. Use short descriptive image filenames inside that folder. Do not repeat the
   date prefix in individual image filenames once the images are grouped under
   the post folder.
5. Add or update the matching English post in `_posts/YYYY-MM-DD-slug.md`.
   Keep the same content structure, headings, code blocks, links, images, and
   `translation_key`; translate naturally rather than literally.
6. Add missing `category_labels` in `_data/i18n.yml` for both `en` and `ko`
   whenever a new category slug is introduced.
7. Run `bundle exec jekyll build` and verify the generated EN/KO URLs and
   language toggle targets.

When a proper noun needs translation or canonicalization, ask the user first.
This includes names of people, books, papers, products, events, and
organizations. Do not guess canonical English titles for Korean proper nouns
unless the user has already provided the desired wording.

### Adding a layout/include
Layouts go in `_layouts/`, includes in `_includes/`. Reference includes via
`{% include name.html %}`. Layouts inherit via `layout:` front matter.

### Styling
Single SCSS file, GitHub Primer color tokens at the top as `:root` CSS
variables. Don't introduce SCSS partials unless the file gets too long.
Don't add serif fonts or pink/dusty-rose colors — those were explicitly
removed.

### Pagination
Pagination is **client-side**, not server-side. `assets/js/pagination.js`
finds any container marked `data-paginate="N"`, slices descendants tagged
`data-paginate-item` into pages of N, and appends a prev/info/next nav at
the end. Currently 5 per page on the home post list, each category page,
and the recent-posts widget.

The `jekyll-paginate` gem is intentionally NOT installed — it only paginates
the home page, while we need the same UX on category pages and the
recent-posts widget. One JS paginator handles all three. Trade-off: no
URL-based `/page2/` (the page state lives in JS). For a small blog this is
fine; if SEO of paginated lists matters, swap to `jekyll-paginate-v2`.

Two extra hooks the paginator understands:

- **`data-paginate-current`** on an item — the paginator opens on the page
  containing this item instead of page 1. Set in Liquid by comparing
  `post.url` to `page.url`. Used by `recent-posts.html` so the widget tracks
  the post the user is reading.
- **`data-last-visible`** — set by the paginator at runtime on the last
  currently-visible item. Lets CSS hide the bottom border on the last row of
  any page, not just the DOM-last `<li>`.

### Mobile post navigation
`pagination.js` also handles one non-pagination concern: on viewports
≤880px (the single-column breakpoint), it scrolls `.post__title` into
view on fresh post-page loads so the stacked sidebar doesn't push the
content below the fold. Back/forward navigations and URLs with a hash
fragment are left alone so browser scroll restoration and anchor links
still work. **The 880px breakpoint must stay in sync with the CSS
`@media (max-width: 880px)` rule** — if you change one, change both.

### Categories vs. tags
- **Categories** are navigable. They render as accent-blue chips on the post
  detail header and link to `/categories/:name/`. Drive the sidebar list.
- **Tags** are not navigable. They render as muted-gray chips on the post
  list (home + category pages) and at the post detail footer. Used for
  topical labels only — there are no per-tag pages.

Both chip types share identical sizing (`display: inline-block`, `padding:
0 8px`, `line-height: 20px`, `font-size: 0.8125rem`, `border-radius: 999px`,
`white-space: nowrap`). Only the color tokens differ. If you add another
chip variant, follow the same shape rules.

## What to NOT do

- Don't add the `github-pages` gem back (see "Tech stack quirks" above)
- Don't manually create `_pages/categories/foo.md` — `jekyll-archives` owns
  per-category pages
- Don't manually create `ko/categories/foo/` pages — `_plugins/ko_categories.rb`
  owns those (mirror of the jekyll-archives constraint)
- Don't change the EN→KO render order in `lang-toggle.html` — the fixed order
  is intentional to prevent header layout shift on language change
- Don't use serif font stacks (Georgia, Iowan, Garamond) — sans only
- Don't use the dusty-rose palette (`#f3dfdb`, `#c89a93`, etc.) — GitHub
  Primer colors only
