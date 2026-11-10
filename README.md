# Local AI Lab

A zero-to-elite study site for local LLMs: how they work, LM Studio mastery on a 16 GB Mac, prompting, RAG and tools, security workflows, building agents, a bug-bounty capstone, fine-tuning, and weight-space editing.

It's a static site with no build step: plain HTML, CSS and JavaScript, with lessons written in Markdown and rendered in the browser. All libraries are vendored in `assets/vendor/`, so it works fully offline once the files are on disk (fonts fall back to system fonts without internet).

## Run it locally

```bash
cd local-ai-lab
python3 serve.py
```

Open http://127.0.0.1:8000. The preview server binds to localhost only and sends the same security headers as `vercel.json`, so anything that works locally works on Vercel.

Opening `index.html` directly from the file system won't work, because browsers block `fetch()` of local files.

## Deploy to Vercel

The `local-ai-lab` folder is the whole site. There is nothing to build.

**Option A: from GitHub (recommended)**

1. Push the `local-ai-lab` folder to a GitHub repository (as the repo root, or as a subfolder).
2. In Vercel: **Add New → Project**, import the repository.
3. Framework Preset: **Other**. Leave Build Command and Output Directory empty.
4. If the site is in a subfolder, set **Root Directory** to `local-ai-lab`.
5. Deploy. Every push to the main branch redeploys automatically.

**Option B: from your terminal**

```bash
npm install -g vercel
cd local-ai-lab
vercel          # first run links the project and creates a preview deployment
vercel --prod   # promote to production
```

Vercel deployments are public by default. If you want the site visible only to you, look at the project's **Deployment Protection** settings in the Vercel dashboard.

## Structure

```text
local-ai-lab/
├── index.html            # app shell
├── vercel.json           # security headers (strict CSP, no third-party scripts)
├── serve.py              # local preview server (not deployed)
├── assets/
│   ├── app.js            # router, Markdown rendering, sidebar, search, progress
│   ├── widgets.js        # interactive widgets (sampler, memory calculator, token explorer)
│   ├── style.css         # light and dark themes
│   ├── theme.js          # applies the saved theme before first paint
│   └── vendor/           # marked 12.0.2, highlight.js 11.9.0, mermaid 10.9.1
└── content/
    ├── manifest.js       # course structure: parts and lessons, in order
    └── <part>/<lesson>.md
```

## Adding or editing lessons

1. Write `content/<part>/<slug>.md`.
2. Add `{ id: "<part>/<slug>", title: "..." }` to the right part in `content/manifest.js`.

Lessons are GitHub-flavoured Markdown with a few extras:

| Feature | Syntax |
|---|---|
| Callouts | `> [!NOTE]`, `[!TIP]`, `[!WARNING]`, `[!DANGER]`, `[!CONCEPT] Term`, `[!LAB] Title`, `[!MAC]`, `[!GOALS]` |
| Code with a filename | ```` ```python title="agent.py" ```` |
| Diagrams | ```` ```mermaid ```` blocks (re-rendered for light and dark themes) |
| Widgets | `<div data-widget="sampler"></div>`, `memory`, `tokens` |
| Persistent checklists | `- [ ] item` (ticks are remembered per lesson) |
| Internal links | `[text](#/foundations/tokens)` or `#/foundations/tokens#section-id` |

## Keyboard shortcuts

- <kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>K</kbd> or <kbd>/</kbd>: search all lessons
- <kbd>↑</kbd> <kbd>↓</kbd> <kbd>Enter</kbd>: navigate search results
- <kbd>Esc</kbd>: close search or the mobile menu

## Privacy

Progress, checklist ticks and theme choice are stored in your browser's `localStorage` only. The site makes no network requests other than loading its own files and Google Fonts.
