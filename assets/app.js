/* Local AI Lab — client-side router, Markdown renderer, navigation, search, progress */
(() => {
  "use strict";

  const COURSE = window.COURSE;
  const MERMAID_URL = "assets/vendor/mermaid.min.js";
  const KEYS = { theme: "lal-theme", done: "lal-done", open: "lal-open", last: "lal-last", tasks: "lal-tasks" };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------- storage (never trusted to exist) ----------
  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v == null ? fallback : JSON.parse(v);
      } catch (e) {
        return fallback;
      }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode */ }
    },
  };

  // ---------- course index ----------
  const LESSONS = [];
  COURSE.parts.forEach((part, pi) => {
    part.lessons.forEach((l, li) => {
      LESSONS.push({ id: l.id, title: l.title, part, partIndex: pi, indexInPart: li, index: LESSONS.length });
    });
  });
  const byId = new Map(LESSONS.map((l) => [l.id, l]));

  const PART_BLURBS = {
    start: "Orientation, your Mac, your model and how to study.",
    foundations: "Tokens, transformers, sampling, quantization, memory and alignment from first principles.",
    lmstudio: "Every load and sampling knob, presets for your Mac, performance and the API.",
    using: "Prompting small models, structured output, tool calling, RAG and MCP.",
    security: "Private security workflows, personas, verification and threat-modelling your AI.",
    agents: "The agent loop from scratch, memory, skills, frameworks and reliability.",
    bugbounty: "Capstone: a scoped, local, human-in-the-loop hunting assistant.",
    finetune: "LoRA, datasets, MLX on the Mac, Unsloth in the cloud, GGUF export and evals.",
    weights: "The MIT study, interpretability, steering, ablation and Claude as weight programmer.",
    infra: "An offline AI stack, hardening, and what to buy when 16 GB is not enough.",
    ollama: "Install, Modelfiles, server tuning, the API, security, and a fair benchmark against LM Studio.",
    models: "Model families, size tiers, the right model per job, and running your own model shootout.",
    reference: "Study plan, cheatsheet, glossary and reading list.",
  };

  let done = new Set(store.get(KEYS.done, []));
  let openParts = new Set(store.get(KEYS.open, ["start"]));

  // ---------- markdown ----------
  const LANG_ALIAS = { sh: "bash", zsh: "bash", shell: "bash", console: "bash", py: "python", js: "javascript", ts: "typescript", yml: "yaml", toml: "ini", jinja: "django", jinja2: "django", text: "plaintext", txt: "plaintext", docker: "dockerfile", jsonl: "json" };

  const COPY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';

  const renderer = new marked.Renderer();
  renderer.code = function (code, infostring) {
    const info = (infostring || "").trim();
    const lang = (info.split(/\s+/)[0] || "").toLowerCase();
    const titleMatch = info.match(/title=["']([^"']+)["']/);
    const title = titleMatch ? titleMatch[1] : "";
    if (lang === "mermaid") {
      return `<div class="mermaid-wrap"><pre class="mermaid-src" hidden>${esc(code)}</pre><div class="mermaid-out"></div></div>`;
    }
    const hl = LANG_ALIAS[lang] || lang;
    let body;
    if (hl && hl !== "plaintext" && window.hljs && hljs.getLanguage(hl)) {
      try { body = hljs.highlight(code, { language: hl, ignoreIllegals: true }).value; } catch (e) { body = esc(code); }
    } else {
      body = esc(code);
    }
    const label = lang && lang !== "plaintext" ? lang : "";
    return `<figure class="code"><figcaption>${title ? `<span class="code-title">${esc(title)}</span>` : ""}<span class="code-lang">${esc(label)}</span><button class="copy-btn" type="button" aria-label="Copy code">${COPY_ICON}<span>Copy</span></button></figcaption><pre><code>${body}</code></pre></figure>`;
  };
  renderer.listitem = function (text, task, checked) {
    if (task) {
      const clean = text.replace(/<input[^>]*>\s*/, "");
      return `<li class="task${checked ? " checked" : ""}"><input type="checkbox"${checked ? " checked" : ""}><span class="task-body">${clean}</span></li>\n`;
    }
    return `<li>${text}</li>\n`;
  };
  marked.use({ renderer, gfm: true, breaks: false });

  // ---------- callouts ----------
  const ICONS = {
    goals: '<path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    concept: '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z"/>',
    note: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    tip: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/>',
    warning: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    danger: '<path d="M12 2 3 6v6c0 5 3.8 9.4 9 10 5.2-.6 9-5 9-10V6l-9-4z"/><path d="M12 8v4M12 16h.01"/>',
    lab: '<path d="M9 3h6M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3"/><path d="M7 15h10"/>',
    mac: '<rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  };
  const CALLOUT = {
    GOALS: ["goals", "What you'll learn"],
    CONCEPT: ["concept", "Concept"],
    NOTE: ["note", "Note"],
    IMPORTANT: ["note", "Important"],
    TIP: ["tip", "Tip"],
    WARNING: ["warning", "Warning"],
    CAUTION: ["warning", "Caution"],
    DANGER: ["danger", "Danger"],
    LAB: ["lab", "Hands-on lab"],
    MAC: ["mac", "On your Mac"],
  };

  function buildCallouts(root) {
    $$("blockquote", root).forEach((bq) => {
      const first = bq.firstElementChild;
      if (!first || first.tagName !== "P") return;
      const html = first.innerHTML;
      const nl = html.indexOf("\n");
      const line = nl < 0 ? html : html.slice(0, nl);
      const m = line.match(/^\s*\[!([A-Za-z]+)\]\s*(.*)$/);
      if (!m) return;
      const def = CALLOUT[m[1].toUpperCase()];
      if (!def) return;
      const [kind, label] = def;
      const title = m[2].trim();
      const rest = nl < 0 ? "" : html.slice(nl + 1);
      if (rest.trim()) first.innerHTML = rest; else first.remove();

      const box = document.createElement("div");
      box.className = `callout ${kind}`;
      box.innerHTML = `<div class="callout-head"><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[kind]}</svg><span>${label}</span></div>${title ? `<p class="callout-title">${title}</p>` : ""}<div class="callout-body"></div>`;
      const body = $(".callout-body", box);
      while (bq.firstChild) body.appendChild(bq.firstChild);
      bq.replaceWith(box);
    });
  }

  // ---------- headings, links, tables ----------
  function slugify(text) {
    return text.toLowerCase().replace(/[`'"’]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || "section";
  }

  function decorate(root, lessonId) {
    const seen = new Map();
    $$("h2, h3", root).forEach((h) => {
      let slug = slugify(h.textContent);
      const n = seen.get(slug) || 0;
      seen.set(slug, n + 1);
      if (n) slug += "-" + n;
      h.id = slug;
      const a = document.createElement("a");
      a.className = "anchor";
      a.href = `#/${lessonId}#${slug}`;
      a.setAttribute("aria-label", "Link to this section");
      a.textContent = "#";
      h.prepend(a);
    });

    $$("a[href]", root).forEach((a) => {
      const href = a.getAttribute("href");
      if (/^https?:\/\//i.test(href)) {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
      }
    });

    $$("table", root).forEach((t) => {
      if (t.parentElement.classList.contains("table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      t.replaceWith(wrap);
      wrap.appendChild(t);
    });

    const h1 = $("h1", root);
    if (h1) {
      let p = h1.nextElementSibling;
      if (p && p.tagName === "P") p.classList.add("lead");
    }
  }

  // ---------- task lists (persist per lesson) ----------
  function wireTasks(root, lessonId) {
    const all = store.get(KEYS.tasks, {});
    const state = all[lessonId] || {};
    $$("li.task input", root).forEach((box, i) => {
      if (i in state) box.checked = !!state[i];
      box.closest("li").classList.toggle("checked", box.checked);
      box.addEventListener("change", () => {
        box.closest("li").classList.toggle("checked", box.checked);
        const cur = store.get(KEYS.tasks, {});
        cur[lessonId] = cur[lessonId] || {};
        cur[lessonId][i] = box.checked;
        store.set(KEYS.tasks, cur);
      });
    });
  }

  // ---------- mermaid ----------
  let mermaidLoader = null;
  let mermaidSeq = 0;

  function loadMermaid() {
    if (!mermaidLoader) {
      mermaidLoader = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = MERMAID_URL;
        s.onload = () => resolve(window.mermaid);
        s.onerror = () => { mermaidLoader = null; reject(new Error("mermaid failed to load")); };
        document.head.appendChild(s);
      });
    }
    return mermaidLoader;
  }

  function mermaidTheme() {
    const dark = document.documentElement.dataset.theme === "dark";
    const v = dark
      ? { background: "#131418", primaryColor: "#1c1d36", primaryTextColor: "#ececef", primaryBorderColor: "#5f5dd6", secondaryColor: "#1a1b20", secondaryTextColor: "#ececef", secondaryBorderColor: "#33343b", tertiaryColor: "#17181d", tertiaryTextColor: "#ececef", tertiaryBorderColor: "#33343b", lineColor: "#7a7a85", textColor: "#c9c9cf", mainBkg: "#1c1d36", nodeBorder: "#5f5dd6", clusterBkg: "#17181d", clusterBorder: "#33343b", edgeLabelBackground: "#131418", titleColor: "#ececef", actorBkg: "#1c1d36", actorBorder: "#5f5dd6", actorTextColor: "#ececef", actorLineColor: "#55555f", signalColor: "#c9c9cf", signalTextColor: "#ececef", labelBoxBkgColor: "#1c1d36", labelBoxBorderColor: "#5f5dd6", labelTextColor: "#ececef", loopTextColor: "#c9c9cf", noteBkgColor: "#2a2412", noteBorderColor: "#7a6431", noteTextColor: "#ececef", activationBkgColor: "#26274a", activationBorderColor: "#5f5dd6", sequenceNumberColor: "#0e0f12" }
      : { background: "#f8f8f6", primaryColor: "#eef0ff", primaryTextColor: "#1b1b1f", primaryBorderColor: "#a3a6f2", secondaryColor: "#f1f1ee", secondaryTextColor: "#1b1b1f", secondaryBorderColor: "#d4d4cf", tertiaryColor: "#ffffff", tertiaryTextColor: "#1b1b1f", tertiaryBorderColor: "#d4d4cf", lineColor: "#8a8a93", textColor: "#3f3f46", mainBkg: "#eef0ff", nodeBorder: "#a3a6f2", clusterBkg: "#ffffff", clusterBorder: "#d4d4cf", edgeLabelBackground: "#f8f8f6", titleColor: "#1b1b1f", actorBkg: "#eef0ff", actorBorder: "#a3a6f2", actorTextColor: "#1b1b1f", actorLineColor: "#b4b4bb", signalColor: "#3f3f46", signalTextColor: "#1b1b1f", labelBoxBkgColor: "#eef0ff", labelBoxBorderColor: "#a3a6f2", labelTextColor: "#1b1b1f", loopTextColor: "#3f3f46", noteBkgColor: "#fff7e0", noteBorderColor: "#e3c178", noteTextColor: "#1b1b1f", activationBkgColor: "#e3e5ff", activationBorderColor: "#a3a6f2", sequenceNumberColor: "#ffffff" };
    return { ...v, fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif", fontSize: "14px", darkMode: dark };
  }

  async function renderMermaid(root) {
    const wraps = $$(".mermaid-wrap", root);
    if (!wraps.length) return;
    let mermaid;
    try {
      mermaid = await loadMermaid();
    } catch (e) {
      wraps.forEach((w) => { $(".mermaid-out", w).innerHTML = `<pre class="mermaid-fallback">${esc($(".mermaid-src", w).textContent)}</pre>`; });
      return;
    }
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      themeVariables: mermaidTheme(),
      securityLevel: "antiscript",
      flowchart: { curve: "basis", padding: 12, htmlLabels: true },
      sequence: { mirrorActors: false },
    });
    for (const w of wraps) {
      if (!w.isConnected) continue;
      const src = $(".mermaid-src", w).textContent;
      const out = $(".mermaid-out", w);
      const id = "mmd-" + ++mermaidSeq;
      try {
        const { svg } = await mermaid.render(id, src);
        out.innerHTML = svg;
      } catch (e) {
        out.innerHTML = `<pre class="mermaid-fallback">${esc(src)}</pre>`;
        const stray = document.getElementById("d" + id) || document.getElementById(id);
        if (stray && !out.contains(stray)) stray.remove();
      }
    }
  }

  // ---------- theme ----------
  function setTheme(theme, persist = true) {
    document.documentElement.dataset.theme = theme;
    if (persist) store.set(KEYS.theme, theme);
    renderMermaid($("#article"));
    document.dispatchEvent(new CustomEvent("lal:theme", { detail: theme }));
  }
  const systemDark = matchMedia("(prefers-color-scheme: dark)");
  systemDark.addEventListener?.("change", (e) => {
    const saved = store.get(KEYS.theme, null);
    if (saved !== "light" && saved !== "dark") setTheme(e.matches ? "dark" : "light", false);
  });

  // ---------- sidebar ----------
  const CHEV = '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>';

  function renderNav() {
    const home = `<a class="nav-home" href="#/" data-home><svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V10z"/></svg>Overview</a>`;
    const parts = COURSE.parts.map((part) => `
      <div class="nav-part${openParts.has(part.id) ? " open" : ""}" data-part="${part.id}">
        <button class="nav-part-btn" type="button" aria-expanded="${openParts.has(part.id)}">${CHEV}<span class="label">${esc(part.title)}</span><span class="nav-count"></span></button>
        <ul class="nav-lessons">
          ${part.lessons.map((l) => `<li><a href="#/${l.id}" data-id="${l.id}"><span class="dot"></span><span>${esc(l.title)}</span></a></li>`).join("")}
        </ul>
      </div>`).join("");
    $("#nav").innerHTML = home + parts;
    updateProgressUI();
  }

  function setPartOpen(partId, open) {
    const el = $(`.nav-part[data-part="${partId}"]`);
    if (!el) return;
    el.classList.toggle("open", open);
    $(".nav-part-btn", el).setAttribute("aria-expanded", String(open));
    if (open) openParts.add(partId); else openParts.delete(partId);
    store.set(KEYS.open, [...openParts]);
  }

  function updateProgressUI() {
    const total = LESSONS.length;
    const count = LESSONS.filter((l) => done.has(l.id)).length;
    $("#progress-text").textContent = `${count} / ${total}`;
    $("#progress-fill").style.width = `${total ? (count / total) * 100 : 0}%`;
    $$(".nav-lessons a").forEach((a) => a.classList.toggle("done", done.has(a.dataset.id)));
    COURSE.parts.forEach((part) => {
      const n = part.lessons.filter((l) => done.has(l.id)).length;
      const el = $(`.nav-part[data-part="${part.id}"] .nav-count`);
      if (el) {
        el.textContent = `${n}/${part.lessons.length}`;
        el.classList.toggle("full", n === part.lessons.length);
      }
    });
  }

  function markActive(id) {
    $$(".nav-lessons a").forEach((a) => a.classList.toggle("active", a.dataset.id === id));
    $(".nav-home")?.classList.toggle("active", !id);
    if (!id) return;
    const lesson = byId.get(id);
    if (lesson && !openParts.has(lesson.part.id)) setPartOpen(lesson.part.id, true);
    const link = $(`.nav-lessons a[data-id="${CSS.escape(id)}"]`);
    if (link) {
      const nav = $("#nav");
      const r = link.getBoundingClientRect();
      const nr = nav.getBoundingClientRect();
      if (r.top < nr.top + 40 || r.bottom > nr.bottom - 40) link.scrollIntoView({ block: "center" });
    }
  }

  function toggleDone(id) {
    if (done.has(id)) done.delete(id); else done.add(id);
    store.set(KEYS.done, [...done]);
    updateProgressUI();
  }

  // ---------- table of contents ----------
  let tocObserver = null;

  function buildToc(root) {
    const toc = $("#toc");
    if (tocObserver) { tocObserver.disconnect(); tocObserver = null; }
    const heads = $$("h2, h3", root).filter((h) => !h.closest(".callout"));
    if (heads.length < 2) { toc.innerHTML = ""; return; }
    const lessonId = currentRoute().id;
    toc.innerHTML = `<div class="toc-title">On this page</div><ul>${heads.map((h) => {
      const text = h.textContent.replace(/^#/, "").trim();
      return `<li class="${h.tagName.toLowerCase()}"><a href="#/${lessonId}#${h.id}" data-target="${h.id}">${esc(text)}</a></li>`;
    }).join("")}</ul>`;

    const links = new Map($$("a[data-target]", toc).map((a) => [a.dataset.target, a]));
    const visible = new Set();
    tocObserver = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) visible.add(e.target.id); else visible.delete(e.target.id); });
      let active = null;
      for (const h of heads) {
        if (visible.has(h.id)) { active = h.id; break; }
      }
      if (!active) {
        const above = heads.filter((h) => h.getBoundingClientRect().top < 120);
        active = above.length ? above[above.length - 1].id : heads[0].id;
      }
      links.forEach((a, id) => a.classList.toggle("active", id === active));
    }, { rootMargin: "-72px 0px -65% 0px" });
    heads.forEach((h) => tocObserver.observe(h));
  }

  // ---------- content loading ----------
  const mdCache = new Map();

  async function fetchLesson(id) {
    if (mdCache.has(id)) return mdCache.get(id);
    const res = await fetch(`content/${id}.md`, { cache: "no-cache" });
    if (!res.ok) throw new Error(String(res.status));
    const text = await res.text();
    mdCache.set(id, text);
    return text;
  }

  function readingMinutes(md) {
    const words = md.replace(/```[\s\S]*?```/g, " ").split(/\s+/).filter(Boolean).length;
    return Math.max(1, Math.round(words / 220));
  }

  function currentRoute() {
    const raw = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
    const i = raw.indexOf("#");
    const id = (i < 0 ? raw : raw.slice(0, i)).replace(/\/+$/, "");
    const anchor = i < 0 ? "" : raw.slice(i + 1);
    return { id, anchor };
  }

  let renderToken = 0;
  let shownId = null;

  async function route() {
    const { id, anchor } = currentRoute();
    closeNav();
    if (id && id === shownId) {
      scrollToAnchor(anchor, true);
      return;
    }
    if (!id) return showHome();
    const lesson = byId.get(id);
    if (!lesson) return showMissing(id);
    await showLesson(lesson, anchor);
  }

  async function showLesson(lesson, anchor) {
    const token = ++renderToken;
    const article = $("#article");
    let md = null;
    try { md = await fetchLesson(lesson.id); } catch (e) { md = null; }
    if (token !== renderToken) return;

    shownId = lesson.id;
    store.set(KEYS.last, lesson.id);
    document.title = `${lesson.title} · ${COURSE.title}`;

    const pos = `Lesson ${lesson.indexInPart + 1} of ${lesson.part.lessons.length}`;
    if (md == null) {
      article.innerHTML = `<div class="kicker"><span>${esc(lesson.part.title)}</span></div><h1>${esc(lesson.title)}</h1><div class="placeholder">This lesson hasn't been written yet. Check back soon.</div>`;
    } else {
      const html = marked.parse(md);
      const kicker = `<div class="kicker"><span>${esc(lesson.part.title)}</span><span class="sep">/</span><span class="meta">${pos}</span><span class="sep">/</span><span class="meta">${readingMinutes(md)} min read</span></div>`;
      article.innerHTML = /^\s*<h1/i.test(html) ? kicker + html : kicker + `<h1>${esc(lesson.title)}</h1>` + html;
      buildCallouts(article);
      decorate(article, lesson.id);
      wireTasks(article, lesson.id);
      if (window.LAL_WIDGETS) window.LAL_WIDGETS.mount(article);
    }

    renderFoot(lesson);
    buildToc(article);
    markActive(lesson.id);
    if (anchor) requestAnimationFrame(() => scrollToAnchor(anchor, false));
    else window.scrollTo(0, 0);
    renderMermaid(article);
  }

  function scrollToAnchor(anchor, smooth) {
    if (!anchor) { window.scrollTo({ top: 0, behavior: smooth ? "smooth" : "auto" }); return; }
    const el = document.getElementById(anchor);
    if (el) el.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  }

  function renderFoot(lesson) {
    const prev = LESSONS[lesson.index - 1];
    const next = LESSONS[lesson.index + 1];
    const isDone = done.has(lesson.id);
    $("#lesson-foot").innerHTML = `
      <div class="complete-row">
        <p>${isDone ? "Nice work. This lesson is marked complete." : "Finished the reading and the lab?"}</p>
        <button class="btn ${isDone ? "done" : "primary"}" type="button" data-action="toggle-done">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
          ${isDone ? "Completed" : "Mark as complete"}
        </button>
      </div>
      <nav class="pager" aria-label="Lesson navigation">
        ${prev ? `<a class="prev" href="#/${prev.id}"><span class="dir">&larr; Previous</span><span class="t">${esc(prev.title)}</span></a>` : ""}
        ${next ? `<a class="next" href="#/${next.id}"><span class="dir">Next &rarr;</span><span class="t">${esc(next.title)}</span></a>` : ""}
      </nav>`;
  }

  function showHome() {
    ++renderToken;
    shownId = null;
    document.title = `${COURSE.title}: ${COURSE.tagline}`;
    const total = LESSONS.length;
    const count = LESSONS.filter((l) => done.has(l.id)).length;
    const last = byId.get(store.get(KEYS.last, ""));
    const nextUp = LESSONS.find((l) => !done.has(l.id)) || LESSONS[0];
    const resume = last && !done.has(last.id) ? last : nextUp;
    const started = count > 0 || !!last;

    $("#article").innerHTML = `
      <section class="home-hero">
        <div class="eyebrow">A private study lab for Sethu</div>
        <h1>From your first local model to editing its weights.</h1>
        <p>A step-by-step path through how LLMs work, mastering LM Studio on a 16 GB Mac, building agents, using AI for security research, fine-tuning, and weight-space surgery. Every concept is explained before it is used, and every module ends with something you build.</p>
        <div class="home-actions">
          <a class="btn primary" href="#/${resume.id}">${started ? "Continue: " + esc(resume.title) : "Start learning"} &rarr;</a>
          <a class="btn" href="#/reference/roadmap">See the 14-week plan</a>
        </div>
        <div class="home-stats">
          <div><b>${COURSE.parts.length - 2}</b>core modules</div>
          <div><b>${total}</b>lessons</div>
          <div><b>${count}</b>completed</div>
          <div><b>${total ? Math.round((count / total) * 100) : 0}%</b>progress</div>
        </div>
      </section>
      <h2 style="border-top:0;padding-top:0;margin-top:8px">Modules</h2>
      <div class="part-grid">
        ${COURSE.parts.map((part, i) => {
          const n = part.lessons.filter((l) => done.has(l.id)).length;
          const firstOpen = part.lessons.find((l) => !done.has(l.id)) || part.lessons[0];
          const num = part.title.match(/^(\d+)\s*·\s*(.*)$/);
          return `<a class="part-card" href="#/${firstOpen.id}">
            <span class="num">${num ? "Module " + num[1] : esc(part.title)}</span>
            <span class="name">${esc(num ? num[2] : part.title)}</span>
            <span class="desc">${esc(PART_BLURBS[part.id] || "")}</span>
            <span class="bar"><i style="width:${(n / part.lessons.length) * 100}%"></i></span>
            <span class="count">${n} of ${part.lessons.length} lessons done</span>
          </a>`;
        }).join("")}
      </div>`;
    $("#lesson-foot").innerHTML = "";
    $("#toc").innerHTML = "";
    if (tocObserver) { tocObserver.disconnect(); tocObserver = null; }
    markActive(null);
    window.scrollTo(0, 0);
  }

  function showMissing(id) {
    ++renderToken;
    shownId = null;
    $("#article").innerHTML = `<h1>Page not found</h1><p class="lead">There is no lesson called <code>${esc(id)}</code>.</p><p><a href="#/">Back to the overview</a></p>`;
    $("#lesson-foot").innerHTML = "";
    $("#toc").innerHTML = "";
    markActive(null);
  }

  // ---------- search ----------
  let searchIndex = null;
  let searchBuilding = null;
  let results = [];
  let activeResult = 0;

  function stripMd(md) {
    return md
      .replace(/```mermaid[\s\S]*?```/g, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/^>\s?\[![A-Z]+\]\s*/gm, "")
      .replace(/^>\s?/gm, "")
      .replace(/[*_`#|]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function buildIndex() {
    if (searchIndex) return Promise.resolve(searchIndex);
    if (searchBuilding) return searchBuilding;
    searchBuilding = Promise.all(LESSONS.map(async (lesson) => {
      let md = "";
      try { md = await fetchLesson(lesson.id); } catch (e) { return []; }
      const sections = [];
      const lines = md.split("\n");
      let heading = lesson.title, anchor = "", buf = [];
      const seen = new Map();
      let inFence = false;
      const flush = () => {
        const text = stripMd(buf.join("\n"));
        if (text || anchor === "") sections.push({ lesson, heading, anchor, text, lower: text.toLowerCase(), hlower: heading.toLowerCase() });
        buf = [];
      };
      for (const line of lines) {
        if (/^```/.test(line)) inFence = !inFence;
        const m = !inFence && line.match(/^(##|###)\s+(.*)$/);
        if (m) {
          flush();
          heading = m[2].replace(/[`*_]/g, "").trim();
          let slug = slugify(heading);
          const n = seen.get(slug) || 0;
          seen.set(slug, n + 1);
          if (n) slug += "-" + n;
          anchor = slug;
        } else if (!/^#\s/.test(line)) {
          buf.push(line);
        }
      }
      flush();
      return sections;
    })).then((all) => (searchIndex = all.flat()));
    return searchBuilding;
  }

  function snippet(text, terms) {
    const lower = text.toLowerCase();
    let pos = -1;
    for (const t of terms) { pos = lower.indexOf(t); if (pos >= 0) break; }
    const start = Math.max(0, pos - 60);
    let s = (start > 0 ? "…" : "") + text.slice(start, start + 200) + (start + 200 < text.length ? "…" : "");
    s = esc(s);
    for (const t of terms) {
      if (t.length < 2) continue;
      s = s.replace(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), (x) => `<mark>${x}</mark>`);
    }
    return s;
  }

  function runSearch(q) {
    const box = $("#search-results");
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      results = LESSONS.slice(0, 0);
      box.innerHTML = `<div class="empty">Type to search across all ${LESSONS.length} lessons.</div>`;
      return;
    }
    if (!searchIndex) {
      box.innerHTML = `<div class="empty">Indexing lessons…</div>`;
      buildIndex().then(() => { if ($("#search-input").value.trim().toLowerCase().split(/\s+/).join(" ") === terms.join(" ")) runSearch(q); });
      return;
    }
    const scored = [];
    for (const s of searchIndex) {
      const ltitle = s.lesson.title.toLowerCase();
      let score = 0, ok = true;
      for (const t of terms) {
        const inBody = s.lower.indexOf(t) >= 0, inHead = s.hlower.indexOf(t) >= 0, inTitle = ltitle.indexOf(t) >= 0;
        if (!inBody && !inHead && !inTitle) { ok = false; break; }
        if (inTitle) score += 12;
        if (inHead) score += 8;
        if (inBody) score += Math.min(6, s.lower.split(t).length - 1);
      }
      if (ok) scored.push({ s, score });
    }
    scored.sort((a, b) => b.score - a.score);
    results = scored.slice(0, 30).map((x) => x.s);
    activeResult = 0;
    if (!results.length) { box.innerHTML = `<div class="empty">No results for “${esc(q)}”.</div>`; return; }
    box.innerHTML = results.map((s, i) => `
      <a class="sr-item${i === 0 ? " active" : ""}" href="#/${s.lesson.id}${s.anchor ? "#" + s.anchor : ""}" data-i="${i}">
        <div class="sr-path">${esc(s.lesson.part.title)} / ${esc(s.lesson.title)}</div>
        <div class="sr-title">${esc(s.heading)}</div>
        <div class="sr-snip">${snippet(s.text, terms)}</div>
      </a>`).join("");
  }

  function setActiveResult(i) {
    const items = $$(".sr-item");
    if (!items.length) return;
    activeResult = (i + items.length) % items.length;
    items.forEach((el, j) => el.classList.toggle("active", j === activeResult));
    items[activeResult].scrollIntoView({ block: "nearest" });
  }

  let lastFocus = null;
  function openSearch() {
    lastFocus = document.activeElement;
    const modal = $("#search");
    modal.hidden = false;
    const input = $("#search-input");
    input.value = "";
    runSearch("");
    input.focus();
    buildIndex();
  }
  function closeSearch() {
    $("#search").hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  // ---------- mobile nav ----------
  function openNav() { document.body.classList.add("nav-open"); }
  function closeNav() { document.body.classList.remove("nav-open"); }

  // ---------- copy ----------
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      ta.remove();
      return ok;
    }
  }

  // ---------- events ----------
  document.addEventListener("click", async (e) => {
    const t = e.target;

    const copy = t.closest(".copy-btn");
    if (copy) {
      const code = copy.closest(".code").querySelector("pre code");
      const ok = await copyText(code.textContent.replace(/\n$/, ""));
      const label = copy.querySelector("span");
      copy.classList.toggle("copied", ok);
      label.textContent = ok ? "Copied" : "Press Ctrl+C";
      setTimeout(() => { copy.classList.remove("copied"); label.textContent = "Copy"; }, 1600);
      return;
    }

    const action = t.closest("[data-action]");
    if (action) {
      const a = action.dataset.action;
      if (a === "theme") setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
      else if (a === "search") openSearch();
      else if (a === "toggle-done" && shownId) {
        toggleDone(shownId);
        renderFoot(byId.get(shownId));
      }
      return;
    }

    const partBtn = t.closest(".nav-part-btn");
    if (partBtn) {
      const part = partBtn.closest(".nav-part");
      setPartOpen(part.dataset.part, !part.classList.contains("open"));
      return;
    }

    const sr = t.closest(".sr-item");
    if (sr) { closeSearch(); return; }

    if (t.closest("#menu-btn")) { openNav(); return; }
    if (t.closest("#scrim")) { closeNav(); return; }
    if (t.id === "search") { closeSearch(); return; }

    const tocLink = t.closest(".toc a[data-target]");
    if (tocLink) {
      e.preventDefault();
      const el = document.getElementById(tocLink.dataset.target);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        history.replaceState(null, "", tocLink.getAttribute("href"));
      }
    }
  });

  $("#search-input").addEventListener("input", (e) => runSearch(e.target.value));

  document.addEventListener("keydown", (e) => {
    const searching = !$("#search").hidden;
    if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (searching) closeSearch(); else openSearch();
      return;
    }
    if (searching) {
      if (e.key === "Escape") { e.preventDefault(); closeSearch(); }
      else if (e.key === "ArrowDown") { e.preventDefault(); setActiveResult(activeResult + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActiveResult(activeResult - 1); }
      else if (e.key === "Enter") {
        const item = $$(".sr-item")[activeResult];
        if (item) { e.preventDefault(); location.hash = item.getAttribute("href"); closeSearch(); }
      }
      return;
    }
    if (e.key === "Escape") closeNav();
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
    if (e.key === "/" && !typing) { e.preventDefault(); openSearch(); }
  });

  window.addEventListener("hashchange", route);

  // ---------- boot ----------
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  $("#search-kbd").textContent = isMac ? "⌘ K" : "Ctrl K";
  renderNav();
  route();
})();
