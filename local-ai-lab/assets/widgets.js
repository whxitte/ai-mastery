/* Local AI Lab — interactive widgets: <div data-widget="sampler|memory|tokens"></div> */
(() => {
  "use strict";

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const softmax = (xs) => {
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  };
  const pct = (p) => (p >= 0.995 ? "100%" : p >= 0.1 ? (p * 100).toFixed(0) + "%" : p >= 0.001 ? (p * 100).toFixed(1) + "%" : "<0.1%");
  const field = (label, inner, valueAttr) => `<div class="w-field"><label><span>${label}</span><span class="val" ${valueAttr}></span></label>${inner}</div>`;

  // ============ 1. Sampler ============

  const CONTEXTS = [
    {
      name: "Confident",
      prompt: "To list the files in a directory on Linux, run",
      tokens: [["ls", 7.4], ["the", 4.6], ["`ls", 4.3], ["find", 3.1], ["dir", 2.3], ["tree", 2.0], ["stat", 0.9], ["cat", 0.6], ["pwd", 0.1], ["echo", -0.5], ["rm", -1.2], ["banana", -4.0]],
    },
    {
      name: "Balanced",
      prompt: "The attacker hid the malicious",
      tokens: [["payload", 5.1], ["code", 4.6], ["script", 4.2], ["file", 3.6], ["link", 3.3], ["macro", 2.9], ["binary", 2.5], ["request", 2.0], ["comment", 1.4], ["image", 1.0], ["pizza", -1.2], ["sonnet", -2.6]],
    },
    {
      name: "Open-ended",
      prompt: "My favourite colour is",
      tokens: [["blue", 3.2], ["green", 2.95], ["red", 2.75], ["purple", 2.55], ["black", 2.3], ["yellow", 2.05], ["orange", 1.85], ["pink", 1.7], ["white", 1.45], ["teal", 1.05], ["grey", 0.9], ["ultraviolet", -0.6]],
    },
  ];

  function sampler(el) {
    const st = { ctx: 1, T: 0.7, k: 0, p: 1, minp: 0, order: "last", hits: null, picked: "" };
    el.classList.add("w");
    el.innerHTML = `
      <div class="w-head"><b>Next-token sampler</b><span>Move the knobs and watch which tokens survive</span></div>
      <div class="w-body">
        <div class="w-chips" data-ctx>${CONTEXTS.map((c, i) => `<button type="button" class="w-chip" data-i="${i}">${c.name}</button>`).join("")}</div>
        <div class="w-prompt" style="margin-top:12px"><span data-prompt></span> <span class="gen" data-gen></span><span class="cursor"></span></div>
        <div class="dist" data-dist></div>
        <div class="dist-legend">
          <span><i style="background:color-mix(in srgb, var(--faint) 35%, transparent)"></i>raw probability (temperature 1, no filters)</span>
          <span><i style="background:var(--accent)"></i>final probability with your settings</span>
        </div>
        <div class="w-sep"></div>
        <div class="w-grid">
          ${field("Temperature", '<input type="range" min="0" max="2" step="0.05" data-k="T">', 'data-v="T"')}
          ${field("Top-k (0 = off)", '<input type="range" min="0" max="20" step="1" data-k="k">', 'data-v="k"')}
          ${field("Top-p (1 = off)", '<input type="range" min="0.05" max="1" step="0.01" data-k="p">', 'data-v="p"')}
          ${field("Min-p (0 = off)", '<input type="range" min="0" max="0.5" step="0.01" data-k="minp">', 'data-v="minp"')}
        </div>
        <div class="w-row" style="margin-top:14px">
          <span style="font-size:12.5px;color:var(--muted)">Temperature is applied</span>
          <div class="w-chips" data-order>
            <button type="button" class="w-chip" data-o="last">after the filters (llama.cpp default)</button>
            <button type="button" class="w-chip" data-o="first">before the filters</button>
          </div>
        </div>
        <div class="w-row" style="margin-top:12px">
          <button type="button" class="w-btn" data-preset="think">Qwen thinking</button>
          <button type="button" class="w-btn" data-preset="nothink">Qwen non-thinking</button>
          <button type="button" class="w-btn" data-preset="greedy">Greedy</button>
          <button type="button" class="w-btn" data-preset="off">All filters off</button>
        </div>
        <div class="w-row" style="margin-top:10px">
          <button type="button" class="w-btn" data-sample="1">Sample once</button>
          <button type="button" class="w-btn primary" data-sample="1000">Sample 1,000&times;</button>
        </div>
        <p class="w-note" data-summary></p>
      </div>`;

    const q = (s) => el.querySelector(s);
    const inputs = Array.from(el.querySelectorAll("input[data-k]"));

    function compute() {
      const ctx = CONTEXTS[st.ctx];
      const toks = ctx.tokens.map(([t, l]) => ({ t, l, cut: null, p: 0 }));
      const raw = softmax(toks.map((x) => x.l));
      toks.forEach((x, i) => (x.raw = raw[i]));
      toks.sort((a, b) => b.l - a.l);

      const greedy = st.T < 0.001;
      const probs = (list, temp) => softmax(list.map((x) => x.l / temp));
      const filterT = st.order === "first" && !greedy ? st.T : 1;
      let alive = toks.slice();

      if (st.order === "first" && greedy) {
        alive.slice(1).forEach((x) => (x.cut = "T = 0"));
        alive = alive.slice(0, 1);
      }
      if (st.k > 0 && alive.length > st.k) {
        alive.slice(st.k).forEach((x) => (x.cut = "top-k"));
        alive = alive.slice(0, st.k);
      }
      if (st.p < 1 && alive.length > 1) {
        const ps = probs(alive, filterT);
        let cum = 0, keep = alive.length;
        for (let i = 0; i < alive.length; i++) {
          cum += ps[i];
          if (cum >= st.p) { keep = i + 1; break; }
        }
        alive.slice(keep).forEach((x) => (x.cut = "top-p"));
        alive = alive.slice(0, keep);
      }
      if (st.minp > 0 && alive.length > 1) {
        const ps = probs(alive, filterT);
        const keep = [];
        alive.forEach((x, i) => (ps[i] >= st.minp * ps[0] ? keep.push(x) : (x.cut = "min-p")));
        alive = keep;
      }
      if (greedy) {
        alive.forEach((x, i) => { x.p = i === 0 ? 1 : 0; if (i) x.cut = "T = 0"; });
      } else {
        const ps = probs(alive, st.T);
        alive.forEach((x, i) => (x.p = ps[i]));
      }
      return toks;
    }

    function draw() {
      const ctx = CONTEXTS[st.ctx];
      const toks = compute();
      st._toks = toks;
      el.querySelectorAll("[data-ctx] .w-chip").forEach((b) => b.classList.toggle("on", +b.dataset.i === st.ctx));
      el.querySelectorAll("[data-order] .w-chip").forEach((b) => b.classList.toggle("on", b.dataset.o === st.order));
      inputs.forEach((inp) => { inp.value = st[inp.dataset.k]; });
      q('[data-v="T"]').textContent = st.T < 0.001 ? "0 (greedy)" : st.T.toFixed(2);
      q('[data-v="k"]').textContent = st.k === 0 ? "off" : st.k;
      q('[data-v="p"]').textContent = st.p >= 1 ? "off" : st.p.toFixed(2);
      q('[data-v="minp"]').textContent = st.minp <= 0 ? "off" : st.minp.toFixed(2);
      q("[data-prompt]").textContent = ctx.prompt;
      q("[data-gen]").textContent = st.picked;

      const totalHits = st.hits ? Object.values(st.hits).reduce((a, b) => a + b, 0) : 0;
      q("[data-dist]").innerHTML = toks.map((x) => {
        const hits = st.hits && st.hits[x.t] ? `<span class="hits">&times;${st.hits[x.t]}</span>` : "";
        return `<div class="dist-row${x.cut ? " cut" : ""}">
          <span class="tok" title="${esc(x.t)}">${esc(x.t)}</span>
          <span class="track"><span class="ghost" style="width:${(x.raw * 100).toFixed(2)}%"></span><span class="fill" style="width:${x.cut ? 0 : (x.p * 100).toFixed(2)}%"></span>${hits}</span>
          <span class="pct">${x.cut ? "cut: " + x.cut : pct(x.p)}</span>
        </div>`;
      }).join("");

      const alive = toks.filter((x) => !x.cut);
      const top = alive[0];
      let note = `${alive.length} of ${toks.length} tokens can be picked. Most likely: <b>${esc(top.t)}</b> at ${pct(top.p)}.`;
      if (totalHits) note += ` After ${totalHits.toLocaleString()} samples, <b>${esc(top.t)}</b> was chosen ${((st.hits[top.t] || 0) / totalHits * 100).toFixed(1)}% of the time.`;
      q("[data-summary]").innerHTML = note;
    }

    function sample(n) {
      const alive = (st._toks || compute()).filter((x) => !x.cut);
      const hits = {};
      let last = alive[0].t;
      for (let i = 0; i < n; i++) {
        let r = Math.random(), acc = 0;
        last = alive[alive.length - 1].t;
        for (const x of alive) { acc += x.p; if (r <= acc) { last = x.t; break; } }
        hits[last] = (hits[last] || 0) + 1;
      }
      st.picked = last;
      st.hits = n > 1 ? hits : null;
      draw();
    }

    const reset = () => { st.hits = null; st.picked = ""; };

    el.addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      st[k] = parseFloat(e.target.value);
      reset();
      draw();
    });
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.i != null) { st.ctx = +b.dataset.i; reset(); }
      else if (b.dataset.o) { st.order = b.dataset.o; reset(); }
      else if (b.dataset.preset) {
        const P = { think: [0.6, 20, 0.95, 0], nothink: [0.7, 20, 0.8, 0], greedy: [0, 0, 1, 0], off: [1, 0, 1, 0] }[b.dataset.preset];
        [st.T, st.k, st.p, st.minp] = P;
        reset();
      } else if (b.dataset.sample) { sample(+b.dataset.sample); return; }
      draw();
    });
    draw();
  }

  // ============ 2. Memory calculator ============

  const GiB = 1024 ** 3;
  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
  const KV_TYPES = { F16: 2, Q8_0: 1.0625, Q4_0: 0.5625 };
  const BPW = { Q3_K_M: 3.9, IQ4_XS: 4.25, Q4_K_M: 4.85, Q5_K_M: 5.7, Q6_K: 6.56, Q8_0: 8.5, BF16: 16 };
  const MODELS = {
    q35: { label: "Qwen3.5-9B (your model, hybrid)", params: 8.95, kvLayers: 8, kvHeads: 4, headDim: 256, stateGiB: 0.05, vision: 0.82,
      files: { Q4_K_M: 5.24, Q6_K: 6.85, Q8_0: 8.88, BF16: 16.67 } },
    classic: { label: "Classic 8B transformer (Qwen3-8B)", params: 8.19, kvLayers: 36, kvHeads: 8, headDim: 128, stateGiB: 0, vision: 0 },
    custom: { label: "Custom model", params: 4, kvLayers: 36, kvHeads: 8, headDim: 128, stateGiB: 0, vision: 0 },
  };
  const fmtG = (g) => (g >= 10 ? g.toFixed(1) : g.toFixed(2)) + " GB";
  const fmtCtx = (n) => (n >= 1024 ? n / 1024 + "K" : n);

  function memory(el) {
    const st = { model: "q35", quant: "Q4_K_M", ctx: 3, kv: "F16", vision: false, ram: 16, apps: 4.5, custom: { ...MODELS.custom } };
    el.classList.add("w");
    el.innerHTML = `
      <div class="w-head"><b>Will it fit? Memory calculator</b><span>Estimates for llama.cpp / LM Studio on a Mac</span></div>
      <div class="w-body">
        <div class="w-grid">
          ${field("Model", `<select data-k="model">${Object.entries(MODELS).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join("")}</select>`, "")}
          ${field("Weights quantization", '<select data-k="quant"></select>', 'data-v="quant"')}
          ${field("Context length", `<input type="range" min="0" max="${CTX_STEPS.length - 1}" step="1" data-k="ctx">`, 'data-v="ctx"')}
          ${field("KV cache precision", `<select data-k="kv">${Object.keys(KV_TYPES).map((k) => `<option value="${k}">${k}${k === "F16" ? " (default)" : ""}</option>`).join("")}</select>`, "")}
          ${field("Mac unified memory", `<select data-k="ram">${[8, 16, 24, 32, 48, 64, 96, 128].map((r) => `<option value="${r}">${r} GB</option>`).join("")}</select>`, "")}
          ${field("macOS + your other apps", '<input type="range" min="2" max="10" step="0.5" data-k="apps">', 'data-v="apps"')}
        </div>
        <div class="w-grid" data-custom style="margin-top:14px">
          ${field("Parameters (billions)", '<input type="number" min="0.1" max="1000" step="0.1" data-c="params">', "")}
          ${field("Layers with a KV cache", '<input type="number" min="1" max="200" step="1" data-c="kvLayers">', "")}
          ${field("KV heads", '<input type="number" min="1" max="128" step="1" data-c="kvHeads">', "")}
          ${field("Head dimension", '<input type="number" min="16" max="512" step="16" data-c="headDim">', "")}
        </div>
        <label class="w-row" data-vision-row style="margin-top:12px;font-size:13px;cursor:pointer"><input type="checkbox" data-k="vision"> Load the vision encoder (mmproj, needed only for images)</label>
        <div class="w-sep"></div>
        <div class="mem-bar" data-bar></div>
        <div class="mem-scale" data-scale></div>
        <div class="mem-legend" data-legend></div>
        <div class="verdict" data-verdict></div>
        <p class="w-note">1 GB here means 1 GiB (1024&sup3; bytes). Compute buffers vary with batch size and backend, so treat the total as a planning estimate and confirm with LM Studio's own estimate and Activity Monitor. The GPU limit shown is macOS's approximate default.</p>
      </div>`;

    const q = (s) => el.querySelector(s);

    function model() { return st.model === "custom" ? st.custom : MODELS[st.model]; }

    function quantOptions() {
      const m = model();
      const keys = m.files ? Object.keys(m.files) : Object.keys(BPW);
      if (!keys.includes(st.quant)) st.quant = keys.includes("Q4_K_M") ? "Q4_K_M" : keys[0];
      q('[data-k="quant"]').innerHTML = keys.map((k) => `<option value="${k}">${k}${m.files ? " (" + fmtG(m.files[k]) + " file)" : " (~" + BPW[k] + " bits/weight)"}</option>`).join("");
    }

    function calc() {
      const m = model();
      const ctx = CTX_STEPS[st.ctx];
      const weights = m.files ? m.files[st.quant] : (m.params * 1e9 * BPW[st.quant]) / 8 / GiB;
      const kv = (2 * m.kvLayers * m.kvHeads * m.headDim * KV_TYPES[st.kv] * ctx) / GiB;
      const buffers = 0.45 + (ctx / 131072) * 0.5 + (m.stateGiB || 0);
      const vision = st.vision && m.vision ? m.vision : 0;
      const modelTotal = weights + kv + buffers + vision;
      const gpuLimit = st.ram * (st.ram > 36 ? 0.75 : 0.67);
      return { ctx, weights, kv, buffers, vision, modelTotal, gpuLimit, all: modelTotal + st.apps, perTokenKiB: (2 * m.kvLayers * m.kvHeads * m.headDim * KV_TYPES[st.kv]) / 1024 };
    }

    function draw() {
      const m = model();
      q('[data-k="model"]').value = st.model;
      q('[data-k="quant"]').value = st.quant;
      q('[data-k="ctx"]').value = st.ctx;
      q('[data-k="kv"]').value = st.kv;
      q('[data-k="ram"]').value = st.ram;
      q('[data-k="apps"]').value = st.apps;
      q('[data-k="vision"]').checked = st.vision;
      q("[data-custom]").style.display = st.model === "custom" ? "" : "none";
      q("[data-vision-row]").style.display = m.vision ? "" : "none";
      el.querySelectorAll("[data-c]").forEach((inp) => { if (document.activeElement !== inp) inp.value = st.custom[inp.dataset.c]; });

      const r = calc();
      q('[data-v="ctx"]').textContent = `${fmtCtx(r.ctx)} tokens (~${Math.round((r.ctx * 0.75) / 1000)}K words)`;
      q('[data-v="apps"]').textContent = st.apps.toFixed(1) + " GB";

      const scale = Math.max(st.ram, r.all) * 1.04;
      const segs = [
        ["Model weights", r.weights, "var(--seg-1)"],
        ["KV cache", r.kv, "var(--seg-2)"],
        ["Compute buffers + state", r.buffers, "var(--seg-3)"],
        ["Vision encoder", r.vision, "var(--seg-5)"],
        ["macOS + other apps", st.apps, "var(--seg-4)"],
      ].filter((s) => s[1] > 0);
      q("[data-bar]").innerHTML = segs.map(([n, v, c]) => `<span title="${n}: ${fmtG(v)}" style="width:${(v / scale) * 100}%;background:${c}"></span>`).join("");
      q("[data-scale]").innerHTML =
        `<span class="mem-marker" style="left:${(r.gpuLimit / scale) * 100}%"><em>GPU limit ~${r.gpuLimit.toFixed(1)}</em></span>` +
        `<span class="mem-marker" style="left:${(st.ram / scale) * 100}%;border-left-style:solid"><em style="transform:translateX(-104%)">${st.ram} GB RAM</em></span>`;
      q("[data-legend]").innerHTML = segs.map(([n, v, c]) => `<div><span><i style="background:${c}"></i>${n}</span><b>${fmtG(v)}</b></div>`).join("") +
        `<div><span>KV cost per token</span><b>${r.perTokenKiB.toFixed(r.perTokenKiB < 10 ? 1 : 0)} KiB</b></div>` +
        `<div><span>Model total (GPU side)</span><b>${fmtG(r.modelTotal)}</b></div>`;

      const v = q("[data-verdict]");
      if (r.modelTotal > r.gpuLimit) {
        v.className = "verdict bad";
        v.innerHTML = `<b>Over the GPU limit.</b> The model side needs ${fmtG(r.modelTotal)} but macOS lets the GPU use about ${fmtG(r.gpuLimit)} by default. Expect a failed load or partial CPU offload (much slower). Lower the context, quantize the KV cache, or choose a smaller quant.`;
      } else if (r.all > st.ram) {
        v.className = "verdict bad";
        v.innerHTML = `<b>Swap territory.</b> Model plus apps is ${fmtG(r.all)} on a ${st.ram} GB machine. macOS will compress and swap memory, and generation speed will collapse. Close apps or shrink the model side.`;
      } else if (r.modelTotal > r.gpuLimit * 0.9 || r.all > st.ram * 0.92) {
        v.className = "verdict tight";
        v.innerHTML = `<b>Tight.</b> It should load, with only ${fmtG(Math.min(r.gpuLimit - r.modelTotal, st.ram - r.all))} of headroom. Watch memory pressure in Activity Monitor.`;
      } else {
        v.className = "verdict ok";
        v.innerHTML = `<b>Fits comfortably.</b> About ${fmtG(r.gpuLimit - r.modelTotal)} of GPU headroom and ${fmtG(st.ram - r.all)} of free RAM.`;
      }
    }

    el.addEventListener("input", (e) => {
      const t = e.target;
      if (t.dataset.c) {
        const val = parseFloat(t.value);
        if (val > 0) st.custom[t.dataset.c] = val;
        if (t.dataset.c === "params") quantOptions();
        draw();
        return;
      }
      const k = t.dataset.k;
      if (!k) return;
      if (k === "vision") st.vision = t.checked;
      else if (k === "model") { st.model = t.value; quantOptions(); }
      else if (k === "quant" || k === "kv") st[k] = t.value;
      else st[k] = parseFloat(t.value);
      draw();
    });
    quantOptions();
    draw();
  }

  // ============ 3. Token estimator ============

  // Qwen-style pre-tokenizer (the regex split that happens before BPE merges)
  const PRE = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
  const SAMPLES = {
    English: "Local models keep your client data on your own machine. Nothing leaves the laptop.",
    "curl": 'curl -s -H "Authorization: Bearer $TOKEN" https://api.example.com/v1/users?id=1337 | jq .',
    JWT: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    Python: "def is_admin(user):\n    return user.get(\"role\") == \"admin\"  # TODO: check the token",
    Malayalam: "എന്റെ പേര് സേതു. ഞാൻ ഒരു സുരക്ഷാ ഗവേഷകനാണ്.",
    Numbers: "Port 8443 answered 1,048,576 requests in 2026.",
  };

  function estimate(piece) {
    if (/^\s+$/.test(piece)) return 1;
    if (/^\p{N}$/u.test(piece)) return 1;
    const letters = piece.replace(/[^\p{L}]/gu, "");
    if (letters.length) {
      if (/^[A-Za-z]+$/.test(letters)) return letters.length <= 8 ? 1 : 1 + Math.ceil((letters.length - 8) / 5);
      if (/^[\p{Script=Latin}]+$/u.test(letters)) return Math.max(1, Math.ceil(letters.length / 4));
      return Math.max(1, Math.ceil(letters.length / 2.5));
    }
    return Math.max(1, Math.ceil(piece.trim().length / 2));
  }

  function tokens(el) {
    el.classList.add("w");
    el.innerHTML = `
      <div class="w-head"><b>Token explorer</b><span>How text is split before BPE merges</span></div>
      <div class="w-body">
        <div class="w-chips" data-samples>${Object.keys(SAMPLES).map((k) => `<button type="button" class="w-chip" data-s="${esc(k)}">${esc(k)}</button>`).join("")}</div>
        <textarea data-text style="margin-top:12px" spellcheck="false"></textarea>
        <div class="tok-out" data-out></div>
        <div class="tok-stats" data-stats></div>
        <p class="w-note">Each chip is one <em>pre-token</em> from a Qwen-style splitting rule (spaces shown as &middot;, newlines as &crarr;). The real tokenizer then merges or splits each chip into vocabulary pieces, so the true count is at least the number of chips. The estimate is a heuristic; run the real tokenizer in the lab below for exact ids.</p>
      </div>`;
    const ta = el.querySelector("[data-text]");
    const colors = ["var(--seg-1)", "var(--seg-2)", "var(--seg-3)", "var(--seg-5)", "var(--c-tip)"];

    function draw() {
      const text = ta.value;
      const pieces = text.match(PRE) || [];
      const shown = pieces.slice(0, 500);
      el.querySelector("[data-out]").innerHTML = shown.map((p, i) => {
        const ws = /^\s+$/.test(p);
        const label = p.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
        return `<span class="${ws ? "ws" : ""}" style="background:color-mix(in srgb, ${colors[i % colors.length]} ${ws ? 6 : 16}%, transparent)">${esc(label)}</span>`;
      }).join("") + (pieces.length > shown.length ? `<span class="ws">&hellip; +${pieces.length - shown.length} more</span>` : "");
      const est = pieces.reduce((a, p) => a + estimate(p), 0);
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      el.querySelector("[data-stats]").innerHTML = `
        <div><b>${text.length.toLocaleString()}</b>characters</div>
        <div><b>${words.toLocaleString()}</b>words</div>
        <div><b>${pieces.length.toLocaleString()}</b>pre-tokens</div>
        <div><b>~${est.toLocaleString()}</b>tokens (estimate)</div>
        <div><b>${est ? (text.length / est).toFixed(1) : "0"}</b>chars per token</div>`;
    }

    el.addEventListener("click", (e) => {
      const b = e.target.closest("[data-s]");
      if (!b) return;
      ta.value = SAMPLES[b.dataset.s];
      el.querySelectorAll("[data-s]").forEach((x) => x.classList.toggle("on", x === b));
      draw();
    });
    ta.addEventListener("input", () => {
      el.querySelectorAll("[data-s]").forEach((x) => x.classList.remove("on"));
      draw();
    });
    ta.value = SAMPLES.English;
    el.querySelector('[data-s="English"]').classList.add("on");
    draw();
  }

  const WIDGETS = { sampler, memory, tokens };

  window.LAL_WIDGETS = {
    mount(root) {
      root.querySelectorAll("[data-widget]").forEach((el) => {
        const fn = WIDGETS[el.dataset.widget];
        if (!fn || el.dataset.mounted) return;
        el.dataset.mounted = "1";
        try { fn(el); } catch (e) { el.textContent = "This interactive widget failed to load."; console.error(e); }
      });
    },
  };
})();
