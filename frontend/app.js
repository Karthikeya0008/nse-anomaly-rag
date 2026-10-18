(() => {
  "use strict";

  // Served live by backend/src/server.py; the static file is the fallback when
  // the page is hosted without the server (then explanations can't be generated).
  const API_DATA_URL = "api/data?sector=NIFTY50";
  const STATIC_DATA_URL = "data/NIFTY50.json";

  const TYPES = {
    shock:  { label: "Shock",          symbol: "circle",  cssVar: "--shock",  note: "single-day extremes" },
    regime: { label: "Regime",         symbol: "rect",    cssVar: "--regime", note: "days in unusual periods" },
    both:   { label: "Shock + regime", symbol: "diamond", cssVar: "--both",   note: "both at once" },
  };
  const TYPE_ORDER = ["shock", "regime", "both"];

  const SOURCE_LABELS = { sebi: "SEBI", business_standard: "Business Standard", wikipedia: "Wikipedia" };
  const EXPLANATION_FIELDS = ["Likely Cause", "Supporting Evidence", "Confidence", "Caveats"];
  const CONFIDENCE_LEVELS = { low: 1, medium: 2, high: 3 };
  const RANGE_MONTHS = { "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60 };
  const CLICK_TOLERANCE_PX = 12;

  const state = {
    data: null,
    prices: [],            // [[timestamp, close], ...]
    anomalies: [],         // sorted ascending by date
    byDate: new Map(),
    returnByDate: new Map(),
    firstT: 0,
    lastT: 0,
    range: { from: 0, to: 0 },
    rangeKey: "ALL",
    types: new Set(TYPE_ORDER),
    explainedOnly: false,
    selected: null,        // date string
    chart: null,
    colors: null,
    applyingRange: false,
    canExplain: false,     // true when served by server.py
    service: null,         // /api/status response
    generating: new Set(), // dates with a request in flight
    errors: new Map(),     // date -> last error message
  };

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  function svg(tag, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function typeMark(type, size = 10) {
    const root = svg("svg", { width: size, height: size, viewBox: "0 0 10 10", class: "type-mark", "aria-hidden": "true" });
    const style = `fill: var(${TYPES[type].cssVar})`;
    if (type === "shock") root.append(svg("circle", { cx: 5, cy: 5, r: 4.5, style }));
    else if (type === "regime") root.append(svg("rect", { x: 0.75, y: 0.75, width: 8.5, height: 8.5, rx: 1, style }));
    else root.append(svg("polygon", { points: "5,0 10,5 5,10 0,5", style }));
    return root;
  }

  function typePill(type, text = TYPES[type].label, large = false) {
    return el("span", { class: `type-pill type-${type}${large ? " type-pill-lg" : ""}` }, typeMark(type, large ? 10 : 9), text);
  }

  // 24x24 stroke icons (paths only; drawn with currentColor).
  const ICONS = {
    calendar: "M7 3v3M17 3v3M4 9h16M5 5h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z",
    flag: "M5 21V4M5 4h11l-2 4 2 4H5",
    doc: "M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM14 3v5h5M9 13h6M9 17h4",
    chart: "M4 19h16M7 15l3-4 3 2 4-6",
    cause: "M9 18h6M10 21h4M12 3a6 6 0 0 0-3.6 10.8c.7.5 1.1 1.3 1.1 2.1V16h5v-.1c0-.8.4-1.6 1.1-2.1A6 6 0 0 0 12 3z",
    evidence: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4",
    confidence: "M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6l8-3zM9 12l2 2 4-4",
    caveats: "M12 4l9 16H3L12 4zM12 10v4M12 17h.01",
    link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
    sparkles: "M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z",
    refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
    info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v5M12 8h.01",
  };

  function icon(name, size = 16) {
    const root = svg("svg", { width: size, height: size, viewBox: "0 0 24 24", class: "icon", "aria-hidden": "true" });
    root.append(svg("path", { d: ICONS[name], fill: "none", stroke: "currentColor", "stroke-width": 1.9, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    return root;
  }

  const signClass = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : null);

  const ts = (iso) => Date.parse(iso + "T00:00:00Z");
  const isoFromTs = (t) => new Date(t).toISOString().slice(0, 10);

  const fmtShort = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  const fmtLong = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  const fmtWeekday = new Intl.DateTimeFormat("en-GB", { weekday: "long", timeZone: "UTC" });
  const fmtPrice = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtInt = new Intl.NumberFormat("en-IN");

  const shortDate = (iso) => fmtShort.format(ts(iso));
  const longDate = (iso) => fmtLong.format(ts(iso));
  const MINUS = "−";
  const signed = (x, digits = 2) => (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(x).toFixed(digits);
  const pct = (x, digits = 2) => signed(x * 100, digits) + "%";
  const simpleReturn = (logReturn) => Math.exp(logReturn) - 1;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /** "#rrggbb" -> "rgba(r, g, b, a)" */
  function withAlpha(hex, alpha) {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }

  function readColors() {
    const c = {};
    for (const name of ["ink", "ink-2", "muted", "grid", "axis", "border", "surface", "accent", "price", "band", "shock", "regime", "both"]) {
      c[name.replace("-", "")] = cssVar("--" + name);
    }
    return c;
  }

  function safeUrl(url) {
    return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
  }

  // ---------------------------------------------------------------------------
  // Data selection
  // ---------------------------------------------------------------------------
  function hasExplanation(a) {
    return Boolean(a.explanation_date && state.data.explanations[a.explanation_date]);
  }

  function passesFilters(a) {
    return state.types.has(a.anomaly_type) && (!state.explainedOnly || hasExplanation(a));
  }

  function inRange(t) {
    return t >= state.range.from && t <= state.range.to;
  }

  /** Anomalies passing the type/explanation filters (all dates). */
  const filtered = () => state.anomalies.filter(passesFilters);

  /** Anomalies passing all filters, including the visible date range. */
  const visible = () => state.anomalies.filter((a) => passesFilters(a) && inRange(a.t));

  // ---------------------------------------------------------------------------
  // Masthead, legend, filters
  // ---------------------------------------------------------------------------
  function renderProvenance() {
    const d = state.data;
    const rows = [
      ["Index", d.sector === "NIFTY50" ? "NIFTY 50" : d.sector],
      ["Data through", longDate(d.prices.date[d.prices.date.length - 1])],
      ["Trading days", fmtInt.format(d.prices.date.length)],
      ["Data exported", longDate(d.generated_at.slice(0, 10))],
    ];
    $("provenance").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
  }

  function renderLegend() {
    const priceKey = svg("svg", { width: 16, height: 10, "aria-hidden": "true", class: "type-mark" });
    priceKey.append(svg("line", { x1: 0, y1: 5, x2: 16, y2: 5, style: "stroke: var(--price); stroke-width: 2" }));
    const bandKey = svg("svg", { width: 14, height: 10, "aria-hidden": "true", class: "type-mark" });
    bandKey.append(svg("rect", { x: 0, y: 0, width: 14, height: 10, rx: 1, style: "fill: var(--band); stroke: var(--regime); stroke-opacity: 0.4" }));

    $("legend").replaceChildren(
      el("li", {}, priceKey, "Closing price"),
      ...TYPE_ORDER.map((t) => el("li", {}, typeMark(t), TYPES[t].label)),
      el("li", {}, bandKey, "Regime episode"),
    );
  }

  function renderTypeFilters() {
    $("type-filters").replaceChildren(...TYPE_ORDER.map((t) =>
      el("button", {
        type: "button",
        class: `chip type-${t}`,
        "aria-pressed": String(state.types.has(t)),
        onclick: () => {
          if (state.types.has(t)) state.types.delete(t); else state.types.add(t);
          renderTypeFilters();
          refresh();
        },
      }, typeMark(t), TYPES[t].label)
    ));
  }

  function setRangeButtons(key) {
    for (const b of $("range-buttons").querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.dataset.range === key));
    }
  }

  function rangeFor(key) {
    if (key === "ALL" || !RANGE_MONTHS[key]) return { from: state.firstT, to: state.lastT };
    const d = new Date(state.lastT);
    d.setUTCMonth(d.getUTCMonth() - RANGE_MONTHS[key]);
    return { from: Math.max(state.firstT, d.getTime()), to: state.lastT };
  }

  function applyRange(key) {
    state.rangeKey = key;
    state.range = rangeFor(key);
    setRangeButtons(key);
    state.applyingRange = true;
    state.chart.dispatchAction({ type: "dataZoom", startValue: state.range.from, endValue: state.range.to });
    state.applyingRange = false;
    renderStats();
    renderTable();
    renderDetail();
  }

  // ---------------------------------------------------------------------------
  // Stat tiles
  // ---------------------------------------------------------------------------
  function renderStats() {
    const days = state.prices.filter(([t]) => inRange(t)).length;
    const v = visible();
    const count = (type) => v.filter((a) => a.anomaly_type === type).length;
    const explained = v.filter(hasExplanation).length;
    const from = isoFromTs(Math.max(state.range.from, state.firstT));
    const to = isoFromTs(Math.min(state.range.to, state.lastT));

    const tile = (label, value, sub, { type = null, iconName = null, progress = null } = {}) =>
      el("div", { class: type ? `stat stat-${type}` : "stat" },
        el("div", { class: "stat-label" },
          el("span", { class: "stat-icon" }, type ? typeMark(type, 11) : icon(iconName, 15)),
          label),
        el("div", { class: "stat-value", text: value }),
        progress !== null
          ? el("div", { class: "progress", role: "presentation" }, el("span", { style: `width: ${Math.round(progress * 100)}%` }))
          : null,
        el("div", { class: "stat-sub", text: sub }),
      );

    $("stats").replaceChildren(
      tile("Trading days", fmtInt.format(days), `${shortDate(from)} – ${shortDate(to)}`, { iconName: "calendar" }),
      tile("Flagged days", fmtInt.format(v.length), days ? `${((v.length / days) * 100).toFixed(1)}% of trading days` : "—", { iconName: "flag" }),
      ...TYPE_ORDER.map((t) => tile(TYPES[t].label, fmtInt.format(count(t)), TYPES[t].note, { type: t })),
      tile("Explained", `${fmtInt.format(explained)} / ${fmtInt.format(v.length)}`, "with a generated explanation",
        { iconName: "doc", progress: v.length ? explained / v.length : 0 }),
    );
  }

  // ---------------------------------------------------------------------------
  // Table
  // ---------------------------------------------------------------------------
  function explanationStatus(a) {
    if (hasExplanation(a)) return { text: "Available", cls: "status-yes" };
    if (state.generating.has(a.date)) return { text: "Generating…", cls: "status-busy" };
    return { text: "Not generated", cls: "status-no" };
  }

  function agreementText(a) {
    const n = state.data.ensemble_size;
    if (a.anomaly_type === "shock") return `${a.shock_votes}/${n}`;
    if (a.anomaly_type === "regime") return `${a.regime_votes}/${n}`;
    return `${a.shock_votes}/${n} · ${a.regime_votes}/${n}`;
  }

  function renderTable() {
    const rows = visible().slice().reverse();
    $("table-note").textContent = rows.length
      ? `${fmtInt.format(rows.length)} days in the selected period, newest first. Agreement is the number of models (of ${state.data.ensemble_size}) that individually flagged the day.`
      : "No flagged days match the current filters.";

    if (!rows.length) {
      $("anomaly-rows").replaceChildren(el("tr", { class: "empty-row" }, el("td", { colspan: 7, text: "No flagged days in this period." })));
      return;
    }

    $("anomaly-rows").replaceChildren(...rows.map((a) => {
      const status = explanationStatus(a);
      const select = () => selectDay(a.date, { scrollRow: false });
      return el("tr", {
        tabindex: "0",
        "data-date": a.date,
        "aria-selected": String(a.date === state.selected),
        onclick: select,
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } },
      },
        el("td", { class: "date-cell", text: shortDate(a.date) }),
        el("td", {}, typePill(a.anomaly_type)),
        el("td", { class: "num", text: fmtPrice.format(a.close) }),
        el("td", { class: `num ${signClass(a.log_return) || ""}`, text: pct(simpleReturn(a.log_return)) }),
        el("td", { class: "num", text: signed(a.return_zscore) }),
        el("td", { class: "num", text: agreementText(a) }),
        el("td", {}, el("span", { class: `exp-status ${status.cls}`, text: status.text })),
      );
    }));
  }

  function syncTableSelection(scroll) {
    for (const tr of $("anomaly-rows").querySelectorAll("tr[data-date]")) {
      const on = tr.dataset.date === state.selected;
      tr.setAttribute("aria-selected", String(on));
      if (on && scroll) scrollRowIntoTable(tr);
    }
  }

  /** Scroll only the table's own container -- scrollIntoView would also move the page. */
  function scrollRowIntoTable(tr) {
    const wrap = tr.closest(".table-wrap");
    const header = wrap.querySelector("thead").offsetHeight;
    const top = tr.offsetTop - header;
    const bottom = tr.offsetTop + tr.offsetHeight;
    if (top < wrap.scrollTop) wrap.scrollTop = top;
    else if (bottom > wrap.scrollTop + wrap.clientHeight) wrap.scrollTop = bottom - wrap.clientHeight;
  }

  // ---------------------------------------------------------------------------
  // Detail panel
  // ---------------------------------------------------------------------------
  function parseExplanation(text) {
    const pattern = new RegExp(`\\*\\*\\s*(${EXPLANATION_FIELDS.join("|")})\\s*:?\\s*\\*\\*\\s*:?`, "gi");
    const matches = [...text.matchAll(pattern)];
    if (!matches.length) return null;
    return matches.map((m, i) => {
      const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
      const label = EXPLANATION_FIELDS.find((f) => f.toLowerCase() === m[1].toLowerCase()) || m[1];
      return { label, body: text.slice(m.index + m[0].length, end).replace(/\*\*/g, "").trim() };
    });
  }

  function confidenceNode(body) {
    const m = body.match(/^(high|medium|low)\b[\s.,:;—-]*/i);
    if (!m) return el("p", { class: "exp-text", text: body });
    const level = CONFIDENCE_LEVELS[m[1].toLowerCase()];
    const meter = el("span", { class: "confidence-meter", "aria-hidden": "true" },
      [1, 2, 3].map((i) => el("i", { class: i <= level ? "on" : "" })));
    const rest = body.slice(m[0].length).trim();
    return el("div", { class: `conf-${level}` },
      el("span", { class: "confidence" }, meter, m[1][0].toUpperCase() + m[1].slice(1).toLowerCase() + " confidence"),
      rest ? el("p", { class: "conf-rest", text: rest }) : null,
    );
  }

  const FIELD_STYLE = {
    "Likely Cause": { cls: "exp-cause", icon: "cause" },
    "Supporting Evidence": { cls: "exp-evidence", icon: "evidence" },
    "Confidence": { cls: "exp-confidence", icon: "confidence" },
    "Caveats": { cls: "exp-caveats", icon: "caveats" },
  };

  function explanationBody(text) {
    const sections = parseExplanation(text);
    if (!sections) return el("p", { class: "plain", text: text.replace(/\*\*/g, "") });
    return el("div", { class: "exp-blocks" }, sections.map((s) => {
      const style = FIELD_STYLE[s.label] || { cls: "", icon: "doc" };
      return el("div", { class: `exp-block ${style.cls}` },
        el("div", { class: "exp-label" }, icon(style.icon, 15), s.label),
        s.label === "Confidence" ? confidenceNode(s.body) : el("p", { class: "exp-text", text: s.body }),
      );
    }));
  }

  function sourcesSection(exp) {
    // The retriever can return several chunks of one document; list each document once.
    const seen = new Set();
    const items = (exp.context || []).filter((c) => {
      if (!c || !(c.title || c.url)) return false;
      const key = `${c.url || ""}|${c.title || ""}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const loneUrls = items.some((c) => c.url) ? [] : (exp.urls || []).map(safeUrl).filter(Boolean);
    const notes = (exp.notes || []).map((n) => el("li", { text: n }));
    const notesList = notes.length ? el("ul", { class: "retrieval-notes", "aria-label": "How the context was retrieved" }, notes) : null;
    if (!items.length && !loneUrls.length) {
      return notesList ? el("section", { class: "detail-section" },
        el("h3", {}, icon("link", 14), "Retrieved sources (0)"), notesList) : null;
    }

    const sourceRow = ({ source, date, title, href }) => {
      const label = SOURCE_LABELS[source] || source || "Source";
      const tag = source === "business_standard" ? "BS" : source === "wikipedia" ? "Wiki" : source === "sebi" ? "SEBI" : "Doc";
      const when = date && /^\d{4}-\d{2}-\d{2}$/.test(date) ? shortDate(date) : date;
      const body = [
        el("span", { class: `source-tag src-${source || "other"}`, title: label, text: tag }),
        el("span", { class: "source-body" },
          el("span", { class: "source-title", text: title || href }),
          el("span", { class: "source-meta", text: [label, when].filter(Boolean).join(" · ") })),
        href ? icon("external", 14) : null,
      ];
      return el("li", {}, href
        ? el("a", { class: "source", href, target: "_blank", rel: "noopener noreferrer" }, body)
        : el("div", { class: "source" }, body));
    };

    const list = el("ul", { class: "sources" },
      items.map((c) => sourceRow({ source: c.source, date: c.date, title: c.title, href: safeUrl(c.url) })),
      loneUrls.map((href) => sourceRow({ href })),
    );
    return el("section", { class: "detail-section" },
      el("h3", {}, icon("link", 14), `Retrieved sources (${items.length + loneUrls.length})`), list, notesList);
  }

  function pips(n, total) {
    return el("span", { class: "agreement", "aria-hidden": "true" },
      Array.from({ length: total }, (_, i) => el("i", { class: i < n ? "on" : "" })));
  }

  function metric(label, value, { wide = false, sub = null, extra = null, cls = null } = {}) {
    return el("div", { class: wide ? "wide" : null },
      el("dt", { text: label }),
      el("dd", { class: cls }, value, extra, sub ? el("span", { class: "sub", text: sub }) : null),
    );
  }

  function neighbours() {
    const list = visible();
    const sel = state.selected;
    const prev = [...list].reverse().find((a) => a.date < sel) || null;
    const next = list.find((a) => a.date > sel) || null;
    const pos = list.findIndex((a) => a.date === sel);
    return { prev, next, pos, total: list.length };
  }

  function renderDetail() {
    const panel = $("detail");
    const a = state.selected && state.byDate.get(state.selected);
    if (!a) {
      panel.replaceChildren(el("p", { class: "detail-empty", text: "Select a flagged day on the chart or in the table to see its explanation." }));
      return;
    }

    const n = state.data.ensemble_size;
    const { prev, next, pos, total } = neighbours();

    const head = el("div", { class: `detail-head type-${a.anomaly_type}` },
      el("div", { class: "detail-nav" },
        el("span", { text: pos >= 0 ? `Flagged day ${pos + 1} of ${total} in view` : "Outside the current view" }),
        el("span", { class: "nav-buttons" },
          el("button", { type: "button", disabled: !prev, title: "Previous flagged day (←)", onclick: () => prev && selectDay(prev.date, { scrollRow: true }) }, "‹ Previous"),
          el("button", { type: "button", disabled: !next, title: "Next flagged day (→)", onclick: () => next && selectDay(next.date, { scrollRow: true }) }, "Next ›"),
        ),
      ),
      el("div", { class: "detail-date", text: longDate(a.date) }),
      el("div", { class: "detail-weekday", text: fmtWeekday.format(a.t) }),
      el("div", { class: "detail-badges" },
        typePill(a.anomaly_type, TYPES[a.anomaly_type].label + " anomaly", true),
        a.near_rebalancing ? el("span", { class: "badge", text: "Near index rebalancing" }) : null,
      ),
      a.event ? el("div", { class: "event-note" },
        icon("info", 15),
        el("div", {},
          el("strong", { text: a.event.event }),
          a.event.notes ? el("span", { text: a.event.notes }) : null),
      ) : null,
    );

    const episodeDays = a.regime_episode_start
      ? state.anomalies.filter((x) => x.regime_episode_start === a.regime_episode_start).length
      : 0;

    const market = el("section", { class: "detail-section" },
      el("h3", {}, icon("chart", 14), "Market data"),
      el("dl", { class: "metrics" },
        metric("Close", fmtPrice.format(a.close)),
        metric("Daily return", pct(simpleReturn(a.log_return)), { cls: signClass(a.log_return) }),
        metric("Return z-score", signed(a.return_zscore), { sub: "vs. 20-day mean" }),
        metric("60-day return", pct(a.return_60d), { cls: signClass(a.return_60d) }),
        metric("Shock models", `${a.shock_votes} of ${n}`, { extra: pips(a.shock_votes, n) }),
        metric("Regime models", `${a.regime_votes} of ${n}`, { extra: pips(a.regime_votes, n) }),
        a.regime_episode_start ? metric("Regime episode",
          a.regime_episode_start === a.regime_episode_end
            ? shortDate(a.regime_episode_start)
            : `${shortDate(a.regime_episode_start)} – ${shortDate(a.regime_episode_end)}`,
          { wide: true, sub: `${episodeDays} flagged ${episodeDays === 1 ? "day" : "days"} in this episode` }) : null,
        a.near_rebalancing && a.rebalancing_notes ? metric("Rebalancing", "", { wide: true, sub: a.rebalancing_notes }) : null,
      ),
    );

    const exp = hasExplanation(a) ? state.data.explanations[a.explanation_date] : null;
    panel.replaceChildren(...[head, market, explanationSection(a, exp), exp && sourcesSection(exp)].filter(Boolean));
    panel.scrollTop = 0;
  }

  function serviceProblem() {
    if (!state.canExplain) return "static";
    if (state.service && !state.service.ready) return "config";
    return null;
  }

  function explanationSection(a, exp) {
    const busy = state.generating.has(a.date);
    const error = state.errors.get(a.date);
    const problem = serviceProblem();
    const section = el("section", { class: "detail-section explanation" });
    const heading = el("div", { class: "section-head" }, el("h3", {}, icon("doc", 14), "Explanation"));
    section.append(heading);

    if (busy) {
      section.append(el("div", { class: "generating", role: "status" },
        el("span", { class: "spinner", "aria-hidden": "true" }),
        el("div", {},
          el("strong", { text: "Generating explanation…" }),
          el("span", { text: "Retrieving SEBI filings, news and background for this date, then asking the language model. This usually takes 10–30 seconds." }),
        ),
      ));
      return section;
    }

    if (exp) {
      const own = a.explanation_date === a.date;
      if (!problem) {
        heading.append(el("button", {
          type: "button", class: "btn btn-ghost",
          title: own ? "Generate a fresh explanation for this day" : "Generate an explanation for this day on its own",
          onclick: () => generateExplanation(a.date, { force: own }),
        }, icon(own ? "refresh" : "sparkles", 13), own ? "Regenerate" : "Explain this day"));
      }
      if (!own) {
        section.append(el("p", { class: "callout shared-note",
          text: `This regime day shares the explanation generated for ${longDate(a.explanation_date)}, another day of the same episode.` }));
      }
      if (error) section.append(errorCallout(error));
      section.append(explanationBody(exp.explanation || ""));
      return section;
    }

    const cta = el("div", { class: "explain-cta" },
      el("p", { class: "explain-intro",
        text: "No explanation has been generated for this day yet. The explainer retrieves SEBI filings and Business Standard coverage dated near this day, plus Wikipedia background, and asks the language model to explain the anomaly using only that material." }));
    if (error) cta.append(errorCallout(error));

    cta.append(el("button", {
      type: "button", class: "btn btn-primary btn-block",
      disabled: Boolean(problem),
      onclick: () => generateExplanation(a.date),
    }, icon(error ? "refresh" : "sparkles", 16), error ? "Try again" : "See explanation"));

    if (problem === "static") {
      cta.append(el("p", { class: "hint" },
        "Generating explanations needs the dashboard server. Start it with ",
        el("code", { text: "python backend/src/server.py" }), " and open the address it prints."));
    } else if (problem === "config") {
      cta.append(el("p", { class: "hint", text: state.service.problems.join(" ") }));
    }
    section.append(cta);
    return section;
  }

  function errorCallout(message) {
    return el("p", { class: "callout callout-error", role: "alert" },
      el("strong", { text: "Could not generate the explanation. " }), message);
  }

  /** Which explanation each anomaly shows -- mirrors export_frontend_data.explanation_date. */
  function assignExplanationDates() {
    const explanations = state.data.explanations;
    const episodeExplained = new Map();
    for (const a of state.anomalies) {
      if (a.anomaly_type === "regime" && explanations[a.date]) episodeExplained.set(a.regime_episode_start, a.date);
    }
    for (const a of state.anomalies) {
      a.explanation_date = explanations[a.date] ? a.date
        : a.anomaly_type === "regime" ? episodeExplained.get(a.regime_episode_start) || null
        : null;
    }
  }

  async function generateExplanation(date, { force = false } = {}) {
    if (state.generating.has(date)) return;
    state.generating.add(date);
    state.errors.delete(date);
    refreshExplanationViews(date);

    try {
      const res = await fetch("api/explain?sector=" + encodeURIComponent(state.data.sector), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date, force }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.error || `The server returned HTTP ${res.status}.`);
      state.data.explanations[payload.date] = payload.explanation;
      assignExplanationDates();
    } catch (err) {
      state.errors.set(date, err.message || String(err));
    } finally {
      state.generating.delete(date);
      refreshExplanationViews(date);
    }
  }

  function refreshExplanationViews(date) {
    renderStats();
    renderTable();
    if (state.explainedOnly) updateMarkers();
    // Re-render the panel only if it shows this day or shares its regime episode,
    // so a reader on another day doesn't lose their scroll position.
    const sel = state.byDate.get(state.selected);
    const gen = state.byDate.get(date);
    if (sel && (sel.date === date ||
        (sel.regime_episode_start && sel.regime_episode_start === gen?.regime_episode_start))) {
      renderDetail();
    }
  }

  function renderServiceStatus() {
    let tone, label, title;
    if (!state.canExplain) {
      tone = "off"; label = "Static mode · explanations read-only";
      title = "Opened without server.py, so new explanations can't be generated.";
    } else if (!state.service) {
      tone = "off"; label = "Explanation service unreachable"; title = "";
    } else if (state.service.ready) {
      tone = "ok"; label = `Explanation service ready · ${state.service.model}`; title = "";
    } else {
      tone = "warn"; label = "Explanation service not configured"; title = state.service.problems.join(" ");
    }
    $("service-status").replaceChildren(
      el("span", { class: `status-pill status-${tone}`, title: title || null },
        el("span", { class: "status-dot", "aria-hidden": "true" }), label));
  }

  function selectDay(date, { scrollRow = true } = {}) {
    state.selected = date;
    renderDetail();
    syncTableSelection(scrollRow);
    updateSelectedMarker();
  }

  // ---------------------------------------------------------------------------
  // Chart
  // ---------------------------------------------------------------------------
  function markerData(type) {
    return state.anomalies
      .filter((a) => a.anomaly_type === type && passesFilters(a))
      .map((a) => [a.t, a.close, a.date]);
  }

  function selectedData() {
    const a = state.selected && state.byDate.get(state.selected);
    return a ? [[a.t, a.close, a.date]] : [];
  }

  function tooltip(params) {
    const c = state.colors;
    const p = params.find((x) => x.seriesId === "price") || params[0];
    if (!p) return "";
    const date = isoFromTs(p.value[0]);
    const close = p.value[1];
    const lr = state.returnByDate.get(date);
    const a = state.byDate.get(date);
    const visibleAnomaly = a && passesFilters(a);

    let html =
      `<div style="font-size:11px;color:${c.muted};margin-bottom:2px">${shortDate(date)}</div>` +
      `<div><strong style="font-size:14px;color:${c.ink}">${fmtPrice.format(close)}</strong>` +
      ` <span style="color:${c.ink2}">close</span></div>`;
    if (lr !== undefined) {
      html += `<div><strong style="color:${c.ink}">${pct(simpleReturn(lr))}</strong> <span style="color:${c.ink2}">daily return</span></div>`;
    }
    if (visibleAnomaly) {
      const color = c[a.anomaly_type];
      html += `<div style="margin-top:6px;display:flex;align-items:center;gap:6px;color:${c.ink}">` +
        `<span style="display:inline-block;width:12px;height:2px;background:${color}"></span>` +
        `${TYPES[a.anomaly_type].label} anomaly</div>` +
        `<div style="font-size:11px;color:${c.muted}">Click for details</div>`;
    }
    return html;
  }

  function buildOption() {
    const c = (state.colors = readColors());
    const episodes = state.data.episodes.map((e) => [
      { xAxis: ts(e.start) - 0.5 * 86400000 },
      { xAxis: ts(e.end) + 0.5 * 86400000 },
    ]);

    return {
      animation: false,
      textStyle: { fontFamily: cssVar("--font") },
      grid: { left: 8, right: 16, top: 12, bottom: 64, containLabel: true },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: c.muted, width: 1 } },
        backgroundColor: c.surface,
        borderColor: c.border,
        borderWidth: 1,
        padding: [8, 10],
        textStyle: { color: c.ink, fontSize: 12 },
        extraCssText: "box-shadow: 0 4px 16px rgba(0,0,0,0.12); border-radius: 6px;",
        formatter: tooltip,
      },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: c.axis } },
        axisTick: { show: false },
        axisLabel: { color: c.muted, fontSize: 11, hideOverlap: true },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: c.muted, fontSize: 11, formatter: (v) => fmtInt.format(v) },
        splitLine: { lineStyle: { color: c.grid, width: 1 } },
      },
      dataZoom: [
        {
          type: "inside", xAxisIndex: 0, filterMode: "filter",
          zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false, moveOnMouseMove: true,
          startValue: state.range.from, endValue: state.range.to,
        },
        {
          type: "slider", xAxisIndex: 0, filterMode: "filter", height: 26, bottom: 14,
          startValue: state.range.from, endValue: state.range.to,
          borderColor: c.border, backgroundColor: "transparent", fillerColor: c.band,
          handleStyle: { color: c.surface, borderColor: c.muted },
          moveHandleStyle: { color: c.axis },
          emphasis: { handleStyle: { borderColor: c.accent }, moveHandleStyle: { color: c.accent } },
          dataBackground: { lineStyle: { color: c.axis, width: 1 }, areaStyle: { color: c.grid, opacity: 0.6 } },
          selectedDataBackground: { lineStyle: { color: c.price, width: 1 }, areaStyle: { color: c.band, opacity: 1 } },
          textStyle: { color: c.muted, fontSize: 11 },
          labelFormatter: (v) => shortDate(isoFromTs(v)),
          brushSelect: false,
        },
      ],
      series: [
        {
          id: "price", name: "Close", type: "line", data: state.prices,
          showSymbol: false, symbol: "none", sampling: "lttb",
          lineStyle: { width: 1.6, color: c.price }, itemStyle: { color: c.price },
          areaStyle: {
            origin: "start",
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: withAlpha(c.price, 0.16) },
              { offset: 1, color: withAlpha(c.price, 0) },
            ]),
          },
          emphasis: { disabled: true }, z: 2,
          markArea: { silent: true, itemStyle: { color: c.band }, data: episodes },
        },
        ...TYPE_ORDER.map((t) => ({
          id: t, name: TYPES[t].label, type: "scatter", data: markerData(t),
          symbol: TYPES[t].symbol, symbolSize: t === "regime" ? 9 : 10,
          itemStyle: { color: c[t], borderColor: c.surface, borderWidth: 1.5,
                       shadowColor: "rgba(15, 31, 61, 0.18)", shadowBlur: 3 },
          emphasis: { scale: 1.4 }, cursor: "pointer", z: 3,
        })),
        {
          id: "selected", type: "scatter", data: selectedData(), silent: true,
          symbol: "circle", symbolSize: 20, z: 4,
          itemStyle: { color: "transparent", borderColor: c.ink, borderWidth: 2 },
          tooltip: { show: false },
        },
      ],
    };
  }

  function nearestAnomaly(x, y) {
    const chart = state.chart;
    if (!chart.containPixel({ gridIndex: 0 }, [x, y])) return null;
    let best = null;
    let bestScore = Infinity;
    for (const a of state.anomalies) {
      if (!passesFilters(a) || !inRange(a.t)) continue;
      const [px, py] = chart.convertToPixel({ seriesId: "price" }, [a.t, a.close]);
      const dx = Math.abs(px - x);
      if (dx > CLICK_TOLERANCE_PX) continue;
      // Horizontal distance decides (it matches the crosshair); vertical only breaks ties.
      const score = dx + Math.abs(py - y) * 0.01;
      if (score < bestScore) { bestScore = score; best = a; }
    }
    return best;
  }

  function onDataZoom() {
    if (state.applyingRange) return;
    const dz = state.chart.getOption().dataZoom[0];
    let { startValue: from, endValue: to } = dz;
    if (typeof from !== "number" || typeof to !== "number") {
      const span = state.lastT - state.firstT;
      from = state.firstT + (span * dz.start) / 100;
      to = state.firstT + (span * dz.end) / 100;
    }
    state.range = { from, to };
    state.rangeKey = null;
    setRangeButtons(null);
    renderStats();
    renderTable();
    renderDetail();
  }

  function initChart() {
    const chart = (state.chart = echarts.init($("chart"), null, { renderer: "canvas" }));
    chart.setOption(buildOption());

    chart.on("datazoom", onDataZoom);
    const zr = chart.getZr();
    zr.on("click", (e) => {
      const a = nearestAnomaly(e.offsetX, e.offsetY);
      if (a) selectDay(a.date, { scrollRow: true });
    });
    zr.on("mousemove", (e) => {
      zr.setCursorStyle(nearestAnomaly(e.offsetX, e.offsetY) ? "pointer" : "default");
    });

    new ResizeObserver(() => chart.resize()).observe($("chart"));
  }

  function updateMarkers() {
    state.chart.setOption({
      series: [...TYPE_ORDER.map((t) => ({ id: t, data: markerData(t) })), { id: "selected", data: selectedData() }],
    });
  }

  function updateSelectedMarker() {
    state.chart.setOption({ series: [{ id: "selected", data: selectedData() }] });
  }

  // ---------------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------------
  function refresh() {
    updateMarkers();
    renderStats();
    renderTable();
    renderDetail();
  }

  function renderFooter() {
    const d = state.data;
    $("footer").textContent =
      `Data exported ${longDate(d.generated_at.slice(0, 10))}. Flagging threshold: top ${Math.round(d.contamination * 100)}% of ensemble scores per feature set. ` +
      "Explanations are machine-generated from retrieved documents and are not investment advice.";
  }

  function bindControls() {
    for (const b of $("range-buttons").querySelectorAll("button")) {
      b.addEventListener("click", () => applyRange(b.dataset.range));
    }
    $("explained-only").addEventListener("change", (e) => {
      state.explainedOnly = e.target.checked;
      refresh();
    });
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input, textarea, select") || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" || !state.selected) return;
      const { prev, next } = neighbours();
      const target = e.key === "ArrowLeft" ? prev : next;
      if (target) { e.preventDefault(); selectDay(target.date, { scrollRow: true }); }
    });
  }

  function prepare(data) {
    state.data = data;
    const { date, close, log_return } = data.prices;
    state.prices = date.map((d, i) => [ts(d), close[i]]);
    date.forEach((d, i) => state.returnByDate.set(d, log_return[i]));
    state.firstT = state.prices[0][0];
    state.lastT = state.prices[state.prices.length - 1][0];
    state.range = rangeFor("ALL");

    state.anomalies = data.anomalies
      .map((a) => ({ ...a, t: ts(a.date) }))
      .sort((x, y) => x.t - y.t);
    for (const a of state.anomalies) state.byDate.set(a.date, a);

    const explained = state.anomalies.filter(hasExplanation);
    const initial = explained[explained.length - 1] || state.anomalies[state.anomalies.length - 1];
    state.selected = initial ? initial.date : null;
  }

  function showError(message) {
    const box = el("div", { class: "load-error" },
      el("strong", { text: "The dashboard could not load its data." }),
      el("p", { text: message }),
      el("p", {}, "Start the dashboard server with ", el("code", { text: "python backend/src/server.py" }),
        " and open the address it prints. It needs ", el("code", { text: "db/processed/NIFTY50_final_anomalies.csv" }),
        ", produced by ", el("code", { text: "train_multi_model.py" }), " and ", el("code", { text: "build_final_anomalies.py" }), "."),
    );
    document.querySelector(".layout").replaceWith(box);
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${url} returned HTTP ${res.status}.`);
    return res.json();
  }

  /** Live data from server.py; the static export only when no server API exists
   *  (network error, or a plain static host's non-JSON 404). Any error the API
   *  itself reports is surfaced instead of hidden behind stale static data. */
  async function loadData() {
    const res = await fetch(API_DATA_URL, { cache: "no-store" }).catch(() => null);
    const isJson = res && (res.headers.get("Content-Type") || "").includes("application/json");
    if (res && isJson) {
      const payload = await res.json().catch((err) => { throw new Error(`The server's data could not be read: ${err.message}`); });
      if (!res.ok) throw new Error(payload.error || `${API_DATA_URL} returned HTTP ${res.status}.`);
      state.canExplain = Boolean(payload.can_explain);
      return payload;
    }
    return fetchJson(STATIC_DATA_URL);
  }

  async function main() {
    if (typeof echarts === "undefined") {
      showError("The charting library (ECharts) failed to load. Check your internet connection.");
      return;
    }
    let data;
    try {
      data = await loadData();
    } catch (err) {
      showError(err.message || String(err));
      return;
    }
    if (state.canExplain) {
      state.service = await fetchJson("api/status").catch(() => null);
    }

    prepare(data);
    renderServiceStatus();
    renderProvenance();
    renderLegend();
    renderTypeFilters();
    renderFooter();
    bindControls();
    initChart();
    renderStats();
    renderTable();
    renderDetail();
    syncTableSelection(true);
    document.body.classList.remove("is-loading");
  }

  main();
})();
