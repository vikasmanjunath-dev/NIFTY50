// All text from the server is added with textContent (never innerHTML): filing text is untrusted.
const view = document.getElementById("view");
const state = { horizon: "5", query: "", sort: "prob", only: "all" };

// The watchlist is a per-browser convenience; storage can be unavailable (private windows).
const watch = new Set((() => { try { return JSON.parse(localStorage.getItem("watch")) || []; } catch { return []; } })());
function toggleWatch(id) {
  watch.has(id) ? watch.delete(id) : watch.add(id);
  try { localStorage.setItem("watch", JSON.stringify([...watch])); } catch { /* keep it for this visit only */ }
}
function star(id, after) {
  const b = h("button", { class: "star", type: "button" });
  const paint = () => { b.textContent = watch.has(id) ? "★" : "☆"; b.setAttribute("aria-pressed", watch.has(id));
    b.setAttribute("aria-label", watch.has(id) ? "Remove from watchlist" : "Add to watchlist"); };
  b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); toggleWatch(id); paint(); after?.(); });
  paint();
  return b;
}
const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);

function h(tag, attrs = {}, ...children) {
  const svg = ["svg", "path", "line", "text", "polyline"].includes(tag);
  const el = svg ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k.startsWith("aria-") && v != null) el.setAttribute(k, v); // "true"/"false" as text
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}
const card = (...c) => h("div", { class: "card" }, ...c);
const stat = (label, value, note) => h("div", { class: "card stat" }, h("small", {}, label), h("b", {}, value), h("small", {}, note || ""));
const header = (title, sub) => [h("h1", {}, title), h("p", { class: "sub" }, sub)];
const notices = { HALTED: "Forecasts are withheld: the kill switch is on.", STALE: "Forecasts are withheld: the data is out of date. Run scripts/daily_refresh.py.",
  NO_DATA: "No forecasts yet. Run scripts/forecast_volatility.py." };
const notice = (status) => notices[status] && h("div", { class: "card notice" }, notices[status]);

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(res.status === 404 ? "Not found." : `The server returned ${res.status}.`);
  return res.json();
}

function chart(series, labels, format) {
  const W = Math.max(300, Math.min(800, innerWidth - 68)), H = 240, L = 52, R = 8, T = 10, B = 22;
  const all = series.flatMap((s) => s.values);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad;
  const x = (i) => L + (i / Math.max(labels.length - 1, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const readout = h("div", { class: "readout" });
  const cursor = h("line", { class: "cursor", y1: T, y2: H - B, x1: -9, x2: -9 });
  const show = (i) => {
    cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i));
    readout.textContent = `${labels[i]} · ${series.map((s) => `${s.name ? s.name + " " : ""}${format(s.values[i])}`).join(" · ")}`;
  };
  const svg = h("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": series.map((s) => s.name || "Closing price").join(", ") },
    [0, 0.5, 1].map((f) => { const v = lo + f * (hi - lo); return [h("line", { class: "grid-line", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      h("text", { x: 0, y: y(v) + 4 }, format(v))]; }),
    series.map((s, k) => h("polyline", { class: k ? "line second" : "line", points: s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ") })),
    h("text", { x: L, y: H - 4 }, labels[0]), h("text", { x: W - R, y: H - 4, "text-anchor": "end" }, labels.at(-1)), cursor);
  svg.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    const i = Math.round((((e.clientX - box.left) / box.width) * W - L) / (W - L - R) * (labels.length - 1));
    show(Math.max(0, Math.min(labels.length - 1, i)));
  });
  show(labels.length - 1);
  return h("div", {}, readout, svg);
}

const legend = (a, b) => h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--accent)" }), a),
  h("span", {}, h("i", { style: "background:var(--orange)" }), b));

const meter = (p) => h("span", { class: "meter" }, h("i", {}, h("b", { class: p >= 0.6 ? "high" : "", style: `width:${p * 100}%` })), pct(p, 0));
const range = (f) => (f && f.move_low_90 != null ? `${pct(f.move_low_90)} to +${pct(f.move_high_90)}` : "–");

// Spreadsheet apps run cells that start with =, +, - or @ as formulas; prefix text cells so they stay text.
const csvCell = (v) => { const s = String(v ?? ""); return `"${(typeof v === "string" && /^[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""')}"`; };
function download(name, rows) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.map((r) => r.map(csvCell).join(",")).join("\n")], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function today() {
  const data = await get("/api/overview");
  const list = h("div", { class: "list" });
  const count = h("p", { class: "dim" });
  const draw = () => {
    const key = `h${state.horizon}`;
    const q = state.query.trim().toLowerCase();
    const rows = data.rows.filter((r) => r[key] && `${r.symbol} ${r.sector || ""}`.toLowerCase().includes(q) && (state.only === "all" || watch.has(r.id)));
    rows.sort({ prob: (a, b) => b[key].prob_above_usual - a[key].prob_above_usual,
      vol: (a, b) => b[key].forecast_vol_annualised - a[key].forecast_vol_annualised, name: (a, b) => a.symbol.localeCompare(b.symbol) }[state.sort]);
    list.replaceChildren(h("div", { class: "row head" }, h("span", {}, "Stock"), h("span", {}, "Chance of above-usual volatility"),
      h("span", {}, "Volatility (yearly)"), h("span", {}, "9-in-10 price move")),
      ...rows.map((r) => h("a", { class: "row", href: `#/stock/${encodeURIComponent(r.id)}` },
        h("span", { class: "name" }, star(r.id, draw), h("span", {}, h("strong", {}, r.symbol), h("div", { class: "dim" }, r.sector || ""))), meter(r[key].prob_above_usual),
        h("span", {}, pct(r[key].forecast_vol_annualised)), h("span", { class: "dim" }, range(r[key])))),
      ...(rows.length ? [] : [h("div", { class: "empty" }, state.only === "watch" ? "Tap ☆ beside a stock to add it to your watchlist." : "No matching stocks.")]));
    count.textContent = `${rows.length} stocks · next ${state.horizon} trading days`;
    for (const b of seg.children) b.setAttribute("aria-pressed", b.dataset.h === state.horizon);
    const c = data.changes[state.horizon];
    moved.replaceChildren(...(c && c.rows.length ? [h("h2", {}, `Biggest changes since ${c.since}`), h("div", { class: "tiles" }, c.rows.map((m) =>
      h("a", { class: "card", href: `#/stock/${encodeURIComponent(m.id)}` }, h("strong", {}, m.symbol),
        h("div", { class: "dim" }, `${m.now > m.before ? "▲" : "▼"} ${pct(m.before, 0)} → ${pct(m.now, 0)}`))))] : []));
    for (const b of which.children) b.setAttribute("aria-pressed", b.dataset.w === state.only);
    const groups = new Map();
    for (const r of data.rows) if (r[key]) groups.set(r.sector, [...(groups.get(r.sector) || []), r[key].prob_above_usual]);
    sectors.replaceChildren(...[...groups].map(([name, v]) => [name, v.reduce((a, b) => a + b, 0) / v.length, v.length]).sort((a, b) => b[1] - a[1])
      .map(([name, mean, n]) => h("button", { class: "tile", type: "button", style: `--heat:${Math.max(0, Math.min(1, (mean - 0.35) / 0.35))}`,
        onclick: () => { state.query = state.query === name ? "" : name; search.value = state.query; draw(); } },
        h("strong", {}, name || "Other"), h("span", {}, `${pct(mean, 0)} · ${n}`))));
  };
  const sectors = h("div", { class: "tiles" }), moved = h("div");
  const which = h("div", { class: "segmented", role: "group", "aria-label": "Show" }, [["all", "All"], ["watch", "Watchlist"]].map(([v, t]) =>
    h("button", { "data-w": v, type: "button", onclick: () => { state.only = v; draw(); } }, t)));
  const seg = h("div", { class: "segmented", role: "group", "aria-label": "Horizon" }, ["5", "20"].map((n) =>
    h("button", { "data-h": n, onclick: () => { state.horizon = n; draw(); } }, `${n} days`)));
  const sort = h("select", { "aria-label": "Sort by", onchange: (e) => { state.sort = e.target.value; draw(); } },
    [["prob", "Highest chance"], ["vol", "Most volatile"], ["name", "Name"]].map(([v, t]) => h("option", { value: v, selected: state.sort === v }, t)));
  const search = h("input", { type: "search", placeholder: "Search stock or sector", value: state.query, "aria-label": "Search",
    oninput: (e) => { state.query = e.target.value; draw(); } });
  const high = data.rows.filter((r) => r.h5 && r.h5.prob_above_usual >= 0.6).length;
  const parts = [...header("Today", data.data_date ? `Market data to ${data.data_date}` : "Waiting for data"),
    h("p", {}, h("a", { href: "#/guide" }, "What do these numbers mean?")), notice(data.status)];
  // Watched stocks worth a look: a high chance of a busy spell, or a meeting already announced.
  const watchlist = () => {
    const hits = data.rows.filter((r) => watch.has(r.id)).map((r) => ({ r, chance: r.h5?.prob_above_usual, meeting: data.upcoming.find((u) => u.id === r.id) }))
      .filter((x) => x.chance >= 0.7 || x.meeting);
    return hits.length ? [h("h2", {}, "On your watchlist"), h("div", { class: "tiles" }, hits.map(({ r, chance, meeting }) =>
      h("a", { class: "card", href: `#/stock/${encodeURIComponent(r.id)}` }, h("strong", {}, r.symbol),
        h("div", { class: "dim" }, [chance >= 0.7 ? `${pct(chance, 0)} chance of a busy week` : "", meeting ? `meeting ${meeting.date}` : ""].filter(Boolean).join(" · ")))))] : [];
  };
  if (data.rows.length) {
    parts.push(h("div", { class: "grid" }, stat("Stocks covered", data.rows.length, "Current index members"),
      stat("Likely busier than usual", high, "5-day chance of 60% or more"), stat("Direction signal", "None", "Up or down is not predictable here")),
      data.upcoming.length ? [h("h2", {}, "Announced meetings, next 30 days"), h("div", { class: "tiles" }, data.upcoming.map((u) =>
        h("a", { class: "card", href: `#/stock/${encodeURIComponent(u.id)}` }, h("strong", {}, u.symbol),
          h("div", { class: "dim" }, `${u.date}${u.results ? " · Results" : ""}`))))] : [],
      watchlist(), moved, h("h2", {}, "By sector"), sectors, h("div", { class: "controls" }, seg, which, search, sort, h("button", { type: "button", class: "primary", onclick: () => download(`volatility-${data.data_date}.csv`,
        [["symbol", "sector", "horizon_days", "prob_above_usual", "forecast_vol", "usual_vol", "move_low_90", "move_high_90", "data_date"],
          ...data.rows.flatMap((r) => ["5", "20"].filter((n) => r[`h${n}`]).map((n) => { const f = r[`h${n}`]; return [r.symbol, r.sector, Number(n),
            f.prob_above_usual, f.forecast_vol_annualised, f.usual_vol_annualised, f.move_low_90, f.move_high_90, data.data_date]; }))]) }, "Export CSV")), count, list);
    draw();
  }
  parts.push(h("footer", {}, data.disclaimer));
  return parts;
}

async function stock(id) {
  const d = await get(`/api/stock/${encodeURIComponent(id)}`);
  const f = Object.entries(d.forecasts).sort(([a], [b]) => a - b);
  return [h("a", { href: "#/today" }, "‹ Today"), h("div", { class: "title" }, h("h1", {}, d.symbol), star(d.id)), h("p", { class: "sub" }, d.sector || "", " · ", h("a", { href: `#/compare/${encodeURIComponent(d.id)}` }, "Compare with another")), notice(d.status),
    h("div", { class: "grid" }, f.map(([n, v]) => stat(`Next ${n} days`, pct(v.prob_above_usual, 0),
      `chance of above-usual volatility · ${pct(v.forecast_vol_annualised)} forecast vs ${pct(v.usual_vol_annualised)} usual · 9-in-10 move ${range(v)}`))),
    h("h2", {}, "Price, past year"), card(chart([{ values: d.prices.map((p) => p[1]) }], d.prices.map((p) => p[0]), (v) => `₹${Math.round(v).toLocaleString("en-IN")}`)),
    Object.entries(d.record).map(([n, r]) => [h("h2", {}, `${n}-day volatility: forecast against what happened`),
      card(legend("Forecast", "Actual"), chart([{ name: "forecast", values: r.forecast }, { name: "actual", values: r.actual }], r.dates, (v) => pct(v, 0)),
        h("p", { class: "dim" }, "Past test forecasts, each made without seeing its outcome. Not the live record."))]),
    h("h2", {}, "Saved forecasts (live record)"), h("div", { class: "list" }, d.forward.length ? d.forward.map((x) => h("div", { class: "filing" },
      h("strong", {}, x.date), ` · ${x.horizon}-day · said ${pct(x.prob, 0)} · `, x.above_usual == null
        ? h("span", { class: "badge" }, `pending, ${x.days_left} trading days left`)
        : h("span", { class: `badge ${x.above_usual === x.prob > 0.5 ? "OK" : "STALE"}` }, x.above_usual ? "was above usual" : "stayed usual or calmer")))
      : h("div", { class: "empty" }, "None saved yet.")),
    h("h2", {}, "Announced board meetings"), h("div", { class: "list" }, d.meetings.length ? d.meetings.map((m) =>
      h("div", { class: "filing" }, h("strong", {}, m.date), " ", m.results && h("span", { class: "badge" }, "Results"))) : h("div", { class: "empty" }, "None on record.")),
    h("h2", {}, "Latest filings"), filingList(d.filings), h("footer", {}, d.disclaimer)];
}

const filingList = (items) => h("div", { class: "list" }, items.length ? items.map((x) => h("article", { class: "filing" },
  h("strong", {}, x.symbol), " ", h("span", { class: "badge" }, x.category), " ", h("span", { class: "dim" }, x.date),
  h("p", {}, x.text), x.url && h("a", { href: x.url, target: "_blank", rel: "noopener noreferrer" }, "Open document"))) : h("div", { class: "empty" }, "No filings found."));

// Holdings stay in this browser; they are sent only to this local server to compute risk.
const holdings = new Map((() => { try { return JSON.parse(localStorage.getItem("holdings")) || []; } catch { return []; } })());
const saveHoldings = () => { try { localStorage.setItem("holdings", JSON.stringify([...holdings])); } catch { /* this visit only */ } };
const rupees = (v) => `₹${Math.round(v).toLocaleString("en-IN")}`;

async function portfolio() {
  const data = await get("/api/overview");
  const symbols = new Map(data.rows.map((r) => [r.id, r.symbol]));
  const rows = h("div", { class: "list" }), result = h("div");
  let latest = 0;
  const analyse = async () => {
    const mine = ++latest;
    const held = [...holdings].filter(([id]) => symbols.has(id));
    if (!held.length) return result.replaceChildren(card("Add a stock and the amount you hold to see its risk."));
    const d = await get(`/api/portfolio?horizon=${state.horizon}&holdings=${encodeURIComponent(held.map(([i, a]) => `${i}:${a}`).join(","))}`);
    if (mine !== latest) return;
    if (d.status !== "OK") return result.replaceChildren(notice(d.status) || card("No forecast available."));
    result.replaceChildren(h("div", { class: "grid" },
      stat("Expected volatility (yearly)", pct(d.volatility), `usual for this mix: ${pct(d.usual_volatility)}`),
      stat(`Typical ${d.horizon}-day swing`, `±${rupees(d.typical_move)}`, `on ${rupees(d.total)}`),
      stat("Rough 1-in-10 swing", `±${rupees(d.large_move)}`, d.large_move_held ? `held ${pct(d.large_move_held, 0)} of the time in a later-period check; extremes can be far larger` : "assumes a bell curve; real extremes can be larger"),
      stat("Diversification saves", pct(d.undiversified - d.volatility), "volatility, versus moving in lockstep")),
      h("h2", {}, "Where the risk comes from"), h("div", { class: "list" }, d.parts.map((p) => h("div", { class: "row" },
        h("strong", {}, p.symbol), meter(Math.max(0, Math.min(1, p.risk_share))), h("span", {}, `${pct(p.weight, 0)} of money`),
        h("span", { class: "dim" }, `${pct(p.volatility)} volatility`)))));
  };
  const draw = () => {
    rows.replaceChildren(...[...holdings].filter(([id]) => symbols.has(id)).map(([id, amount]) => h("div", { class: "filing holding" },
      h("strong", {}, symbols.get(id)), h("span", {}, rupees(amount)),
      h("button", { type: "button", class: "plain", onclick: () => { holdings.delete(id); saveHoldings(); draw(); } }, "Remove"))));
    analyse().catch((err) => result.replaceChildren(card(err.message)));
  };
  const pick = h("select", { "aria-label": "Stock" }, [...symbols].sort((a, b) => a[1].localeCompare(b[1])).map(([id, s]) => h("option", { value: id }, s)));
  const amount = h("input", { type: "number", min: "1", step: "any", placeholder: "Amount in ₹", "aria-label": "Amount in rupees", required: true });
  const form = h("form", { class: "controls", onsubmit: (e) => { e.preventDefault(); const v = Number(amount.value);
    if (v > 0) { holdings.set(pick.value, v); saveHoldings(); amount.value = ""; draw(); } } }, pick, amount, h("button", { class: "primary" }, "Add"));
  const seg = h("div", { class: "segmented", role: "group", "aria-label": "Horizon" }, ["5", "20"].map((n) =>
    h("button", { type: "button", "aria-pressed": state.horizon === n, onclick: (e) => { state.horizon = n;
      for (const b of seg.children) b.setAttribute("aria-pressed", b === e.target); analyse(); } }, `${n} days`)));
  draw();
  return [...header("Portfolio", "How much could your mix of stocks swing? Direction is not predicted."), notice(data.status), form, seg,
    h("h2", {}, "Holdings"), rows, result, h("footer", {}, `${data.disclaimer} Amounts are saved only in this browser.`)];
}

const GUIDE = [
  ["Volatility", "How much a price swings, whichever way. A high figure means bigger moves are likely, not that the price will fall."],
  ["Chance of above-usual volatility", "The chance that the next 5 or 20 trading days swing more than this stock normally does, judged against its own past year."],
  ["Volatility (yearly)", "The expected size of swings, scaled to one year so that different periods can be compared."],
  ["9-in-10 price move", "A range meant to contain the price change about 9 times in 10. In testing it held a little less often, and shocks can go far beyond it."],
  ["Up or down", "Not predicted. Every method tried did no better than a coin flip."],
  ["Accuracy score (AUC)", "If you pick one busy period and one calm period at random, how often the model ranked the busy one higher. 0.5 is guessing."],
  ["Past tests and the forward record", "Past tests reuse history that has been examined many times. The forward record only grades forecasts saved before the outcome, so trust it more."],
  ["Missing or withheld numbers", "Forecasts are hidden when the data is out of date or the kill switch is on."],
];

async function guide() {
  return [h("a", { href: "#/today" }, "‹ Today"), ...header("Guide", "What the numbers mean, and what they do not"),
    h("div", { class: "list" }, GUIDE.map(([term, text]) => h("div", { class: "filing" }, h("strong", {}, term), h("p", {}, text)))),
    h("footer", {}, "Research output, not investment advice.")];
}

async function compare(pair) {
  const data = await get("/api/overview");
  const ids = data.rows.map((r) => r.id);
  const [first, second] = (pair || "").split(",");
  const chosen = [ids.includes(first) ? first : ids[0], ids.includes(second) && second !== first ? second : ids.find((i) => i !== (first || ids[0]))];
  const back = h("a", { href: "#/today" }, "‹ Today");
  if (ids.length < 2) return [back, ...header("Compare", "Two stocks side by side"), notice(data.status) || card("Not enough forecasts to compare.")];
  const picker = (i) => h("select", { "aria-label": i ? "Second stock" : "First stock", onchange: (e) => {
    const next = [...chosen]; next[i] = e.target.value; location.hash = `#/compare/${encodeURIComponent(next.join(","))}`; } },
    [...data.rows].sort((a, b) => a.symbol.localeCompare(b.symbol)).map((r) => h("option", { value: r.id, selected: r.id === chosen[i] }, r.symbol)));
  const stocks = await Promise.all(chosen.map((id) => get(`/api/stock/${encodeURIComponent(id)}`)));
  // line the two price histories up by date and start both at 100
  const maps = stocks.map((s) => new Map(s.prices));
  const dates = stocks[0].prices.map((p) => p[0]).filter((d) => maps[1].has(d));
  const series = stocks.map((s, i) => ({ name: s.symbol, values: dates.map((d) => (100 * maps[i].get(d)) / maps[i].get(dates[0])) }));
  const rows = ["5", "20"].flatMap((n) => [["chance of above-usual volatility", "prob_above_usual"], ["forecast volatility (yearly)", "forecast_vol_annualised"],
    ["usual volatility (yearly)", "usual_vol_annualised"]].map(([label, key]) => [`${n}-day ${label}`, ...stocks.map((s) => pct(s.forecasts[n]?.[key]))]));
  return [back, ...header("Compare", "Two stocks side by side"), h("div", { class: "controls" }, picker(0), picker(1)),
    h("div", { class: "list" }, h("div", { class: "row head three" }, h("span", {}, ""), stocks.map((s) => h("span", {}, s.symbol))),
      rows.map(([label, a, b]) => h("div", { class: "row three" }, h("span", {}, label), h("strong", {}, a), h("strong", {}, b)))),
    dates.length > 1 ? [h("h2", {}, "Price, past year (both start at 100)"), card(legend(stocks[0].symbol, stocks[1].symbol), chart(series, dates, (v) => v.toFixed(0)))] : [],
    h("footer", {}, data.disclaimer)];
}

async function filings() {
  const results = h("div");
  let timer, latest = 0, query = "", category = "";
  const categories = await get("/api/filings/categories");
  const run = async (q) => {
    query = q;
    const mine = ++latest;
    const items = await get(`/api/filings?q=${encodeURIComponent(q)}&category=${encodeURIComponent(category)}`);
    if (mine === latest) results.replaceChildren(filingList(items));
  };
  await run("");
  return [...header("Filings", "Company announcements to the exchange, newest first"), h("div", { class: "controls" },
    h("input", { type: "search", placeholder: "e.g. INFY dividend", "aria-label": "Search filings",
      oninput: (e) => { clearTimeout(timer); timer = setTimeout(() => run(e.target.value), 250); } }),
    h("select", { "aria-label": "Category", onchange: (e) => { category = e.target.value; run(query); } },
      h("option", { value: "" }, "All categories"), categories.map((c) => h("option", { value: c }, c)))), results];
}

const outcomes = { OK: "Answer checked against the filings below", ABSTAIN: "The filings found do not answer this",
  BLOCKED: "The draft answer failed the checks, so it is withheld", DEGRADED: "The language model is unavailable; showing filings only",
  EMPTY: "No matching filings", REFUSED: "This company is not covered" };

async function ask() {
  const data = await get("/api/overview");
  const out = h("div");
  const who = h("select", { "aria-label": "Company" }, h("option", { value: "" }, "All companies"),
    [...data.rows].sort((a, b) => a.symbol.localeCompare(b.symbol)).map((r) => h("option", { value: r.id }, r.symbol)));
  const question = h("input", { type: "search", placeholder: "e.g. What is the latest dividend record date?", "aria-label": "Question",
    minlength: "2", maxlength: "300", required: true });
  const button = h("button", { class: "primary" }, "Ask");
  const form = h("form", { class: "controls ask", onsubmit: async (e) => {
    e.preventDefault();
    if (button.disabled || question.value.trim().length < 2) return;
    button.disabled = true;
    out.replaceChildren(card("Reading the filings… a local model can take up to a minute."));
    try {
      const res = await fetch("/v3/explain", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question.value.trim(), instrument_id: who.value || null }) });
      if (res.status === 404 || res.status === 405) {
        out.replaceChildren(card("Question answering is off. Restart the server with: python3 scripts/serve.py --evidence --explain (needs Ollama)."));
      } else if (!res.ok) {
        out.replaceChildren(card(`The server returned ${res.status}.`));
      } else {
        const d = await res.json();
        // Same order as the model saw them (newest first, documents/llm.py), so "[1]" in the answer is source [1] here.
        const items = [...(d.evidence?.pack?.evidence_items || [])].sort((a, b) => String(b.publication_time).localeCompare(String(a.publication_time)));
        out.replaceChildren(h("div", { class: "card" }, h("span", { class: `badge ${d.status === "OK" ? "OK" : "STALE"}` }, outcomes[d.status] || d.status),
          d.answer && h("p", { class: "answer" }, d.answer), d.answer_source === "digest" && h("p", { class: "dim" }, "Summary assembled from the filings, not written by the model.")),
          items.length ? h("h2", {}, "Sources") : "",
          items.length ? h("div", { class: "list" }, items.map((x, i) => h("article", { class: "filing" }, h("strong", {}, `[${i + 1}] ${x.instrument_id.split(":")[1]}`), " ",
            h("span", { class: "badge" }, x.category), " ", h("span", { class: "dim" }, String(x.publication_time).slice(0, 10)), h("p", {}, x.claim_span.slice(0, 400))))) : "");
      }
    } catch { out.replaceChildren(card("The server cannot be reached.")); }
    button.disabled = false;
  } }, who, question, button);
  return [...header("Ask", "Questions answered only from company filings, by a model running on this computer"), form, out,
    h("footer", {}, "Checks confirm the answer is supported by the filings shown; they cannot guarantee it is complete. Read the source before acting.")];
}

async function model() {
  const d = await get("/api/model");
  const last = d.rounds.at(-1);
  return [...header("Model", "How well it works, and what was tried"),
    h("div", { class: "grid" }, stat("5-day accuracy score (AUC)", last.h5.toFixed(3), "0.5 is a coin flip, 1.0 is perfect"),
      stat("20-day accuracy score (AUC)", last.h20.toFixed(3), "Past data, examined many times"), stat("Up or down", d.direction_auc.toFixed(2), "No better than chance")),
    h("h2", {}, "Progress"), card(legend("5-day", "20-day"),
      chart([{ name: "5-day", values: d.rounds.map((r) => r.h5) }, { name: "20-day", values: d.rounds.map((r) => r.h20) }], d.rounds.map((r) => r.label), (v) => v.toFixed(3))),
    Object.entries(d.reliability).map(([n, rows]) => [h("h2", {}, `${n}-day: when it said this, how often it happened`),
      h("div", { class: "list" }, h("div", { class: "row head" }, h("span", {}, "Model said"), h("span", {}, "Actually happened"), h("span", {}, "Cases")),
        rows.map((r) => h("div", { class: "row" }, h("strong", {}, pct(r.said, 0)), meter(r.happened), h("span", { class: "dim" }, r.n.toLocaleString("en-IN")))))]),
    h("h2", {}, "Price-range reliability"), h("div", { class: "grid" }, Object.entries(d.ranges).filter(([, v]) => v).map(([n, v]) =>
      stat(`${n}-day ranges held`, pct(v.coverage_90), "target 90%"))),
    h("h2", {}, "Tried and rejected"), h("div", { class: "list" }, d.rejected.map((t) => h("div", { class: "filing" }, t))),
    h("footer", {}, "The forward record, built from new data only, is the real test.")];
}

function refreshCard() {
  const status = h("p", { class: "dim" }, "Fetches the latest prices and filings dates, then rebuilds the forecasts. Takes a few minutes.");
  const button = h("button", { class: "primary", type: "button" }, "Refresh data now");
  const follow = async () => {
    const s = await get("/api/refresh").catch(() => null);
    if (!button.isConnected) return;
    if (!s) { status.textContent = "Lost contact with the server."; button.disabled = false; return; }
    button.disabled = s.running;
    if (s.running) { status.textContent = "Refreshing…"; setTimeout(follow, 3000); }
    else if (s.ok != null) status.textContent = s.ok ? "Done. Open Today to see the new forecasts." : `Refresh failed: ${s.message}`;
  };
  button.addEventListener("click", async () => {
    button.disabled = true;
    const res = await fetch("/api/refresh", { method: "POST", headers: { "X-Requested-With": "nifty-web" } }).catch(() => null);
    if (!res || (!res.ok && res.status !== 409)) { status.textContent = "Could not start the refresh."; button.disabled = false; return; }
    setTimeout(follow, 500);
  });
  setTimeout(follow, 0);
  return card(button, status);
}

async function system() {
  const d = await get("/api/system");
  return [...header("System", "Health and records"),
    h("div", { class: "grid" }, h("div", { class: "card stat" }, h("small", {}, "Serving state"), h("b", {}, h("span", { class: `badge ${d.health.state}` }, d.health.state)),
      h("small", {}, (d.health.reasons || []).join("; ") || "No active alerts")),
      stat("Forecasts", d.forecast_status, ""), stat("Audit log", d.audit_ok ? "Intact" : "BROKEN", `${d.audit_entries} entries`)),
    h("h2", {}, "Update"), refreshCard(),
    h("h2", {}, "Forward records"), h("div", { class: "list" }, Object.entries(d.ledger_entries).map(([k, v]) =>
      h("div", { class: "filing" }, h("strong", {}, k.replaceAll("_", " ")), ` · ${v} entries`))),
    h("p", {}, h("a", { href: "/api/backup", download: "" }, "Download a backup of these records")),
    h("h2", {}, "Forward scorecard"), Object.keys(d.scorecard).length ? h("div", { class: "grid" }, Object.entries(d.scorecard).map(([n, s]) =>
      stat(`${n}-day, new data only`, s.auc == null ? "–" : s.auc.toFixed(3), `AUC over ${s.n} graded forecasts`)))
      : card("No forecast has matured yet. Scores appear once each forecast's horizon has passed.")];
}

const routes = { today, portfolio, filings, ask, guide, compare, model, system, stock };
let visit = 0;
async function route() {
  const [, name = "today", arg] = location.hash.split("/");
  const page = routes[name] || today;
  const mine = ++visit;
  for (const a of document.querySelectorAll("nav a")) {
    const on = a.hash === `#/${["stock", "guide", "compare"].includes(name) ? "today" : name}`;
    on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
  }
  let content;
  try { content = await page(arg && decodeURIComponent(arg)); }
  catch (err) { content = [card(h("strong", {}, "Could not load this page. "), navigator.onLine === false || err instanceof TypeError ? "The server cannot be reached. Start it with python3 scripts/serve.py." : err.message)]; }
  if (mine !== visit) return;
  view.replaceChildren(...content.flat(Infinity).filter(Boolean));
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);
navigator.serviceWorker?.register("/sw.js").catch(() => { /* the app works without it */ });
addEventListener("keydown", (e) => { // "/" focuses search, as on Apple's web apps
  const box = view.querySelector("input[type=search]");
  if (e.key === "/" && box && document.activeElement !== box) { e.preventDefault(); box.focus(); }
});
route();
