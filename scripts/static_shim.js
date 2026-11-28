// Static hosting: answers the app's /api calls from the exported JSON snapshot (see export_static.py).
(() => {
  const real = window.fetch.bind(window);
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const load = async (name) => (await real(`/data/${name}.json`));
  const read = async (name) => (await load(name)).json();
  // Same rule as the server: forecasts older than 5 days (IST) are withheld.
  const stale = (date) => !date || (Date.parse(new Date(Date.now() + 19800000).toISOString().slice(0, 10)) - Date.parse(date)) / 86400000 > 5;

  function portfolio(snap, q) {
    const horizon = Number(q.get("horizon") || 5), block = snap.blocks[horizon];
    if (!block) return json({ status: "NO_DATA" });
    if (stale(block.data_date)) return json({ status: "STALE" });
    const amounts = new Map();
    for (const part of (q.get("holdings") || "").split(",").slice(0, 60)) {
      const at = part.lastIndexOf(":"), id = part.slice(0, at), v = Number(part.slice(at + 1));
      if (at < 0 || !(v > 0 && v < 1e12) || !Object.hasOwn(block.forecasts, id)) return json({ detail: "bad holdings" }, 422);
      amounts.set(id, (amounts.get(id) || 0) + v);
    }
    const ids = [...amounts.keys()], total = ids.reduce((s, i) => s + amounts.get(i), 0), w = ids.map((i) => amounts.get(i) / total);
    const corr = ids.map((a) => ids.map((b) => (a === b ? 1 : snap.corr[a][b])));
    const vol = ids.map((i) => block.forecasts[i].forecast_vol_annualised), usual = ids.map((i) => block.forecasts[i].usual_vol_annualised);
    const cov = vol.map((a, i) => vol.map((b, j) => a * b * corr[i][j]));
    const quad = (v, m, keep) => keep.reduce((s, i, a) => s + keep.reduce((t, j, b) => t + v[a] * m[i][j] * v[b], 0), 0);
    const all = ids.map((_, i) => i), variance = quad(w, cov, all), period = Math.sqrt((variance * horizon) / 252);
    const test = snap.test[horizon] || {};
    const without = (k) => {
      const keep = all.filter((j) => j !== k), sum = keep.reduce((s, j) => s + w[j], 0);
      return keep.length ? Math.sqrt(quad(keep.map((j) => w[j] / sum), cov, keep)) : null;
    };
    const pairs = all.flatMap((a) => all.slice(a + 1).map((b) => ({ a: snap.symbols[ids[a]], b: snap.symbols[ids[b]], correlation: corr[a][b] })))
      .sort((x, y) => y.correlation - x.correlation).slice(0, 3).map((p) => ({ ...p, correlation: Math.round(p.correlation * 100) / 100 }));
    return json({ status: "OK", horizon, data_date: block.data_date, total, volatility: Math.sqrt(variance),
      usual_volatility: Math.sqrt(quad(w, usual.map((a, i) => usual.map((b, j) => a * b * corr[i][j])), all)),
      undiversified: w.reduce((s, x, i) => s + x * vol[i], 0), typical_move: period * total,
      large_move: (test.multiplier_for_90pct || 1.645) * period * total, large_move_held: test.second_half_within_calibrated, linked: pairs,
      parts: all.map((k) => ({ without: without(k), id: ids[k], symbol: snap.symbols[ids[k]], weight: w[k],
        risk_share: (w[k] * cov[k].reduce((s, c, j) => s + c * w[j], 0)) / variance, volatility: vol[k] })).sort((a, b) => b.risk_share - a.risk_share) });
  }

  async function api(url) {
    const path = decodeURIComponent(url.pathname.slice(5)), q = url.searchParams;
    if (path === "portfolio") return portfolio(await read("portfolio"), q);
    if (path === "filings") {
      let docs = await read("filings");
      const category = q.get("category"), ids = q.get("instrument")?.split(",").map((i) => i.split(":").pop());
      if (category) docs = docs.filter((d) => d.category === category);
      if (ids) docs = docs.filter((d) => ids.includes(d.symbol));
      for (const word of (q.get("q") || "").toLowerCase().split(/\s+/).filter(Boolean).slice(0, 6))
        docs = docs.filter((d) => `${d.text} ${d.category} ${d.symbol}`.toLowerCase().includes(word));
      return json(docs.slice(0, Math.max(1, Math.min(Number(q.get("limit") || 30), 100))));
    }
    const res = await load(path.replace("filings/", "filings_").replace(/[/:]/g, "_").replace(/ /g, "-"));
    if (!res.ok) return json({}, 404);
    const body = await res.json();
    const date = body.data_date || Object.values(body.forecasts || {})[0]?.data_date || (await read("overview")).data_date;
    if (body.status === "OK" && stale(date)) {
      Object.assign(body, { status: "STALE" }, path === "overview" ? { rows: [], changes: {} } : path.startsWith("stock/") ? { forecasts: {}, peers: {} } : { rows: [] });
    }
    return json(body);
  }

  window.fetch = (input, init) => {
    const url = new URL(input.url || input, location.href);
    if (url.pathname.startsWith("/v3/")) return Promise.resolve(json({}, 404));
    if (!url.pathname.startsWith("/api/")) return real(input, init);
    return init?.method === "POST" ? Promise.resolve(json({}, 405)) : api(url);
  };
})();
