// Caches only the app shell. /api is never cached: an old forecast must not look current.
const CACHE = "nifty-shell-v12";
const SHELL = ["/", "/static/app.js", "/static/app.css", "/static/icon.svg", "/static/manifest.webmanifest"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(caches.keys().then((keys) =>
  Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !SHELL.includes(url.pathname)) return;
  e.respondWith(fetch(e.request).then((res) => { // network first, so updates appear immediately
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(url.pathname, copy)); }
    return res;
  }).catch(() => caches.match(url.pathname)));
});
