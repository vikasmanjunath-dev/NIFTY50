# Log (append only)
Assumptions: keep the single main branch the owner asked for earlier (the loop prompt says "work on a branch"); commit locally, push only when asked.

- 2026-11-28 committed pending live-record work; 389 tests; 9e3e253
- 2026-11-28 B-01 fixed; 393 tests; 0f24c6a
- 2026-11-28 B-02 fixed; 394 tests; 665cf73
- 2026-11-28 B-03 B-04 fixed, seen at phone width; 395 tests; bb69660
- 2026-11-28 B-05 fixed; fuzzed /v3 posts; bf3df97
- 2026-11-28 B-06 fixed; 398 tests; cf71b75; forecast status OK, 50 stocks, 61 routes checked, 0 failed
- STATUS done: 6 bugs; in progress: none; blocked: backup destination, launchd install, human labels (owner); open bugs: 0 known; next: H-06
- 2026-11-28 H-06 H-07 done (B-07..B-09); 400 tests; d2ef846
- 2026-11-28 H-08 done (B-10); 401 tests, 61 routes ok; c4d04fa
- STATUS done: 10 bugs; in progress: none; blocked: X-01..X-03 (owner); open bugs: 0 known; next: H-09, H-10
- 2026-11-28 reprioritised H-11 and H-13 above H-09 (more user value)
- 2026-11-28 H-11 routes timed: /api/record 161 ms every call -> 1 ms cached; first /api/system 1.2 s after start (cold price load, left); 402 tests; 09e3c9a
- 2026-11-28 H-13 dark theme: every text colour >= 4.5:1; no change needed
- 2026-11-28 H-14 done: first /api/system 1.2 s -> 0.23 s (measured on a running server); console 404s were from the earlier bad-address test; 403 tests; d047af3
