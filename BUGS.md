# Bugs found in the hardening loop (earlier ones are in TRACKER.md)
| ID | Repro | Root cause | Fix | Regression test | Commit |
|---|---|---|---|---|---|
| B-01 | GET /api/stock/NSE_INFY, nse:infy, 500 characters: 500 | cache path sanitiser mapped look-alikes onto real files | exact id check, 404 | test_stock_rejects_malformed_ids_with_404 | 0f24c6a |
| B-02 | page request while a data file is being rewritten: 500 | readers assumed whole files | keep last good copy; atomic forecast write | test_half_written_files_do_not_crash_pages | 665cf73 |
| B-03 | truncated page address shows "URI malformed", no heading | unguarded decodeURIComponent | safe decode, error heading | test_text_colours_meet_wcag_aa_and_errors_have_a_heading | bb69660 |
| B-04 | light-theme links 4.0:1 contrast | system blue below AA for small text | darker blue | same test | bb69660 |
| B-05 | POST /v3/analyse with as_of 0001-01-01: 500 | timestamp overflow | validate year range, 422 | test_out_of_range_as_of_is_a_validation_error_not_a_crash | bf3df97 |
| B-06 | interrupted ingestion could leave half a price file | direct to_parquet onto the live file | write aside, then replace | test_atomic_write_keeps_the_old_file_if_writing_fails | cf71b75 |
