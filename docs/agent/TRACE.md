# TRACE

This file maps task IDs to commits for idempotency.

Format:
- <TASK_ID> -> <COMMIT_SHA> : <summary>

M0.5 -> 4257224a78ad096d9e3b1f33365113c39667d17a : docs: consolidate M0/M1 DoD pointers (one canonical place)
M0.4 -> bafe65d95a51bc679b0644e1871bf558ff758ee0 : docs: tighten spec_alignment + scope notes for M0/M1
M0.3 -> e4d2b5a25aea779a812a5e38a0888546aeef270f : canonical event schema + validation scaffold + tests
M1.2 -> a4fe6be83552ce9913aa35b9ec7bd51b34016ce0 : docs: add minimal bundle load + version validation snippet (and link from README)

M1.3 -> 89d4c1e66d425435c16498f769cf5e6f54c809c4 : spatial: add lon/lat -> cell_id lookup helper (lookup + tests) [tick_ts=20260214T220122Z]
