# Phase 1B.4.2 — Clean Candidate Enrichment Integration

## Fix
- Moved deterministic PV Core candidate enrichment into a dedicated adapter method.
- `query()` now applies enrichment even when `retrieve_candidates()` is replaced or mocked.
- Preserves FAISS similarity while merging verified curated provenance.
- Keeps the validated `threw up` → `vomiting AE` → `10047700` behavior.

## Validation target
All existing tests plus the FAISS-shadowing regression must pass.
