# PV NarrativeGuard 1D7

**AI-Assisted ICSR Duplicate Screening & Safety Narrative Generation**

PV NarrativeGuard is a human-in-the-loop pharmacovigilance application that:

1. validates incoming ICSR data,
2. screens potential duplicates using structured and semantic evidence,
3. routes potential duplicates for qualified human review,
4. generates evidence-grounded safety narratives on demand,
5. performs source-grounded Narrative QC,
6. provides controlled MedDRA terminology assistance using deterministic safeguards,
   curated PV expressions, and FAISS retrieval,
7. records operational audit events.

## Final application

- Dashboard
- ICSR Work Queue
- Duplicate Review
- Narrative Generation
- Narrative QC
- Audit Trail
- Shared MedDRA Assistant

## Validation

The supplied Phase 1C baseline passed **50/50 tests**. The 1D7 release adds a
canonical final acceptance suite.

Run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v
```

The expected final suite is the original regression coverage plus the 1D7 release
acceptance tests.

See `FINAL_RELEASE_1D7.md` for the release definition and safety controls.
