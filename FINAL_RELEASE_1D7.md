# PV NarrativeGuard 1D7 — Final Consolidated Release

## Release status

**Final consolidated build based on the validated Phase 1C baseline.**

The Phase 1C baseline supplied with this release had **50/50 regression tests passing**.
1D7 consolidates that baseline into one clean project tree and adds a final acceptance
layer covering navigation, runtime artifacts, pipeline counts, Narrative QC negative
control, knowledge-gap governance, and audit integrity.

## Canonical application workflow

1. Dashboard
2. ICSR Work Queue
3. Duplicate Review
4. Narrative Generation
5. Narrative QC
6. Audit Trail

The MedDRA Assistant is a shared capability and is not duplicated as a standalone page.

## MedDRA safety architecture

The final terminology path preserves the controlled ordering:

- validated deterministic mappings
- curated PV Core expression resolution
- controlled inventory
- FAISS retrieval/evidence
- ambiguity safeguards
- unsupported/knowledge-gap handling

A verified curated PV expression must not be shadowed by a weak FAISS hit.

Representative protected scenarios include:

- `patient has headache`
- `patient has loose stools`
- `patient threw up twice`
- `patient has itchy skin`
- `patient denies headache`
- `fast heartbeat`
- measured heart-rate context resolving the appropriate clinical concept
- unknown terminology producing a controlled knowledge gap

## Human-in-the-loop controls

The system does not silently promote an unsupported knowledge-gap query into the
production terminology index. Approval is recorded in the review queue; ingestion,
embedding generation, and FAISS rebuild remain separate controlled operations.

Potential duplicates remain blocked until the configured human PV decision is available.

## Final validation

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v
```

For a fresh environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -v
streamlit run app.py
```

## Environment

Copy `.env.example` to `.env` and provide credentials only if an optional LLM-backed
feature is being used. Never commit real credentials.

## Release principle

There is one canonical implementation of each production capability. Backup modules,
temporary test copies, cache directories, and obsolete duplicate implementations are
excluded from the 1D7 release tree.
