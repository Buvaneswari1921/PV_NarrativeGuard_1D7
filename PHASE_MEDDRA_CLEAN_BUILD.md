# PV NarrativeGuard — Clean MedDRA Build

This build consolidates the Phase 1B MedDRA improvements into the existing project structure without duplicate modules or parallel replacement implementations.

## Included behavior
- 150-concept PV Core vocabulary
- natural-language PV expression resolution
- narrative verb normalization
- negation protection
- clinical-context ambiguity handling
- validated loose-stools mapping preservation
- deterministic candidate enrichment at the MedDRA adapter boundary
- FAISS candidate + curated provenance merge without fabricating terminology

## Important fix in this build
Curated PV-expression enrichment is applied after `retrieve_candidates()` returns as well as within the normal retrieval path. This means a mocked, replaced, or future alternative retrieval backend cannot bypass the deterministic enrichment layer.

## Validation
Run the full suite in the project's existing `.venv`:

```powershell
python -m pytest tests -v
```

Do not copy individual modules into the older project. Use this build as a complete project tree. Preserve your existing `.env` separately.


## Phase 1C safety gate
- Curated clinical ambiguities are terminal at the adapter boundary until required clinical context is present.
- `fast heartbeat` / `rapid heartbeat` cannot be auto-promoted to tachycardia from FAISS similarity alone.
- Explicit measured heart-rate context can resolve to the local tachycardia candidate.
