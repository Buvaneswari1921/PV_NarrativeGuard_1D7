# PV NarrativeGuard — Visual Redesign Notes

## What changed
- Added a shared dark visual theme in `.streamlit/config.toml` with midnight navy surfaces, teal accent, high-contrast text, and consistent typography.
- Added reusable presentation-only UI components in `src/ui/components.py`:
  - branded hero headers
  - compact six-step workflow navigator
  - shared card, metric, input, table, and navigation styling
  - responsive layouts and reduced-motion support
- Applied the visual system to all six Streamlit pages: Dashboard, ICSR Work Queue, Duplicate Review, Narrative Generation, Narrative QC, and Audit Trail.
- Added page-specific hero copy and context chips to communicate the purpose and guardrails of each workspace.

## Design principles
- Human PV decisions remain authoritative.
- Similarity scores are presented as evidence for review, not as an automatic duplicate decision.
- Narrative generation remains eligibility-gated and on demand.
- QC and audit surfaces emphasize traceability and review rather than unsupported claims of clinical validation.
- Changes are presentation-only; existing pipeline and persistence modules were not intentionally altered.

## Run
From this project directory, install the dependencies listed in `requirements.txt`, then run:

```bash
streamlit run app.py
```

## Verification note
All Python files compile successfully. The available test run reported 106 passing tests; remaining tests could not be fully evaluated in this environment because packages such as Streamlit, FAISS, and OpenAI were not installed. Install the project requirements and rerun the complete test suite locally before relying on the updated application.
