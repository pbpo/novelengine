# Story Engine Package (modular refactor)

This package hosts the refactored components that mirror the functionality
currently embedded inside `stc.py`. Everything lives side-by-side with the
legacy implementation so we can migrate incrementally without touching the
live pipeline.

## What is available now?

- `story_engine.config` – dataclasses (`EngineConfig`, `AutonomyConfig`, …)
  plus `load_config()` that merges YAML/JSON overrides into typed settings.
- `story_engine.llm` – Anthropic/OpenAI client wrappers, shared cache and
  logging helpers (same behaviour as the inline version in `stc.py`).
- `story_engine.autonomy` – `RhythmUCB`, `CanonCouncil`, `AutonomyScorer`
  lifted verbatim from the engine, ready to be imported as modules.
- `story_engine.canon` – `CanonLock` helper with the same interface.
- `story_engine.writing` – the `RhythmMixer` constraint generator.
- `story_engine.factory` – helper functions that assemble these services from
  the new configuration objects (e.g. `create_autonomy_services`,
  `create_canon_services`, `create_llm_cache`).

## Usage idea

```python
from story_engine.config import load_config
from story_engine.factory import (
    create_autonomy_services,
    create_canon_services,
    create_llm_cache,
)

cfg = load_config("config.yaml")
cache = create_llm_cache(cfg)
autonomy = create_autonomy_services(cfg)
canon = create_canon_services(cfg)
```

These helpers return the same concrete classes that `stc.py` currently
instantiates. Hook them in gradually by switching the imports in `stc.py` once
we are ready to migrate a subsystem.

## Next steps

- Extract QC, pacing, and planner components into `story_engine/qc`,
  `story_engine/planning`, etc.
- Introduce an `Engine` façade inside `story_engine` that composes all new
  services, so `stc.py` can become a thin CLI wrapper.
- Add unit tests per package (LLM clients can be mocked) to validate behaviour
  before swapping the live engine over.
