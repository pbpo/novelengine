# Repository Guidelines

## Project Structure & Module Organization
- Run orchestration lives in `story_engine_main.py`, which wires CLI arguments into `story_engine.engine.runner.StoryEngineRunner`.
- Core modules sit under `story_engine/`: `config/` for typed settings loaders, `factory.py` for service assembly, and feature packages (`autonomy/`, `canon/`, `llm/`, `memory/`, `planning/`, `qc/`, `writing/`). Keep new functionality co-located with the matching domain package.
- The `engine/` package bridges the modular pieces back into the legacy `stc` engine; update it whenever new services need to plug into the old runtime.
- Runtime assets (`config.yaml`, `bible.txt`, `story_rag.sqlite`) are expected beside the entrypoint; document any custom paths in PRs.

## Build, Test, and Development Commands
- `python story_engine_main.py --config config.yaml --bible bible.txt` — run the modular engine end-to-end, emitting the generated novel and updating the RAG cache.
- `python -m story_engine.config.load_config config.yaml` — quick validation that configuration files parse and merge overrides correctly.
- `python -m pip install -r requirements.txt` (add if missing) — standardise local environments before running the engine or tests.

## Coding Style & Naming Conventions
- Use 4-space indentation, type hints, and dataclasses where state is grouped (matches existing modules such as `story_engine.engine.runner`).
- Prefer `snake_case` for functions and modules, `PascalCase` for classes, and module-level constants in `SCREAMING_SNAKE_CASE`.
- Rely on `black` + `ruff` (install if absent) for formatting and linting; run them before submitting patches.
- Keep imports explicit (`from story_engine.autonomy import strategy`) and avoid wildcard imports to preserve clarity.

## Testing Guidelines
- Add `pytest`-style unit tests under `tests/`, mirroring package paths (e.g. `tests/config/test_load_config.py`).
- Mock network-bound dependencies such as Anthropic/OpenAI clients to keep suites deterministic.
- Target >=80% coverage for new modules; include regression tests when fixing bugs uncovered in `engine/legacy_bridge.py`.
- Execute `pytest` before opening a PR and capture key output when tests fail.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit subjects (`Refactor canon services loader`); include a short body when context helps reviewers.
- Group logically-related changes per commit and note config or data updates explicitly.
- PR descriptions should summarise behaviour changes, link tracking issues, and attach artefacts (generated story samples, new configs) when relevant.
- Request reviews from maintainers responsible for the touched modules (e.g. autonomy, qc) and wait for CI/test confirmation before merging.

## Legacy Integration Tips
- When touching legacy interoperability, update `story_engine.engine.legacy_bridge` alongside the new module to keep both runtimes aligned.
- Verify end-to-end runs with an empty `story_rag.sqlite` to confirm ingestion fallbacks still behave as expected.
