# Development Workflow

## Test-Driven Development (TDD)

Follow RED → GREEN → REFACTOR workflow:

1. **RED**: Write a failing test that defines desired behavior
2. **GREEN**: Implement minimal code to make the test pass
3. **REFACTOR**: Improve code quality while keeping tests green

## Package Management

- Use `uv` for all dependency and environment management
- `uv sync` — Install dependencies from pyproject.toml
- `uv run` — Run scripts with proper environment
- Modify `pyproject.toml`, never edit `requirements.txt` or `uv.lock` manually

## Python Version

- Python 3.10 only (pinned via `.python-version` and `pyproject.toml requires-python = "==3.10.*"`)
- Managed by `mise` for consistent Tcl/Tk support

## Commit Conventions

Use conventional commits for clear history and release automation:

- `feat:` — New features (minor version bump)
- `fix:` — Bug fixes (patch version bump)
- `feat!:` or `BREAKING CHANGE:` — Breaking changes (major version bump)
- `chore:`, `docs:`, `style:`, `refactor:` — No version bump
- All commits should be atomic and pass tests

## Task Runner

- Use `just` for project tasks — run `just --list` to see recipes
- Key recipes: `just setup`, `just start`, `just start-cpu`, `just clean`
- Recipes handle Tcl/Tk environment setup for GUI

## Running the Application

### With just (recommended)
```bash
just start            # Platform-default GPU
just start-cpu        # CPU only
just start-with rocm  # Specific provider
```

### With uv directly
```bash
uv run run.py --execution-provider coreml
```

### Headless mode (no Tcl/Tk needed)
```bash
uv run run.py -s source.jpg -t target.mp4 -o output.mp4
```

## Code Quality

- Prefer functional composition over class hierarchies
- Keep functions small and focused on a single responsibility
- Use pure functions without side effects when possible
- Data structures and functions over objects with methods
- YAGNI: Implement only what's immediately necessary
- DRY: Extract common logic, but avoid premature abstraction

## Error Handling

- Validate inputs early and fail immediately on invalid data
- Use "fail fast" principle — surface failures immediately
- Log errors comprehensively before failing
- Propagate errors up the stack to the appropriate handler

## Debugging

- Do not assume root cause without evidence
- Verify hypotheses against logs, timestamps, and context
- If diagnosis is corrected, fully re-investigate rather than patching
- Use systematic debugging approaches

## Security

- Never commit API tokens or secrets
- Use environment variables for configuration
- Validate at system boundaries (user input, external APIs)
- Avoid command injection, XSS, SQL injection risks
