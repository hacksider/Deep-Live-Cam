# Testing Requirements

## Test Framework

- Use `pytest` for unit and integration testing
- Run tests frequently during development (on every change)
- Use `uv run pytest` or `pytest` from activated venv

## Test-Driven Development

When implementing a feature or bug fix:

1. **Write failing test first** — Define expected behavior
2. **Implement minimal code** — Make test pass with simplest solution
3. **Refactor safely** — Keep tests green while improving code quality

## Test Execution Tiers

| Tier | When | Command | Duration |
|------|------|---------|----------|
| Unit | After every change | `pytest -x -q` | < 30s |
| Integration | After feature completion | `pytest -v` | < 5min |
| Full | Before commit/PR | `pytest` | < 30min |

## Test Organization

- Unit tests for individual functions/modules — test behavior in isolation
- Integration tests for multiple components working together
- E2E tests for complete workflows (GUI startup, faceswap pipeline)
- Fixtures for reusable test data and setup

## Coverage

- Aim for >80% code coverage on new code
- Focus on critical paths and error handling
- Use `pytest --cov` to measure coverage
- Track regressions — ensure new tests don't break existing functionality

## Performance Testing

- Benchmark GPU providers (CoreML, CUDA, ONNX CPU)
- Monitor FPS during faceswap operations
- Verify no FPS drops after code changes (15min minimum test duration)
- Profile startup time — detect regressions

## GUI Testing

- Test CustomTkinter UI components separately from business logic
- Mock face detection/swap operations in GUI tests
- Test event handlers and state updates
- Verify no Tk errors or crashes (especially ImageTk/PIL integration)

## Tkinter-Specific Tests

Given the complex Tcl/Tk environment setup:

- Test ImageTk.PhotoImage creation with actual image data
- Verify GUI components initialize with proper window management
- Test camera enumeration on different platforms
- Check for ImageTk `PyImagingPhoto` errors (indicates Tcl/Tk misconfiguration)

## Failing Tests

- Fix tests before committing — never skip with `@pytest.mark.skip`
- If test is expected to fail, mark with `@pytest.mark.xfail` with reason
- Document why test is failing/skipped in a comment

## Continuous Testing

- Run `pytest --watch` during development for instant feedback
- Use `just` recipes that incorporate testing
- Catch regressions before they reach main branch
