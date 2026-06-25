# Plan: Integrate Windows App Patch Layers

## Scope
- In: Replace runtime patch modules with normal app modules and a canonical launcher entrypoint.
- In: Light modularization around app_base, output tasks, UI builders, processing options, and live webcam helpers.
- Out: GUI redesign, new dependencies, tests, validation runs, PyInstaller changes.

## GUI Components Affected
- [x] Main window
- [x] Panels/widgets
- [ ] Dialogs
- [ ] Assets (icons, qss, etc.)
- [x] Config files
- [ ] PyInstaller spec

## Action items
- [x] Create feature branch from main.
- [x] Capture implementation plan.
- [x] Convert patch modules into normal imported modules without side-effect install hooks.
- [x] Route launcher to the canonical app entrypoint.
- [x] Remove obsolete patch files and patch comments/imports.
- [x] Sync release notes and docs.
- [ ] Commit changes on feature branch.
- [ ] Push branch and open PR.

## Decisions
- Use light modularization instead of a full widget rewrite.
- Preserve current GUI behavior and settings compatibility.
- Do not run tests, GUI smoke checks, or builds; user owns validation.

## Open questions
- None blocking.

## Validation
- [ ] Manual GUI test: USER
- [ ] Build exe: USER (if desired)
