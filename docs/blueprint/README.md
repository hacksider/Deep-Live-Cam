# Blueprint Development

This directory contains the Blueprint Development system for deep-live-cam, enabling structured documentation, feature tracking, and architecture decision recording.

## Directory Structure

- **manifest.json** — Configuration and version tracking for Blueprint
- **feature-tracker.json** — Feature requirement tracking synced with README.md
- **work-orders/** — Task packages for subagents (completed/, archived/)
- **ai_docs/** — Curated documentation for LLM context (libraries/, project/)

## Related Documentation

- **docs/prds/** — Product Requirements Documents
- **docs/adrs/** — Architecture Decision Records
- **docs/prps/** — Product Requirement Prompts

## Key Commands

- `/blueprint:status` — Check version and configuration
- `/blueprint:derive-prd` — Derive requirements from documentation
- `/blueprint:feature-tracker-sync` — Sync feature tracker with source document
- `/blueprint:sync` — Check for stale generated content
- `/blueprint:claude-md` — Update CLAUDE.md with project guidelines

## Configuration

- **Rules mode:** Modular (.claude/rules/) + single CLAUDE.md
- **Feature tracking:** Enabled (source: README.md)
- **Document detection:** Enabled (Claude prompts for PRD/ADR/PRP creation)
- **Task scheduling:** Prompt before running (all tasks manual by default)
