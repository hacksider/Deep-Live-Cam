# Document Management

## Document Organization

All project documentation is organized in `docs/` with clear categorization:

- **docs/prds/** — Product Requirements Documents (features, user stories, acceptance criteria)
- **docs/adrs/** — Architecture Decision Records (design decisions, trade-offs, rationale)
- **docs/prps/** — Product Requirement Prompts (detailed implementation guides, PRPs)
- **docs/blueprint/** — Blueprint Development system (manifest, feature tracker, work orders)

## Naming Conventions

- Use kebab-case for filenames: `authentication-flow.md`, not `AuthenticationFlow.md`
- ADRs use numeric prefix: `0001-initial-architecture.md`, `0002-database-choice.md`
- PRDs describe what (requirements): `user-authentication.md`, `faceswap-pipeline.md`
- PRPs describe how (implementation): `implement-face-detection.md`, `optimize-gpu-inference.md`

## When to Create Documents

### Create a PRD when:

- Defining a new feature or subsystem
- Gathering requirements from stakeholders
- Specifying acceptance criteria and user stories
- Describing non-functional requirements (performance, security)

### Create an ADR when:

- Making significant architectural decisions
- Choosing between multiple technical approaches
- Documenting trade-offs and consequences
- Recording rationale for implementation choices

### Create a PRP when:

- Scoping implementation work for a feature
- Breaking down complex tasks into steps
- Defining success criteria for implementation
- Planning test strategies

## Document Detection

Claude will prompt you to create documents when:

- You're discussing requirements in a conversation → "Should we create a PRD?"
- You're making architectural decisions → "Should we record this as an ADR?"
- You're planning implementation details → "Should we create a PRP?"

Accept the prompt to create the document immediately, or use commands:

- `/blueprint:derive-prd` — Create PRD from existing documentation
- `/blueprint:derive-adr` — Derive ADRs from codebase analysis
- `/blueprint:prp-create` — Manually create a Product Requirement Prompt

## Markdown Format

### PRD Structure

```markdown
# Feature: [Feature Name]

## Overview
Brief description of what this feature does.

## User Stories
- As a [user role], I want to [action], so that [benefit]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Non-Functional Requirements
- Performance: [requirements]
- Security: [requirements]
```

### ADR Structure

```markdown
# ADR: [Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context
Why we need to make this decision.

## Decision
What we decided to do.

## Consequences
- Positive: ...
- Negative: ...
```

### PRP Structure

```markdown
# PRP: [Implementation Title]

## Objective
What we're implementing and why.

## Implementation Steps
1. Step 1: Description
2. Step 2: Description

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Testing Strategy
How we'll verify the implementation.
```

## Feature Tracking

The feature tracker (`docs/blueprint/feature-tracker.json`) syncs with README.md to track:

- Feature requirements and their completion status
- Implementation progress against requirements
- Blockers and dependencies

Update the tracker via:

- `/blueprint:feature-tracker-sync` — Sync tracker with source document
- `/blueprint:feature-tracker-status` — View completion statistics

## Version Control

### Always commit:

- `CLAUDE.md` — Project guidelines (overview)
- `.claude/rules/` — Modular rules (specific guidelines)
- `docs/prds/`, `docs/adrs/`, `docs/prps/` — Requirements and decisions

### Do NOT commit:

- `docs/blueprint/work-orders/` — Task packages (task-specific, may contain sensitive details)
- Generated files in `docs/blueprint/ai_docs/` — Curated on-demand

## Related Commands

- `/blueprint:status` — Check documentation status
- `/blueprint:sync` — Detect stale or outdated documents
- `/blueprint:claude-md` — Update CLAUDE.md with current guidelines
- `/blueprint:derive-prd`, `/blueprint:derive-adr` — Auto-generate documents
