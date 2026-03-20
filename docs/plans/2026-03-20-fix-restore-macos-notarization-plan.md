---
title: Restore macOS Notarization in Release Workflow
type: fix
status: completed
date: 2026-03-20
---

# Restore macOS Notarization in Release Workflow

Notarization was disabled in commit `644f1de2d` ("fix: skip notarization for now, keep code signing only") because the required Apple secrets weren't yet configured. Now that the secrets are set up, re-enable notarization so users no longer need to run `xattr -cr /Applications/Char.app`.

## Acceptance Criteria

- [x] `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` env vars are passed to the Tauri build step in `release.yaml`
- [x] Comment about notarization being disabled is removed
- [ ] Built `.dmg` is notarized and passes Gatekeeper without `xattr -cr`

## Context

Tauri v2 automatically submits the app to Apple's notarization service when these three env vars are present during build:
- `APPLE_ID` — Apple Developer account email
- `APPLE_PASSWORD` — app-specific password (generated at appleid.apple.com)
- `APPLE_TEAM_ID` — 10-character team identifier

The upstream workflow (`desktop_cd.yaml:153-155`) already has this working. The fork's `release.yaml` just needs the same three lines restored.

**Prerequisites:** The three GitHub Actions secrets (`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) must be configured in the repository settings before running the workflow.

## MVP

### `.github/workflows/release.yaml`

Replace lines 57-58 (the disabled-notarization comments) with:

```yaml
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_PASSWORD: ${{ secrets.APPLE_PASSWORD }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
```

## Sources

- Commit that removed notarization: `644f1de2d`
- Upstream reference: `.github/workflows/desktop_cd.yaml:153-155`
