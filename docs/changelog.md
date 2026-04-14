# Changelog

All notable changes to Freebase will be documented in this file.

## [1.0.1] - 2026-04-14

### Fixed

- **Execution/Worktree errors**: Fixed `fatal: Not a valid object name: 'unknown'` error during `freebase_execute_step`
  - Root cause: `PlanGenerateResponse` didn't store branch info, causing session creation to fall back to "unknown"
  - Added `branch` and `base_branch` fields to [`models.py`](../models.py) `PlanGenerateResponse`
  - Updated [`client.py`](../client.py) `execute_step()` to use plan's branch info
  - Updated [`client.py`](../client.py) `_setup_worktree()` to use plan's branch directly

### Changed

- Embedding service migrated from OpenAI to OpenRouter (using `mistralai/mistral-embed-2312`)
- Single API key (`OPENROUTER_API_KEY`) now handles both LLM and embeddings

## [1.0.0] - 2026-04-13

### Added

- Initial release of Freebase git history surgeon
- Core modules:
  - [`models.py`](../models.py) - Pydantic models for all data structures
  - [`git_ops.py`](../git_ops.py) - Low-level git subprocess operations
  - [`clustering.py`](../clustering.py) - Commit clustering with embedding-based similarity
  - [`state.py`](../state.py) - Session state persistence to `.git/freebase/`
  - [`client.py`](../client.py) - Main FreebaseClient API
- MCP Server ([`mcp_server.py`](../mcp_server.py)) with 8 tools:
  - `freebase_branch_analyze` - Analyze branch and build commit cluster model
  - `freebase_select_profile` - Auto-select optimal rebase profile via sub-agent
  - `freebase_plan_generate` - Generate machine-readable rebase plan
  - `freebase_plan_preview` - Human-readable plan summary
  - `freebase_execute_step` - Execute single rebase step in temp worktree
  - `freebase_conflict_analyze` - Analyze conflicts and propose resolutions
  - `freebase_abort_to_checkpoint` - Restore to safe checkpoint state
  - `freebase_range_diff_report` - Verify rebase correctness
- Test agent ([`agent.py`](../agent.py)) with CLI interface
- A2A agent card ([`.well-known/agent.json`](../.well-known/agent.json))
- Safety features:
  - Temporary worktree for all rebase operations
  - Checkpoint system for rollback
  - State persistence to disk for crash recovery
  - Backup refs before dangerous operations

### Design Decisions

- **Profile Selection**: Automatic via sub-agent (users don't understand profiles)
- **Worktree Isolation**: All rebases in temporary worktree for safety
- **Message Similarity**: Embedding-based for LLM-verbose commits
- **Test Integration**: None in V1 (agent handles separately)
- **State Persistence**: Disk at `.git/freebase/` for crash recovery
