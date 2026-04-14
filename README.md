# Freebase - Git History Surgeon

An MCP-based interactive git history surgeon designed for AI-heavy branches.

## Overview

Freebase helps transform messy git histories into clean, reviewable commit sequences. It understands that AI-assisted development creates unique patterns - checkpoint commits, WIP saves, fixups, reverts, and interleaved changes that don't map well to traditional git workflows.

## Key Features

- **Semantic Clustering**: Groups commits by intent using file overlap, temporal proximity, and embedding-based message similarity
- **Auto Profile Selection**: Sub-agent automatically selects the optimal rebase strategy
- **Safe Execution**: All operations happen in a temporary worktree - your main branch is never touched
- **Conflict Analysis**: Tiered conflict classification with resolution strategies
- **Checkpoint Recovery**: One-command rollback to any previous state
- **Range-Diff Verification**: Verify rebase correctness before finalizing

## Profiles

| Profile | Optimizes For | When to Use |
|---------|--------------|-------------|
| `pr_clean` | Code review readability | PR preparation |
| `bisect_friendly` | Git bisect debugging | When bugs might need hunting |
| `chronology` | Historical investigation | Audit trails |
| `minimal` | Fewest changes | Conservative approach |
| `aggressive` | Maximum squashing | Messy AI branches |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

## Environment Variables

```bash
# Required - Used for both the agent LLM and embeddings (mistralai/mistral-embed-2312)
export OPENROUTER_API_KEY="your-key"

# Optional: Profile selection model (default: x-ai/grok-4-fast)
export FREEBASE_PROFILE_MODEL="x-ai/grok-4-fast"

# Optional: Logging
export LOGGER_PATH="/path/to/freebase.log"
export LOGGER_NAME="freebase_mcp"

# Optional: Observability
export LOGFIRE_API_KEY="your-key"
```

## Usage

### Running the Agent

```bash
# Default model
python agent.py

# Specific model
python agent.py --model openai/gpt-4o
python agent.py --model anthropic/claude-3.5-sonnet
```

### Example Conversation

```
> Analyze my feature branch against main

Analyzing branch feature-auth against main...

Found 23 commits in 5 clusters:
- Cluster 1: "Auth implementation" (8 commits) → Suggested: squash to 2
- Cluster 2: "Fixup commits" (5 commits) → Suggested: squash all
- Cluster 3: "Test additions" (4 commits) → Suggested: keep separate
- Cluster 4: "Formatting" (3 commits) → Suggested: squash all
- Cluster 5: "Documentation" (3 commits) → Suggested: squash all

Risk: Medium (auth.py touched 6 times)

> Create a rebase plan

Profile auto-selected: pr_clean (92% confidence)
Reason: Preparing for PR with many fixup commits

Plan created: 23 commits → 8 commits
- 3 questions need your input

Question 1: Cluster 1 mixes code and tests. Should it be split?
  [1] Keep together
  [2] Split into code + tests

> 2

Updated plan. Ready to execute.

> Execute the plan

Executing in temporary worktree at /tmp/freebase-work-1234...

Step 1/15: pick abc1234 "Add auth module"
✓ Complete

Step 2/15: fixup def5678 "Fix auth typo"
✓ Complete

...

Step 8/15: pick 111222 "Add auth tests"
⚠️ Conflict in src/auth.py

> Analyze the conflict

Conflict tier: structural
Explanation: Both branches modified the validate() method

Strategies:
1. Accept ours - Keep your new validation logic
2. Accept theirs - Keep base branch validation
3. Manual - Edit the file yourself

Recommended: Accept ours (risk: low)

> Accept ours

Conflict resolved. Continuing...

Step 15/15: Complete!

Rebase finished: 23 → 8 commits
Verifying with range-diff... ✓ Semantically equivalent

Ready to apply. Your main branch is still unchanged.
Run `git checkout feature-auth && git reset --hard freebase-temp-1234` to apply.
```

## Project Structure

```
freebase/
├── models.py           # Pydantic models
├── git_ops.py          # Low-level git operations
├── clustering.py       # Commit clustering with embeddings
├── state.py            # Session state persistence
├── client.py           # Main FreebaseClient
├── mcp_server.py       # MCP server (8 tools)
├── agent.py            # Test agent
├── agents/
│   └── freebase.md     # Agent system prompt
├── .well-known/
│   └── agent.json      # A2A agent card
└── notes/              # Development notes (git-ignored)
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `freebase_branch_analyze` | Analyze branch and build commit cluster model |
| `freebase_select_profile` | Auto-select optimal rebase profile (sub-agent) |
| `freebase_plan_generate` | Generate machine-readable rebase plan |
| `freebase_plan_preview` | Human-readable plan summary |
| `freebase_execute_step` | Execute single rebase step in temp worktree |
| `freebase_conflict_analyze` | Analyze conflicts and propose resolutions |
| `freebase_abort_to_checkpoint` | Restore to safe checkpoint state |
| `freebase_range_diff_report` | Verify rebase correctness |

## Safety Model

1. **Backup refs** created before any operation
2. **Temporary worktree** for all rebases
3. **State persistence** to `.git/freebase/`
4. **Operation logging** for audit
5. **One-command rollback** via checkpoints
6. **Range-diff verification** before finalize

## License

MIT
