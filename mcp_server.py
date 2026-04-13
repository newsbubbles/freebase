"""Freebase MCP Server - Git history surgery tools for AI-heavy branches.

This MCP server exposes tools for intelligent git history manipulation,
including branch analysis, profile selection via sub-agent, rebase planning,
and conflict resolution.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from mcp.server.fastmcp import Context, FastMCP

from client import FreebaseClient
from models import (
    # Request models
    BranchAnalyzeRequest,
    SelectProfileRequest,
    PlanGenerateRequest,
    PlanPreviewRequest,
    ExecuteStepRequest,
    ConflictAnalyzeRequest,
    AbortToCheckpointRequest,
    RangeDiffRequest,
    # Response models
    BranchAnalyzeResponse,
    SelectProfileResponse,
    PlanGenerateResponse,
    PlanPreviewResponse,
    ExecuteStepResponse,
    ConflictAnalyzeResponse,
    AbortToCheckpointResponse,
    RangeDiffResponse,
    # Data models
    ConflictResolution,
    UserDecision,
    Profile,
)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_mcp_logging():
    """Standard logging setup for MCP servers."""
    logger_name = os.getenv("LOGGER_NAME", "freebase_mcp")
    logger_path = os.getenv("LOGGER_PATH")

    logger = logging.getLogger(logger_name)

    if logger_path and not logger.handlers:
        handler = logging.FileHandler(logger_path, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


logger = setup_mcp_logging()


# =============================================================================
# MCP Context
# =============================================================================


class MCPContext:
    """Context holding initialized clients for tool access."""

    def __init__(self, profile_agent: Agent):
        self.profile_agent = profile_agent
        self._clients: dict[str, FreebaseClient] = {}

    def get_client(self, repo_path: str) -> FreebaseClient:
        """Get or create a FreebaseClient for a repository."""
        if repo_path not in self._clients:
            self._clients[repo_path] = FreebaseClient(repo_path)
        return self._clients[repo_path]


# =============================================================================
# Profile Selection Sub-Agent
# =============================================================================


class ProfileSelectionOutput(BaseModel):
    """Output from the profile selection sub-agent."""

    selected_profile: Literal[
        "pr_clean", "bisect_friendly", "chronology", "minimal", "aggressive"
    ] = Field(description="The selected rebase profile")
    reasoning: str = Field(description="Explanation of why this profile was chosen")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")
    alternative: Optional[
        Literal["pr_clean", "bisect_friendly", "chronology", "minimal", "aggressive"]
    ] = Field(default=None, description="Second-best profile if confidence is low")


PROFILE_AGENT_SYSTEM_PROMPT = """
You are a git history optimization expert. Your job is to select the best rebase profile
based on branch analysis data.

Profiles:
- pr_clean: Optimize for code review readability. Best for PR preparation.
- bisect_friendly: Preserve granularity for git bisect debugging. Good when bugs might need hunting.
- chronology: Keep original order for historical investigation. Use for audit trails.
- minimal: Fewest changes possible. Conservative, low risk approach.
- aggressive: Maximum squashing. Good for messy AI-generated branches with many checkpoints.

Consider:
1. Fixup count: High fixups → aggressive or pr_clean
2. WIP/checkpoint count: Many → aggressive
3. Revert count: Reverts suggest debugging needed → bisect_friendly or chronology
4. Mixed-purpose clusters: Need user decisions → minimal or chronology
5. User context: "preparing for PR" → pr_clean, "debugging" → bisect_friendly
6. Risk level: High risk → minimal

Output your selection with clear reasoning.
"""


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[MCPContext]:
    """Manage MCP server lifecycle."""
    logger.info("Initializing Freebase MCP Server")

    try:
        # Create profile selection sub-agent
        model = os.getenv("FREEBASE_PROFILE_MODEL", "openai:gpt-4o-mini")
        profile_agent = Agent(
            model,
            output_type=ProfileSelectionOutput,
            system_prompt=PROFILE_AGENT_SYSTEM_PROMPT,
        )

        logger.info(f"Profile agent initialized with model: {model}")
        yield MCPContext(profile_agent=profile_agent)

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise ValueError(f"Failed to initialize Freebase MCP Server: {str(e)}")
    finally:
        logger.info("Freebase MCP Server shutdown")


# =============================================================================
# FastMCP Server
# =============================================================================

mcp = FastMCP("Freebase", lifespan=lifespan)


# =============================================================================
# Request Models for MCP Tools
# =============================================================================


class FreebaseBranchAnalyzeRequest(BaseModel):
    """Request to analyze a git branch."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to analyze")
    base_branch: str = Field(default="main", description="Base branch for comparison")
    include_diffs: bool = Field(default=False, description="Include full diff content")
    cluster_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Similarity threshold for clustering"
    )


class FreebaseSelectProfileRequest(BaseModel):
    """Request to auto-select the optimal rebase profile."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to analyze")
    base_branch: str = Field(default="main", description="Base branch")
    context: Optional[str] = Field(
        default=None,
        description="User context (e.g., 'preparing for PR', 'debugging issue')",
    )


class FreebasePlanGenerateRequest(BaseModel):
    """Request to generate a rebase plan."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to rebase")
    base_branch: str = Field(default="main", description="Target base branch")
    profile_override: Optional[
        Literal["pr_clean", "bisect_friendly", "chronology", "minimal", "aggressive"]
    ] = Field(default=None, description="Override auto-selected profile (rarely needed)")
    auto_squash_fixups: bool = Field(default=True, description="Auto-squash obvious fixups")
    isolate_formatting: bool = Field(
        default=True, description="Keep formatting commits separate"
    )
    drop_reverted: bool = Field(
        default=True, description="Drop commits neutralized by reverts"
    )
    user_decisions: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Answers to previous questions"
    )


class FreebasePlanPreviewRequest(BaseModel):
    """Request to preview a rebase plan."""

    repo_path: str = Field(description="Path to git repository")
    plan_id: Optional[str] = Field(default=None, description="Existing plan ID to preview")
    branch: Optional[str] = Field(
        default=None, description="Branch if generating new preview"
    )
    base_branch: str = Field(default="main", description="Base branch")


class FreebaseExecuteStepRequest(BaseModel):
    """Request to execute a rebase step."""

    repo_path: str = Field(description="Path to git repository")
    plan_id: str = Field(description="Plan to execute")
    step_index: Optional[int] = Field(default=None, description="Specific step (default: next)")
    conflict_resolution: Optional[dict[str, Any]] = Field(
        default=None, description="Resolution for current conflict"
    )


class FreebaseConflictAnalyzeRequest(BaseModel):
    """Request to analyze a conflict."""

    repo_path: str = Field(description="Path to git repository")
    file_path: Optional[str] = Field(default=None, description="Specific file to analyze")
    include_context: bool = Field(default=True, description="Include surrounding code")
    context_lines: int = Field(default=10, description="Lines of context")


class FreebaseAbortRequest(BaseModel):
    """Request to abort to a checkpoint."""

    repo_path: str = Field(description="Path to git repository")
    checkpoint_id: Optional[str] = Field(default=None, description="Specific checkpoint")
    cleanup_worktree: bool = Field(default=True, description="Remove temp worktree")


class FreebaseRangeDiffRequest(BaseModel):
    """Request for range-diff report."""

    repo_path: str = Field(description="Path to git repository")
    original_range: str = Field(description="Original commit range (base..head)")
    rebased_range: str = Field(description="Rebased commit range")
    include_diffs: bool = Field(default=False, description="Include actual diff content")


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool()
async def freebase_branch_analyze(
    request: FreebaseBranchAnalyzeRequest, ctx: Context
) -> dict[str, Any]:
    """Analyze a git branch and build a structured model of the commit stack.

    This tool inspects a branch and groups commits into semantic clusters,
    identifying fixups, reverts, formatting changes, and more. It provides
    risk assessment and suggested cleanup actions.

    Use this as the first step before generating a rebase plan.
    """
    logger.info(f"Analyzing branch {request.branch} in {request.repo_path}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        result = await client.analyze_branch(
            BranchAnalyzeRequest(
                repo_path=request.repo_path,
                branch=request.branch,
                base_branch=request.base_branch,
                include_diffs=request.include_diffs,
                cluster_threshold=request.cluster_threshold,
            )
        )

        logger.info(
            f"Analysis complete: {result.total_commits} commits, "
            f"{len(result.clusters)} clusters"
        )

        return result.model_dump()

    except Exception as e:
        logger.error(f"Branch analysis failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_select_profile(
    request: FreebaseSelectProfileRequest, ctx: Context
) -> dict[str, Any]:
    """Automatically select the optimal rebase profile for a branch.

    This tool uses a sub-agent to analyze the branch and intelligently
    select the best profile. Users don't need to understand the technical
    differences between profiles.

    Profiles:
    - pr_clean: Optimize for code review readability
    - bisect_friendly: Preserve granularity for debugging with git bisect
    - chronology: Keep original order for historical investigation
    - minimal: Fewest changes possible (conservative)
    - aggressive: Maximum squashing (for messy AI branches)
    """
    logger.info(f"Selecting profile for branch {request.branch}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        # Get context for sub-agent
        profile_context = await client.get_profile_context(
            SelectProfileRequest(
                repo_path=request.repo_path,
                branch=request.branch,
                base_branch=request.base_branch,
                context=request.context,
            )
        )

        # Build prompt for sub-agent
        prompt = f"""Analyze this branch and select the best rebase profile:

Branch Statistics:
- Total commits: {profile_context['total_commits']}
- Clusters: {profile_context['cluster_count']}
- Fixup commits: {profile_context['fixup_count']}
- Revert commits: {profile_context['revert_count']}
- WIP/checkpoint commits: {profile_context['wip_count']}
- Mixed-purpose clusters: {profile_context['mixed_purpose_clusters']}
- High-risk files: {profile_context['high_risk_files']}
- Overall risk: {profile_context['overall_risk']}

User context: {profile_context['user_context'] or 'None provided'}

Select the optimal profile and explain your reasoning.
"""

        # Run sub-agent
        result = await mcp_ctx.profile_agent.run(prompt)
        output: ProfileSelectionOutput = result.output

        logger.info(f"Profile selected: {output.selected_profile} (confidence: {output.confidence})")

        return SelectProfileResponse(
            selected_profile=output.selected_profile,
            reasoning=output.reasoning,
            confidence=output.confidence,
            alternative=output.alternative,
        ).model_dump()

    except Exception as e:
        logger.error(f"Profile selection failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_plan_generate(
    request: FreebasePlanGenerateRequest, ctx: Context
) -> dict[str, Any]:
    """Generate a machine-readable rebase plan.

    This creates a detailed plan for rebasing the branch, including:
    - Operations for each commit (pick, squash, fixup, drop)
    - Expected conflict hotspots
    - Questions requiring user input
    - Git rebase-todo content

    If profile_override is not provided, the profile is auto-selected.
    """
    logger.info(f"Generating plan for branch {request.branch}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        # Get profile (auto-select if not overridden)
        profile: Profile = request.profile_override
        if not profile:
            # Auto-select profile
            profile_result = await freebase_select_profile(
                FreebaseSelectProfileRequest(
                    repo_path=request.repo_path,
                    branch=request.branch,
                    base_branch=request.base_branch,
                ),
                ctx,
            )
            if "error" in profile_result:
                profile = "pr_clean"  # Fallback
            else:
                profile = profile_result["selected_profile"]

        # Convert user decisions
        user_decisions = []
        if request.user_decisions:
            for d in request.user_decisions:
                user_decisions.append(
                    UserDecision(
                        question_id=d.get("question_id", ""),
                        selected_option=d.get("selected_option", ""),
                        custom_value=d.get("custom_value"),
                    )
                )

        result = await client.generate_plan(
            PlanGenerateRequest(
                repo_path=request.repo_path,
                branch=request.branch,
                base_branch=request.base_branch,
                auto_squash_fixups=request.auto_squash_fixups,
                isolate_formatting=request.isolate_formatting,
                drop_reverted=request.drop_reverted,
                user_decisions=user_decisions,
            ),
            profile=profile,
        )

        logger.info(
            f"Plan generated: {result.original_commits} -> {result.planned_commits} commits"
        )

        return result.model_dump()

    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_plan_preview(
    request: FreebasePlanPreviewRequest, ctx: Context
) -> dict[str, Any]:
    """Generate a human-readable preview of a rebase plan.

    This provides a summary of what the plan will do, including:
    - Before/after commit counts
    - Cluster summaries
    - Number of decisions needed
    - Conflict hotspots
    - Risk level
    """
    logger.info(f"Previewing plan for {request.plan_id or request.branch}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        result = await client.preview_plan(
            PlanPreviewRequest(
                repo_path=request.repo_path,
                plan_id=request.plan_id,
                branch=request.branch,
                base_branch=request.base_branch,
            )
        )

        return result.model_dump()

    except Exception as e:
        logger.error(f"Plan preview failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_execute_step(
    request: FreebaseExecuteStepRequest, ctx: Context
) -> dict[str, Any]:
    """Execute the next step in a rebase plan.

    All operations happen in a TEMPORARY WORKTREE for safety.
    Your main working tree is never modified during dangerous operations.

    Returns:
    - Step completion status
    - Conflict details if a conflict occurred
    - Checkpoint ID for rollback
    - Preview of next step
    - Path to temporary worktree
    """
    logger.info(f"Executing step for plan {request.plan_id}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        # Convert conflict resolution if provided
        conflict_resolution = None
        if request.conflict_resolution:
            conflict_resolution = ConflictResolution(
                strategy_id=request.conflict_resolution.get("strategy_id", ""),
                file_resolutions=request.conflict_resolution.get("file_resolutions"),
                custom_message=request.conflict_resolution.get("custom_message"),
            )

        result = await client.execute_step(
            ExecuteStepRequest(
                repo_path=request.repo_path,
                plan_id=request.plan_id,
                step_index=request.step_index,
                conflict_resolution=conflict_resolution,
            )
        )

        status_msg = f"Step {result.step_completed}/{result.total_steps}: {result.status}"
        logger.info(status_msg)
        await ctx.info(status_msg)

        return result.model_dump()

    except Exception as e:
        logger.error(f"Step execution failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_conflict_analyze(
    request: FreebaseConflictAnalyzeRequest, ctx: Context
) -> dict[str, Any]:
    """Analyze a conflict and propose resolution strategies.

    This tool examines conflicted files and provides:
    - Conflict tier (mechanical, structural, or intent)
    - Explanation of why the conflict occurred
    - Resolution strategies with risk levels
    - Recommended strategy

    Conflict Tiers:
    - mechanical: Simple whitespace/formatting conflicts
    - structural: Code structure conflicts (method moves, etc.)
    - intent: Semantic conflicts requiring human judgment
    """
    logger.info(f"Analyzing conflict in {request.repo_path}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        result = await client.analyze_conflict(
            ConflictAnalyzeRequest(
                repo_path=request.repo_path,
                file_path=request.file_path,
                include_context=request.include_context,
                context_lines=request.context_lines,
            )
        )

        logger.info(f"Conflict analysis: {result.conflict_tier} tier")

        return result.model_dump()

    except Exception as e:
        logger.error(f"Conflict analysis failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_abort_to_checkpoint(
    request: FreebaseAbortRequest, ctx: Context
) -> dict[str, Any]:
    """Restore repository to a safe checkpoint state.

    This is the "panic button" - it will:
    - Abort any in-progress rebase
    - Clean up the temporary worktree
    - Reset the branch to the backup state

    Use this when something goes wrong and you need to start over.
    """
    logger.info(f"Aborting to checkpoint in {request.repo_path}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        result = await client.abort_to_checkpoint(
            AbortToCheckpointRequest(
                repo_path=request.repo_path,
                checkpoint_id=request.checkpoint_id,
                cleanup_worktree=request.cleanup_worktree,
            )
        )

        logger.info(f"Restored to checkpoint {result.restored_to}")
        await ctx.info(f"Restored to checkpoint: {result.restored_to}")

        return result.model_dump()

    except Exception as e:
        logger.error(f"Abort failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def freebase_range_diff_report(
    request: FreebaseRangeDiffRequest, ctx: Context
) -> dict[str, Any]:
    """Show before/after equivalence to verify rebase correctness.

    This compares the original commit range with the rebased range to verify:
    - Whether the ranges are semantically equivalent
    - Which commits were added, removed, or modified
    - How commits map between the two ranges

    Use this after completing a rebase to verify the result is correct.
    """
    logger.info(f"Generating range-diff report for {request.original_range}")

    try:
        mcp_ctx: MCPContext = ctx.request_context.lifespan_context
        client = mcp_ctx.get_client(request.repo_path)

        result = await client.range_diff_report(
            RangeDiffRequest(
                repo_path=request.repo_path,
                original_range=request.original_range,
                rebased_range=request.rebased_range,
                include_diffs=request.include_diffs,
            )
        )

        logger.info(f"Range-diff: {result.summary}")

        return result.model_dump()

    except Exception as e:
        logger.error(f"Range-diff failed: {e}")
        return {"error": str(e)}


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    mcp.run()


if __name__ == "__main__":
    main()
