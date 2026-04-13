"""Pydantic models for Freebase git history surgeon."""

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums and Type Aliases
# =============================================================================

CommitTag = Literal[
    "likely_fixup",
    "likely_revert",
    "formatting_only",
    "test_only",
    "docs_only",
    "dependency_update",
    "generated_code",
    "checkpoint",
    "wip",
    "mixed_purpose",
]

ClusterAction = Literal[
    "keep_separate",
    "squash_all",
    "squash_to_n",
    "reorder",
    "drop_all",
    "needs_split",
]

RebaseOp = Literal["pick", "reword", "squash", "fixup", "drop", "edit"]

ConflictTier = Literal["mechanical", "structural", "intent"]

CheckpointState = Literal["clean", "in_progress", "conflict", "completed"]

QuestionType = Literal[
    "squash_decision",
    "split_decision",
    "order_decision",
    "conflict_resolution",
    "drop_decision",
    "profile_choice",
]

Profile = Literal[
    "pr_clean",
    "bisect_friendly",
    "chronology",
    "minimal",
    "aggressive",
]

SessionStatus = Literal[
    "idle",
    "analyzing",
    "planning",
    "executing",
    "conflict",
    "completed",
    "aborted",
]

RiskLevel = Literal["low", "medium", "high"]

StepStatus = Literal["success", "conflict", "error"]


# =============================================================================
# File and Diff Models
# =============================================================================


class FileChange(BaseModel):
    """A file changed in a commit."""

    path: str = Field(description="File path relative to repo root")
    status: Literal["added", "modified", "deleted", "renamed", "copied"] = Field(
        description="Type of change"
    )
    old_path: Optional[str] = Field(default=None, description="Original path if renamed")
    insertions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines removed")
    is_binary: bool = Field(default=False, description="Whether file is binary")


class ConflictMarker(BaseModel):
    """A conflict marker location within a file."""

    start_line: int = Field(description="Line number where conflict starts")
    end_line: int = Field(description="Line number where conflict ends")
    ours_lines: list[str] = Field(description="Lines from current branch")
    theirs_lines: list[str] = Field(description="Lines from incoming changes")
    base_lines: Optional[list[str]] = Field(default=None, description="Lines from common ancestor")


# =============================================================================
# Commit Models
# =============================================================================


class CommitNode(BaseModel):
    """Detailed information about a single commit."""

    sha: str = Field(description="Full commit SHA")
    short_sha: str = Field(description="Short commit SHA (7 chars)")
    message: str = Field(description="Full commit message")
    subject: str = Field(description="First line of commit message")
    body: Optional[str] = Field(default=None, description="Rest of commit message")
    author: str = Field(description="Author name")
    author_email: str = Field(description="Author email")
    timestamp: int = Field(description="Unix timestamp of commit")
    parents: list[str] = Field(default_factory=list, description="Parent commit SHAs")
    files: list[FileChange] = Field(default_factory=list, description="Files changed")
    insertions: int = Field(default=0, description="Total lines added")
    deletions: int = Field(default=0, description="Total lines removed")
    tags: list[CommitTag] = Field(default_factory=list, description="Auto-detected commit tags")
    cluster_id: Optional[str] = Field(default=None, description="Assigned cluster ID")
    depends_on: list[str] = Field(default_factory=list, description="SHAs this commit depends on")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk score 0-1")
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Ambiguity score 0-1")
    conflict_hotspot_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Conflict likelihood 0-1"
    )
    embedding: Optional[list[float]] = Field(default=None, description="Message embedding vector")


class CommitCluster(BaseModel):
    """A group of related commits."""

    cluster_id: str = Field(description="Unique cluster identifier")
    label: str = Field(description="Human-readable cluster label")
    commits: list[str] = Field(description="Commit SHAs in this cluster")
    suggested_action: ClusterAction = Field(description="Recommended action for this cluster")
    target_commits: Optional[int] = Field(
        default=None, description="Target commit count for squash_to_n"
    )
    reasoning: list[str] = Field(default_factory=list, description="Why this action is suggested")
    needs_user_input: bool = Field(default=False, description="Whether user decision is needed")
    question: Optional[str] = Field(default=None, description="Question for user if needed")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in suggestion")


# =============================================================================
# Analysis Models
# =============================================================================


class RiskSummary(BaseModel):
    """Overall risk assessment for a branch."""

    overall_risk: RiskLevel = Field(description="Overall risk level")
    conflict_probability: float = Field(ge=0.0, le=1.0, description="Probability of conflicts")
    high_risk_files: list[str] = Field(default_factory=list, description="Files likely to conflict")
    risk_factors: list[str] = Field(default_factory=list, description="Contributing risk factors")


class SuggestedAction(BaseModel):
    """An auto-applicable cleanup action."""

    action_id: str = Field(description="Unique action identifier")
    action_type: str = Field(description="Type of action")
    description: str = Field(description="Human-readable description")
    commits_affected: list[str] = Field(description="Commit SHAs affected")
    auto_applicable: bool = Field(default=True, description="Can be applied automatically")
    risk: RiskLevel = Field(default="low", description="Risk level of this action")


class ConflictHotspot(BaseModel):
    """A predicted conflict point."""

    file_path: str = Field(description="File likely to conflict")
    probability: float = Field(ge=0.0, le=1.0, description="Conflict probability")
    commits_involved: list[str] = Field(description="Commits touching this file")
    reason: str = Field(description="Why conflict is expected")


# =============================================================================
# Plan Models
# =============================================================================


class RebaseOperation(BaseModel):
    """A single operation in a rebase plan."""

    operation: RebaseOp = Field(description="Git rebase operation")
    sha: str = Field(description="Commit SHA")
    message: Optional[str] = Field(default=None, description="New message for reword")
    cluster_id: Optional[str] = Field(default=None, description="Associated cluster")
    step_index: int = Field(description="Step number in the plan")


class ClusterSummary(BaseModel):
    """Summary of a cluster for preview."""

    cluster_id: str = Field(description="Cluster identifier")
    label: str = Field(description="Cluster label")
    commit_count: int = Field(description="Number of commits")
    action: ClusterAction = Field(description="Planned action")
    result_count: int = Field(description="Resulting commit count")
    description: str = Field(description="Human-readable description")


# =============================================================================
# Conflict Models
# =============================================================================


class ConflictCase(BaseModel):
    """Details of a rebase conflict."""

    files: list[str] = Field(description="Files with conflicts")
    commit_sha: str = Field(description="Commit being applied")
    commit_message: str = Field(description="Commit message")
    tier: ConflictTier = Field(description="Conflict complexity tier")
    ours_content: dict[str, str] = Field(
        default_factory=dict, description="File -> content from current branch"
    )
    theirs_content: dict[str, str] = Field(
        default_factory=dict, description="File -> content from incoming"
    )
    base_content: dict[str, str] = Field(
        default_factory=dict, description="File -> content from base"
    )
    markers: dict[str, list[ConflictMarker]] = Field(
        default_factory=dict, description="File -> conflict markers"
    )


class ConflictedFile(BaseModel):
    """A file with conflicts."""

    path: str = Field(description="File path")
    conflict_count: int = Field(description="Number of conflict regions")
    markers: list[ConflictMarker] = Field(description="Conflict markers")
    ours_content: str = Field(description="Full content from current branch")
    theirs_content: str = Field(description="Full content from incoming")
    base_content: Optional[str] = Field(default=None, description="Full content from base")


class ResolutionStrategy(BaseModel):
    """A proposed conflict resolution strategy."""

    strategy_id: str = Field(description="Unique strategy identifier")
    name: str = Field(description="Strategy name")
    description: str = Field(description="What this strategy does")
    result_preview: Optional[str] = Field(default=None, description="Preview of resolved content")
    risk: RiskLevel = Field(default="low", description="Risk level")
    auto_applicable: bool = Field(default=False, description="Can be applied automatically")


class ConflictResolution(BaseModel):
    """User's chosen conflict resolution."""

    strategy_id: str = Field(description="Chosen strategy ID")
    file_resolutions: Optional[dict[str, str]] = Field(
        default=None, description="File -> resolved content for manual resolution"
    )
    custom_message: Optional[str] = Field(default=None, description="Custom commit message")


# =============================================================================
# Checkpoint and State Models
# =============================================================================


class ExecutionCheckpoint(BaseModel):
    """A checkpoint for rollback during execution."""

    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    timestamp: int = Field(description="Unix timestamp")
    plan_id: str = Field(description="Associated plan ID")
    step_index: int = Field(description="Step index at checkpoint")
    branch_sha: str = Field(description="Branch SHA at checkpoint")
    backup_ref: str = Field(description="Backup branch reference")
    worktree_path: Optional[str] = Field(default=None, description="Worktree path if active")
    state: CheckpointState = Field(description="Checkpoint state")


class SessionState(BaseModel):
    """Persisted session state for crash recovery."""

    session_id: str = Field(description="Unique session identifier")
    started_at: datetime = Field(description="Session start time")
    repo_path: str = Field(description="Repository path")
    branch: str = Field(description="Branch being operated on")
    base_branch: str = Field(default="main", description="Base branch")
    current_plan_id: Optional[str] = Field(default=None, description="Active plan ID")
    current_step: int = Field(default=0, description="Current step index")
    worktree_path: Optional[str] = Field(default=None, description="Temporary worktree path")
    backup_ref: Optional[str] = Field(default=None, description="Backup branch reference")
    profile: Optional[Profile] = Field(default=None, description="Selected rebase profile")
    status: SessionStatus = Field(default="idle", description="Session status")
    last_error: Optional[str] = Field(default=None, description="Last error message")


class OperationLog(BaseModel):
    """A single operation log entry."""

    timestamp: datetime = Field(description="Operation timestamp")
    operation: str = Field(description="Operation name")
    details: dict[str, Any] = Field(default_factory=dict, description="Operation details")
    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# =============================================================================
# Question and Decision Models
# =============================================================================


class QuestionOption(BaseModel):
    """An option for a question."""

    option_id: str = Field(description="Unique option identifier")
    label: str = Field(description="Human-readable label")
    description: str = Field(description="What this option does")
    preview: Optional[str] = Field(default=None, description="Preview of result")


class Question(BaseModel):
    """A question requiring user input."""

    question_id: str = Field(description="Unique question identifier")
    question_type: QuestionType = Field(description="Type of question")
    question_text: str = Field(description="The question to ask")
    context: str = Field(description="Context for the question")
    options: list[QuestionOption] = Field(description="Available options")
    default_option: Optional[str] = Field(default=None, description="Default option ID")
    cluster_id: Optional[str] = Field(default=None, description="Associated cluster")
    commit_shas: list[str] = Field(default_factory=list, description="Related commits")


class UserDecision(BaseModel):
    """User's answer to a question."""

    question_id: str = Field(description="Question being answered")
    selected_option: str = Field(description="Selected option ID")
    custom_value: Optional[str] = Field(default=None, description="Custom value if applicable")


# =============================================================================
# Range Diff Models
# =============================================================================


class CommitMapping(BaseModel):
    """How an original commit maps to rebased commits."""

    original_sha: str = Field(description="Original commit SHA")
    rebased_sha: Optional[str] = Field(default=None, description="Rebased commit SHA")
    status: Literal["identical", "modified", "dropped", "split", "squashed"] = Field(
        description="Mapping status"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ModifiedCommit(BaseModel):
    """A commit that was modified during rebase."""

    original_sha: str = Field(description="Original commit SHA")
    rebased_sha: str = Field(description="Rebased commit SHA")
    message_changed: bool = Field(description="Whether message changed")
    content_changed: bool = Field(description="Whether content changed")
    diff_summary: str = Field(description="Summary of changes")


# =============================================================================
# Request Models
# =============================================================================


class BranchAnalyzeRequest(BaseModel):
    """Request for git_branch_analyze tool."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to analyze")
    base_branch: str = Field(default="main", description="Base branch")
    include_diffs: bool = Field(default=False, description="Include full diff content")
    cluster_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Similarity threshold for clustering"
    )


class SelectProfileRequest(BaseModel):
    """Request for git_select_profile tool."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to analyze")
    base_branch: str = Field(default="main", description="Base branch")
    context: Optional[str] = Field(
        default=None, description="User context (e.g., 'preparing for PR')"
    )


class PlanGenerateRequest(BaseModel):
    """Request for git_rebase_plan_generate tool."""

    repo_path: str = Field(description="Path to git repository")
    branch: str = Field(description="Branch to rebase")
    base_branch: str = Field(default="main", description="Target base branch")
    profile_override: Optional[Profile] = Field(
        default=None, description="Override auto-selected profile"
    )
    auto_squash_fixups: bool = Field(default=True, description="Auto-squash obvious fixups")
    isolate_formatting: bool = Field(default=True, description="Keep formatting commits separate")
    drop_reverted: bool = Field(default=True, description="Drop commits neutralized by reverts")
    user_decisions: list[UserDecision] = Field(
        default_factory=list, description="Answers to previous questions"
    )


class PlanPreviewRequest(BaseModel):
    """Request for git_rebase_plan_preview tool."""

    repo_path: str = Field(description="Path to git repository")
    plan_id: Optional[str] = Field(default=None, description="Existing plan ID to preview")
    branch: Optional[str] = Field(default=None, description="Branch if generating new preview")
    base_branch: str = Field(default="main", description="Base branch")


class ExecuteStepRequest(BaseModel):
    """Request for git_rebase_execute_step tool."""

    repo_path: str = Field(description="Path to git repository")
    plan_id: str = Field(description="Plan to execute")
    step_index: Optional[int] = Field(default=None, description="Specific step (default: next)")
    conflict_resolution: Optional[ConflictResolution] = Field(
        default=None, description="Resolution for current conflict"
    )


class ConflictAnalyzeRequest(BaseModel):
    """Request for git_conflict_analyze tool."""

    repo_path: str = Field(description="Path to git repository")
    file_path: Optional[str] = Field(default=None, description="Specific file to analyze")
    include_context: bool = Field(default=True, description="Include surrounding code")
    context_lines: int = Field(default=10, description="Lines of context")


class AbortToCheckpointRequest(BaseModel):
    """Request for git_rebase_abort_to_checkpoint tool."""

    repo_path: str = Field(description="Path to git repository")
    checkpoint_id: Optional[str] = Field(default=None, description="Specific checkpoint")
    cleanup_worktree: bool = Field(default=True, description="Remove temp worktree")


class RangeDiffRequest(BaseModel):
    """Request for git_range_diff_report tool."""

    repo_path: str = Field(description="Path to git repository")
    original_range: str = Field(description="Original commit range (base..head)")
    rebased_range: str = Field(description="Rebased commit range")
    include_diffs: bool = Field(default=False, description="Include actual diff content")


# =============================================================================
# Response Models
# =============================================================================


class BranchAnalyzeResponse(BaseModel):
    """Response from git_branch_analyze tool."""

    merge_base: str = Field(description="Common ancestor SHA")
    total_commits: int = Field(description="Number of commits in range")
    commits: list[CommitNode] = Field(description="Detailed commit information")
    clusters: list[CommitCluster] = Field(description="Grouped commits with suggestions")
    risk_summary: RiskSummary = Field(description="Overall risk assessment")
    suggested_actions: list[SuggestedAction] = Field(description="Auto-applicable cleanups")


class SelectProfileResponse(BaseModel):
    """Response from git_select_profile tool."""

    selected_profile: Profile = Field(description="Auto-selected profile")
    reasoning: str = Field(description="Explanation of why this profile was chosen")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")
    alternative: Optional[Profile] = Field(
        default=None, description="Second-best profile if confidence is low"
    )


class PlanGenerateResponse(BaseModel):
    """Response from git_rebase_plan_generate tool."""

    plan_id: str = Field(description="Unique identifier for this plan")
    original_commits: int = Field(description="Starting commit count")
    planned_commits: int = Field(description="Resulting commit count")
    operations: list[RebaseOperation] = Field(description="Ordered rebase operations")
    expected_conflicts: list[ConflictHotspot] = Field(description="Predicted conflict points")
    questions: list[Question] = Field(description="Items needing user input")
    todo_content: str = Field(description="Git rebase todo file content")
    profile_used: Profile = Field(description="Profile used for this plan")


class PlanPreviewResponse(BaseModel):
    """Response from git_rebase_plan_preview tool."""

    summary: str = Field(description="Human-readable summary")
    before_count: int = Field(description="Original commit count")
    after_count: int = Field(description="Planned commit count")
    clusters_summary: list[ClusterSummary] = Field(description="Cluster descriptions")
    questions_count: int = Field(description="Number of items needing input")
    conflict_hotspots: list[str] = Field(description="Files likely to conflict")
    estimated_risk: RiskLevel = Field(description="Overall risk level")


class ExecuteStepResponse(BaseModel):
    """Response from git_rebase_execute_step tool."""

    step_completed: int = Field(description="Step index completed")
    total_steps: int = Field(description="Total steps in plan")
    status: StepStatus = Field(description="Step result")
    conflict: Optional[ConflictCase] = Field(
        default=None, description="Conflict details if status is conflict"
    )
    checkpoint_id: str = Field(description="Checkpoint for rollback")
    next_step_preview: Optional[str] = Field(default=None, description="Description of next step")
    worktree_path: str = Field(description="Path to temporary worktree")


class ConflictAnalyzeResponse(BaseModel):
    """Response from git_conflict_analyze tool."""

    conflicted_files: list[ConflictedFile] = Field(description="Files with conflicts")
    conflict_tier: ConflictTier = Field(description="Conflict complexity")
    explanation: str = Field(description="Why the conflict occurred")
    ours_description: str = Field(description="What the branch version does")
    theirs_description: str = Field(description="What the base version does")
    resolution_strategies: list[ResolutionStrategy] = Field(description="Proposed resolutions")
    recommended_strategy: Optional[str] = Field(
        default=None, description="ID of recommended strategy"
    )
    question: Optional[str] = Field(default=None, description="Question for user if intent-level")


class AbortToCheckpointResponse(BaseModel):
    """Response from git_rebase_abort_to_checkpoint tool."""

    restored_to: str = Field(description="Checkpoint restored")
    branch_state: str = Field(description="Current branch SHA")
    worktree_cleaned: bool = Field(description="Whether worktree was removed")
    backup_ref: str = Field(description="Backup branch name")


class RangeDiffResponse(BaseModel):
    """Response from git_range_diff_report tool."""

    equivalent: bool = Field(description="Whether ranges are semantically equivalent")
    original_count: int = Field(description="Commits in original range")
    rebased_count: int = Field(description="Commits in rebased range")
    mappings: list[CommitMapping] = Field(description="How commits correspond")
    added_commits: list[str] = Field(description="Commits only in rebased")
    removed_commits: list[str] = Field(description="Commits only in original")
    modified_commits: list[ModifiedCommit] = Field(description="Commits with changes")
    summary: str = Field(description="Human-readable summary")
