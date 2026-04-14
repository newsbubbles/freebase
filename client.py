"""Freebase Git History Client - Main API for git history surgery."""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from clustering import ClusterBuilder, EmbeddingClient, detect_commit_tags
from git_ops import (
    GitOps,
    GitError,
    parse_commit_log,
    parse_file_status,
    parse_numstat,
    parse_conflict_markers,
)
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
    CommitNode,
    CommitCluster,
    FileChange,
    RiskSummary,
    SuggestedAction,
    ConflictHotspot,
    RebaseOperation,
    ClusterSummary,
    ConflictCase,
    ConflictedFile,
    ConflictMarker,
    ResolutionStrategy,
    CommitMapping,
    ModifiedCommit,
    Question,
    QuestionOption,
    Profile,
)
from state import StateManager


class FreebaseClient:
    """Main client for Freebase git history operations."""

    def __init__(
        self,
        repo_path: str,
        openrouter_api_key: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.git = GitOps(str(self.repo_path))
        self.state = StateManager(str(self.repo_path))

        # Embedding client for commit message similarity (uses OpenRouter)
        self.embedding_client = EmbeddingClient(
            api_key=openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        )

        # Clustering builder
        self.cluster_builder = ClusterBuilder(
            embedding_client=self.embedding_client,
            similarity_threshold=0.6,
        )

    # =========================================================================
    # Tool 1: git_branch_analyze
    # =========================================================================

    async def analyze_branch(self, request: BranchAnalyzeRequest) -> BranchAnalyzeResponse:
        """Analyze a branch and build a structured model of the commit stack."""
        # Get merge base
        merge_base = await self.git.get_merge_base(request.base_branch, request.branch)

        # Get commit log
        range_spec = f"{merge_base}..{request.branch}"
        raw_log = await self.git.get_commit_log(range_spec)
        parsed_commits = parse_commit_log(raw_log)

        if not parsed_commits:
            return BranchAnalyzeResponse(
                merge_base=merge_base,
                total_commits=0,
                commits=[],
                clusters=[],
                risk_summary=RiskSummary(
                    overall_risk="low",
                    conflict_probability=0.0,
                    high_risk_files=[],
                    risk_factors=[],
                ),
                suggested_actions=[],
            )

        # Build CommitNode objects with file info
        commits: list[CommitNode] = []
        for pc in parsed_commits:
            # Get file changes
            file_status = await self.git.get_commit_files(pc["sha"])
            numstat = await self.git.get_commit_numstat(pc["sha"])

            files_info = parse_file_status(file_status)
            stats_info = parse_numstat(numstat)

            # Merge file info with stats
            stats_by_path = {s["path"]: s for s in stats_info}
            files: list[FileChange] = []
            total_insertions = 0
            total_deletions = 0

            for fi in files_info:
                stats = stats_by_path.get(fi["path"], {})
                insertions = stats.get("insertions", 0)
                deletions = stats.get("deletions", 0)
                total_insertions += insertions
                total_deletions += deletions

                files.append(
                    FileChange(
                        path=fi["path"],
                        status=fi["status"],
                        old_path=fi.get("old_path"),
                        insertions=insertions,
                        deletions=deletions,
                        is_binary=stats.get("is_binary", False),
                    )
                )

            commit = CommitNode(
                sha=pc["sha"],
                short_sha=pc["short_sha"],
                message=pc["subject"] + ("\n" + pc["body"] if pc["body"] else ""),
                subject=pc["subject"],
                body=pc["body"],
                author=pc["author"],
                author_email=pc["author_email"],
                timestamp=pc["timestamp"],
                parents=pc["parents"],
                files=files,
                insertions=total_insertions,
                deletions=total_deletions,
            )

            # Detect tags
            commit.tags = detect_commit_tags(commit)
            commits.append(commit)

        # Build clusters
        self.cluster_builder.similarity_threshold = request.cluster_threshold
        clusters = await self.cluster_builder.build_clusters(commits)

        # Calculate risk summary
        risk_summary = self._calculate_risk_summary(commits, clusters)

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(commits, clusters)

        return BranchAnalyzeResponse(
            merge_base=merge_base,
            total_commits=len(commits),
            commits=commits,
            clusters=clusters,
            risk_summary=risk_summary,
            suggested_actions=suggested_actions,
        )

    def _calculate_risk_summary(
        self, commits: list[CommitNode], clusters: list[CommitCluster]
    ) -> RiskSummary:
        """Calculate overall risk assessment."""
        risk_factors = []
        high_risk_files = []

        # Count file touches
        file_touch_count: dict[str, int] = {}
        for commit in commits:
            for f in commit.files:
                file_touch_count[f.path] = file_touch_count.get(f.path, 0) + 1

        # Files touched multiple times are conflict risks
        for path, count in file_touch_count.items():
            if count >= 3:
                high_risk_files.append(path)
                risk_factors.append(f"{path} touched {count} times")

        # Check for reverts
        revert_count = sum(1 for c in commits if "likely_revert" in c.tags)
        if revert_count > 0:
            risk_factors.append(f"{revert_count} revert commits detected")

        # Check for mixed-purpose clusters
        mixed_clusters = sum(1 for c in clusters if c.needs_user_input)
        if mixed_clusters > 0:
            risk_factors.append(f"{mixed_clusters} clusters need user decisions")

        # Calculate conflict probability
        conflict_prob = min(1.0, len(high_risk_files) * 0.15 + revert_count * 0.1)

        # Determine overall risk
        if conflict_prob > 0.6 or len(high_risk_files) > 5:
            overall_risk = "high"
        elif conflict_prob > 0.3 or len(high_risk_files) > 2:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return RiskSummary(
            overall_risk=overall_risk,
            conflict_probability=conflict_prob,
            high_risk_files=high_risk_files[:10],  # Limit to top 10
            risk_factors=risk_factors[:10],
        )

    def _generate_suggested_actions(
        self, commits: list[CommitNode], clusters: list[CommitCluster]
    ) -> list[SuggestedAction]:
        """Generate auto-applicable cleanup actions."""
        actions = []

        # Suggest squashing fixup clusters
        for cluster in clusters:
            if cluster.suggested_action == "squash_all" and not cluster.needs_user_input:
                actions.append(
                    SuggestedAction(
                        action_id=f"squash_{cluster.cluster_id}",
                        action_type="squash",
                        description=f"Squash {len(cluster.commits)} commits in '{cluster.label}'",
                        commits_affected=cluster.commits,
                        auto_applicable=True,
                        risk="low",
                    )
                )

        return actions

    # =========================================================================
    # Tool 2: git_select_profile (Sub-agent will call this)
    # =========================================================================

    async def get_profile_context(self, request: SelectProfileRequest) -> dict:
        """Get context for profile selection sub-agent.

        This returns the data needed for the sub-agent to make a decision.
        The actual selection is done by the sub-agent in the MCP server.
        """
        # Analyze the branch first
        analysis = await self.analyze_branch(
            BranchAnalyzeRequest(
                repo_path=request.repo_path,
                branch=request.branch,
                base_branch=request.base_branch,
            )
        )

        # Gather statistics for sub-agent
        fixup_count = sum(1 for c in analysis.commits if "likely_fixup" in c.tags)
        revert_count = sum(1 for c in analysis.commits if "likely_revert" in c.tags)
        wip_count = sum(1 for c in analysis.commits if "wip" in c.tags or "checkpoint" in c.tags)
        mixed_count = sum(1 for c in analysis.clusters if c.needs_user_input)

        return {
            "total_commits": analysis.total_commits,
            "cluster_count": len(analysis.clusters),
            "fixup_count": fixup_count,
            "revert_count": revert_count,
            "wip_count": wip_count,
            "mixed_purpose_clusters": mixed_count,
            "high_risk_files": len(analysis.risk_summary.high_risk_files),
            "overall_risk": analysis.risk_summary.overall_risk,
            "user_context": request.context,
        }

    # =========================================================================
    # Tool 3: git_rebase_plan_generate
    # =========================================================================

    async def generate_plan(
        self, request: PlanGenerateRequest, profile: Profile
    ) -> PlanGenerateResponse:
        """Generate a rebase plan based on analysis."""
        # Analyze branch first
        analysis = await self.analyze_branch(
            BranchAnalyzeRequest(
                repo_path=request.repo_path,
                branch=request.branch,
                base_branch=request.base_branch,
            )
        )

        # Generate plan ID
        plan_id = str(uuid.uuid4())[:8]

        # Build operations based on clusters and profile
        operations: list[RebaseOperation] = []
        questions: list[Question] = []
        step_index = 0

        for cluster in analysis.clusters:
            cluster_ops, cluster_questions = self._plan_cluster_operations(
                cluster,
                analysis.commits,
                profile,
                request,
                step_index,
            )
            operations.extend(cluster_ops)
            questions.extend(cluster_questions)
            step_index += len(cluster_ops)

        # Calculate expected conflicts
        expected_conflicts = self._predict_conflicts(analysis)

        # Generate todo content
        todo_content = self._generate_todo_content(operations, analysis.commits)

        # Calculate planned commit count
        planned_commits = sum(1 for op in operations if op.operation in ("pick", "reword"))

        response = PlanGenerateResponse(
            plan_id=plan_id,
            original_commits=analysis.total_commits,
            planned_commits=planned_commits,
            operations=operations,
            expected_conflicts=expected_conflicts,
            questions=questions,
            todo_content=todo_content,
            profile_used=profile,
        )

        # Save plan
        self.state.save_plan(response)

        return response

    def _plan_cluster_operations(
        self,
        cluster: CommitCluster,
        commits: list[CommitNode],
        profile: Profile,
        request: PlanGenerateRequest,
        start_index: int,
    ) -> tuple[list[RebaseOperation], list[Question]]:
        """Plan operations for a single cluster."""
        operations: list[RebaseOperation] = []
        questions: list[Question] = []

        # Get commits in this cluster
        cluster_commits = [c for c in commits if c.sha in cluster.commits]

        # Apply profile-specific logic
        action = cluster.suggested_action

        # Profile adjustments
        if profile == "minimal":
            # Minimal changes - only squash obvious fixups
            if action not in ("squash_all",) or cluster.confidence < 0.9:
                action = "keep_separate"
        elif profile == "aggressive":
            # Aggressive - squash everything possible
            if action in ("keep_separate", "squash_to_n"):
                action = "squash_all"
        elif profile == "bisect_friendly":
            # Preserve granularity for bisect
            if action == "squash_all" and len(cluster_commits) > 2:
                action = "squash_to_n"
                cluster.target_commits = max(2, len(cluster_commits) // 2)

        # Apply user decisions if any
        for decision in request.user_decisions:
            if decision.question_id.startswith(cluster.cluster_id):
                action = decision.selected_option

        # Generate operations based on action
        step = start_index

        if action == "keep_separate":
            for commit in cluster_commits:
                operations.append(
                    RebaseOperation(
                        operation="pick",
                        sha=commit.sha,
                        cluster_id=cluster.cluster_id,
                        step_index=step,
                    )
                )
                step += 1

        elif action == "squash_all":
            if cluster_commits:
                # First commit is pick, rest are fixup
                operations.append(
                    RebaseOperation(
                        operation="pick",
                        sha=cluster_commits[0].sha,
                        cluster_id=cluster.cluster_id,
                        step_index=step,
                    )
                )
                step += 1
                for commit in cluster_commits[1:]:
                    operations.append(
                        RebaseOperation(
                            operation="fixup",
                            sha=commit.sha,
                            cluster_id=cluster.cluster_id,
                            step_index=step,
                        )
                    )
                    step += 1

        elif action == "drop_all":
            for commit in cluster_commits:
                operations.append(
                    RebaseOperation(
                        operation="drop",
                        sha=commit.sha,
                        cluster_id=cluster.cluster_id,
                        step_index=step,
                    )
                )
                step += 1

        elif action == "squash_to_n":
            target = cluster.target_commits or 2
            # Group commits into target buckets
            bucket_size = max(1, len(cluster_commits) // target)
            for i, commit in enumerate(cluster_commits):
                if i % bucket_size == 0:
                    operations.append(
                        RebaseOperation(
                            operation="pick",
                            sha=commit.sha,
                            cluster_id=cluster.cluster_id,
                            step_index=step,
                        )
                    )
                else:
                    operations.append(
                        RebaseOperation(
                            operation="fixup",
                            sha=commit.sha,
                            cluster_id=cluster.cluster_id,
                            step_index=step,
                        )
                    )
                step += 1

        elif action == "needs_split":
            # Keep separate but ask user
            for commit in cluster_commits:
                operations.append(
                    RebaseOperation(
                        operation="pick",
                        sha=commit.sha,
                        cluster_id=cluster.cluster_id,
                        step_index=step,
                    )
                )
                step += 1

        # Add question if needed and not already answered
        if cluster.needs_user_input and cluster.question:
            already_answered = any(
                d.question_id.startswith(cluster.cluster_id)
                for d in request.user_decisions
            )
            if not already_answered:
                questions.append(
                    Question(
                        question_id=f"{cluster.cluster_id}_action",
                        question_type="squash_decision",
                        question_text=cluster.question,
                        context=f"Cluster: {cluster.label}\nCommits: {len(cluster.commits)}",
                        options=[
                            QuestionOption(
                                option_id="keep_separate",
                                label="Keep separate",
                                description="Keep all commits as-is",
                            ),
                            QuestionOption(
                                option_id="squash_all",
                                label="Squash all",
                                description="Combine into single commit",
                            ),
                            QuestionOption(
                                option_id="squash_to_n",
                                label="Squash to 2-3",
                                description="Combine into 2-3 logical commits",
                            ),
                        ],
                        default_option=action,
                        cluster_id=cluster.cluster_id,
                        commit_shas=cluster.commits,
                    )
                )

        return operations, questions

    def _predict_conflicts(self, analysis: BranchAnalyzeResponse) -> list[ConflictHotspot]:
        """Predict likely conflict points."""
        hotspots = []

        # Files touched multiple times
        file_commits: dict[str, list[str]] = {}
        for commit in analysis.commits:
            for f in commit.files:
                if f.path not in file_commits:
                    file_commits[f.path] = []
                file_commits[f.path].append(commit.sha)

        for path, commits in file_commits.items():
            if len(commits) >= 3:
                prob = min(0.9, 0.3 + (len(commits) - 3) * 0.15)
                hotspots.append(
                    ConflictHotspot(
                        file_path=path,
                        probability=prob,
                        commits_involved=commits,
                        reason=f"Modified in {len(commits)} commits",
                    )
                )

        # Sort by probability
        hotspots.sort(key=lambda h: h.probability, reverse=True)
        return hotspots[:10]

    def _generate_todo_content(
        self, operations: list[RebaseOperation], commits: list[CommitNode]
    ) -> str:
        """Generate git rebase-todo file content."""
        commit_map = {c.sha: c for c in commits}
        lines = []

        for op in operations:
            commit = commit_map.get(op.sha)
            subject = commit.subject[:50] if commit else "Unknown commit"
            short_sha = op.sha[:7]
            lines.append(f"{op.operation} {short_sha} {subject}")

        return "\n".join(lines)

    # =========================================================================
    # Tool 4: git_rebase_plan_preview
    # =========================================================================

    async def preview_plan(self, request: PlanPreviewRequest) -> PlanPreviewResponse:
        """Generate a human-readable preview of a rebase plan."""
        plan = None

        if request.plan_id:
            plan = self.state.load_plan(request.plan_id)
            if not plan:
                raise ValueError(f"Plan not found: {request.plan_id}")

        if not plan and request.branch:
            # Generate a new plan for preview
            plan = await self.generate_plan(
                PlanGenerateRequest(
                    repo_path=request.repo_path,
                    branch=request.branch,
                    base_branch=request.base_branch,
                ),
                profile="pr_clean",  # Default profile for preview
            )

        if not plan:
            raise ValueError("Either plan_id or branch must be provided")

        # Build cluster summaries
        cluster_summaries = []
        cluster_ops: dict[str, list[RebaseOperation]] = {}

        for op in plan.operations:
            if op.cluster_id:
                if op.cluster_id not in cluster_ops:
                    cluster_ops[op.cluster_id] = []
                cluster_ops[op.cluster_id].append(op)

        for cluster_id, ops in cluster_ops.items():
            pick_count = sum(1 for o in ops if o.operation == "pick")
            total_count = len(ops)

            # Determine action description
            if pick_count == total_count:
                action = "keep_separate"
                description = f"Keep {total_count} commits separate"
            elif pick_count == 1:
                action = "squash_all"
                description = f"Squash {total_count} commits into 1"
            else:
                action = "squash_to_n"
                description = f"Squash {total_count} commits into {pick_count}"

            cluster_summaries.append(
                ClusterSummary(
                    cluster_id=cluster_id,
                    label=cluster_id,
                    commit_count=total_count,
                    action=action,
                    result_count=pick_count,
                    description=description,
                )
            )

        # Build summary text
        summary_lines = [
            f"Rebase plan: {plan.original_commits} commits → {plan.planned_commits} commits",
            f"Profile: {plan.profile_used}",
            "",
            "Clusters:",
        ]
        for cs in cluster_summaries:
            summary_lines.append(f"  - {cs.description}")

        if plan.questions:
            summary_lines.append(f"\n{len(plan.questions)} decisions needed")

        # Determine risk
        conflict_files = [h.file_path for h in plan.expected_conflicts]
        if len(conflict_files) > 5:
            risk = "high"
        elif len(conflict_files) > 2:
            risk = "medium"
        else:
            risk = "low"

        return PlanPreviewResponse(
            summary="\n".join(summary_lines),
            before_count=plan.original_commits,
            after_count=plan.planned_commits,
            clusters_summary=cluster_summaries,
            questions_count=len(plan.questions),
            conflict_hotspots=conflict_files[:5],
            estimated_risk=risk,
        )

    # =========================================================================
    # Tool 5: git_rebase_execute_step
    # =========================================================================

    async def execute_step(self, request: ExecuteStepRequest) -> ExecuteStepResponse:
        """Execute the next step in a rebase plan."""
        # Load plan
        plan = self.state.load_plan(request.plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {request.plan_id}")

        # Load or create session
        session = self.state.load_session()
        if not session or session.current_plan_id != request.plan_id:
            # Start new execution
            session = self.state.create_session(
                branch=session.branch if session else "unknown",
                base_branch=session.base_branch if session else "main",
            )
            session = self.state.update_session(
                current_plan_id=request.plan_id,
                current_step=0,
                status="executing",
            )

            # Create worktree
            worktree_path = await self._setup_worktree(plan)
            session = self.state.update_session(worktree_path=str(worktree_path))

            # Create backup
            backup_ref = await self.git.create_backup_ref(session.branch)
            session = self.state.update_session(backup_ref=backup_ref)

        # Determine which step to execute
        step_index = request.step_index if request.step_index is not None else session.current_step

        if step_index >= len(plan.operations):
            # All steps complete
            return ExecuteStepResponse(
                step_completed=step_index - 1,
                total_steps=len(plan.operations),
                status="success",
                checkpoint_id=self.state.get_latest_checkpoint(request.plan_id).checkpoint_id,
                next_step_preview=None,
                worktree_path=session.worktree_path or "",
            )

        # Get worktree path
        worktree_path = Path(session.worktree_path) if session.worktree_path else None
        if not worktree_path or not worktree_path.exists():
            raise ValueError("Worktree not found. Session may be corrupted.")

        # Create checkpoint before execution
        current_sha = await self.git.get_head_sha()
        checkpoint = self.state.create_checkpoint(
            plan_id=request.plan_id,
            step_index=step_index,
            branch_sha=current_sha,
            backup_ref=session.backup_ref or "",
            worktree_path=str(worktree_path),
            state="in_progress",
        )

        # Check if we're in a conflict state
        if await self.git.is_rebase_in_progress(cwd=worktree_path):
            if request.conflict_resolution:
                # Apply resolution and continue
                await self._apply_conflict_resolution(
                    request.conflict_resolution, worktree_path
                )
                result = await self.git.rebase_continue(cwd=worktree_path)
                if not result.success:
                    # Still conflicts
                    conflict = await self._get_conflict_case(worktree_path)
                    return ExecuteStepResponse(
                        step_completed=step_index,
                        total_steps=len(plan.operations),
                        status="conflict",
                        conflict=conflict,
                        checkpoint_id=checkpoint.checkpoint_id,
                        next_step_preview=self._get_step_preview(plan, step_index + 1),
                        worktree_path=str(worktree_path),
                    )
            else:
                # Return conflict info
                conflict = await self._get_conflict_case(worktree_path)
                return ExecuteStepResponse(
                    step_completed=step_index,
                    total_steps=len(plan.operations),
                    status="conflict",
                    conflict=conflict,
                    checkpoint_id=checkpoint.checkpoint_id,
                    next_step_preview=None,
                    worktree_path=str(worktree_path),
                )

        # Execute the step (first step starts the rebase)
        if step_index == 0:
            # Start interactive rebase
            result = await self.git.rebase_interactive_start(
                onto=session.base_branch,
                todo_content=plan.todo_content,
                cwd=worktree_path,
            )
        else:
            # Continue rebase (should already be in progress)
            result = await self.git.rebase_continue(cwd=worktree_path)

        # Check result
        if not result.success and "CONFLICT" in result.stderr:
            conflict = await self._get_conflict_case(worktree_path)
            self.state.update_session(status="conflict")
            return ExecuteStepResponse(
                step_completed=step_index,
                total_steps=len(plan.operations),
                status="conflict",
                conflict=conflict,
                checkpoint_id=checkpoint.checkpoint_id,
                next_step_preview=None,
                worktree_path=str(worktree_path),
            )
        elif not result.success:
            self.state.log_operation(
                "execute_step",
                {"step": step_index, "error": result.stderr},
                success=False,
                error=result.stderr,
            )
            return ExecuteStepResponse(
                step_completed=step_index,
                total_steps=len(plan.operations),
                status="error",
                checkpoint_id=checkpoint.checkpoint_id,
                next_step_preview=None,
                worktree_path=str(worktree_path),
            )

        # Success - update session
        self.state.update_session(current_step=step_index + 1)
        self.state.log_operation(
            "execute_step",
            {"step": step_index},
            success=True,
        )

        # Update checkpoint state
        checkpoint.state = "clean"
        self.state.save_checkpoint(checkpoint)

        return ExecuteStepResponse(
            step_completed=step_index,
            total_steps=len(plan.operations),
            status="success",
            checkpoint_id=checkpoint.checkpoint_id,
            next_step_preview=self._get_step_preview(plan, step_index + 1),
            worktree_path=str(worktree_path),
        )

    async def _setup_worktree(self, plan: PlanGenerateResponse) -> Path:
        """Set up a temporary worktree for rebase operations."""
        timestamp = int(time.time())
        worktree_name = f"freebase-work-{timestamp}"
        worktree_path = self.repo_path.parent / worktree_name

        session = self.state.load_session()
        branch = session.branch if session else "HEAD"

        await self.git.worktree_add(
            path=str(worktree_path),
            branch=f"freebase-temp-{timestamp}",
            start_point=branch,
            create_branch=True,
        )

        return worktree_path

    async def _get_conflict_case(self, worktree_path: Path) -> ConflictCase:
        """Get details about current conflict."""
        conflicted_files = await self.git.get_conflicted_files(cwd=worktree_path)

        ours_content = {}
        theirs_content = {}
        base_content = {}
        markers = {}

        for file_path in conflicted_files:
            content = await self.git.get_conflict_content(file_path, cwd=worktree_path)
            ours_content[file_path] = content["ours"]
            theirs_content[file_path] = content["theirs"]
            base_content[file_path] = content["base"]

            # Get current file with markers
            current_content = await self.git.get_file_content(file_path, cwd=worktree_path)
            parsed_markers = parse_conflict_markers(current_content)
            markers[file_path] = [
                ConflictMarker(
                    start_line=m["start_line"],
                    end_line=m["end_line"],
                    ours_lines=m["ours_lines"],
                    theirs_lines=m["theirs_lines"],
                    base_lines=m.get("base_lines"),
                )
                for m in parsed_markers
            ]

        return ConflictCase(
            files=conflicted_files,
            commit_sha="",  # Would need to track this
            commit_message="",
            tier="structural",  # Default, would need analysis
            ours_content=ours_content,
            theirs_content=theirs_content,
            base_content=base_content,
            markers=markers,
        )

    async def _apply_conflict_resolution(
        self, resolution, worktree_path: Path
    ) -> None:
        """Apply a conflict resolution."""
        if resolution.file_resolutions:
            for file_path, content in resolution.file_resolutions.items():
                (worktree_path / file_path).write_text(content)
                await self.git.add(file_path, cwd=worktree_path)

    def _get_step_preview(self, plan: PlanGenerateResponse, step_index: int) -> Optional[str]:
        """Get a preview of the next step."""
        if step_index >= len(plan.operations):
            return None
        op = plan.operations[step_index]
        return f"{op.operation} {op.sha[:7]}"

    # =========================================================================
    # Tool 6: git_conflict_analyze
    # =========================================================================

    async def analyze_conflict(self, request: ConflictAnalyzeRequest) -> ConflictAnalyzeResponse:
        """Analyze a conflict and propose resolution strategies."""
        session = self.state.load_session()
        worktree_path = Path(session.worktree_path) if session and session.worktree_path else self.repo_path

        # Get conflicted files
        if request.file_path:
            conflicted_files = [request.file_path]
        else:
            conflicted_files = await self.git.get_conflicted_files(cwd=worktree_path)

        if not conflicted_files:
            raise ValueError("No conflicts found")

        # Analyze each file
        analyzed_files = []
        for file_path in conflicted_files:
            content = await self.git.get_conflict_content(file_path, cwd=worktree_path)
            current_content = await self.git.get_file_content(file_path, cwd=worktree_path)
            parsed_markers = parse_conflict_markers(current_content)

            analyzed_files.append(
                ConflictedFile(
                    path=file_path,
                    conflict_count=len(parsed_markers),
                    markers=[
                        ConflictMarker(
                            start_line=m["start_line"],
                            end_line=m["end_line"],
                            ours_lines=m["ours_lines"],
                            theirs_lines=m["theirs_lines"],
                            base_lines=m.get("base_lines"),
                        )
                        for m in parsed_markers
                    ],
                    ours_content=content["ours"],
                    theirs_content=content["theirs"],
                    base_content=content["base"],
                )
            )

        # Determine conflict tier
        tier = self._determine_conflict_tier(analyzed_files)

        # Generate resolution strategies
        strategies = self._generate_resolution_strategies(analyzed_files, tier)

        return ConflictAnalyzeResponse(
            conflicted_files=analyzed_files,
            conflict_tier=tier,
            explanation=self._generate_conflict_explanation(analyzed_files),
            ours_description="Changes from the branch being rebased",
            theirs_description="Changes from the base branch",
            resolution_strategies=strategies,
            recommended_strategy=strategies[0].strategy_id if strategies else None,
            question="How should this conflict be resolved?" if tier == "intent" else None,
        )

    def _determine_conflict_tier(self, files: list[ConflictedFile]) -> str:
        """Determine the complexity tier of conflicts."""
        total_markers = sum(f.conflict_count for f in files)

        if total_markers <= 2:
            # Check if it's just whitespace/formatting
            for f in files:
                for m in f.markers:
                    ours = "\n".join(m.ours_lines).strip()
                    theirs = "\n".join(m.theirs_lines).strip()
                    if ours.replace(" ", "") == theirs.replace(" ", ""):
                        return "mechanical"
            return "structural"

        return "intent"

    def _generate_resolution_strategies(
        self, files: list[ConflictedFile], tier: str
    ) -> list[ResolutionStrategy]:
        """Generate resolution strategies based on conflict analysis."""
        strategies = []

        # Always offer ours/theirs
        strategies.append(
            ResolutionStrategy(
                strategy_id="accept_ours",
                name="Accept ours",
                description="Keep the changes from the branch being rebased",
                risk="low" if tier == "mechanical" else "medium",
                auto_applicable=tier == "mechanical",
            )
        )

        strategies.append(
            ResolutionStrategy(
                strategy_id="accept_theirs",
                name="Accept theirs",
                description="Keep the changes from the base branch",
                risk="low" if tier == "mechanical" else "medium",
                auto_applicable=tier == "mechanical",
            )
        )

        if tier != "intent":
            strategies.append(
                ResolutionStrategy(
                    strategy_id="merge_both",
                    name="Merge both",
                    description="Combine both sets of changes",
                    risk="medium",
                    auto_applicable=False,
                )
            )

        strategies.append(
            ResolutionStrategy(
                strategy_id="manual",
                name="Manual resolution",
                description="Manually edit the conflicted files",
                risk="low",
                auto_applicable=False,
            )
        )

        return strategies

    def _generate_conflict_explanation(self, files: list[ConflictedFile]) -> str:
        """Generate a human-readable explanation of the conflict."""
        total_conflicts = sum(f.conflict_count for f in files)
        file_list = ", ".join(f.path for f in files[:3])

        if len(files) > 3:
            file_list += f" and {len(files) - 3} more"

        return f"{total_conflicts} conflict(s) in {len(files)} file(s): {file_list}"

    # =========================================================================
    # Tool 7: git_rebase_abort_to_checkpoint
    # =========================================================================

    async def abort_to_checkpoint(
        self, request: AbortToCheckpointRequest
    ) -> AbortToCheckpointResponse:
        """Restore repository to a safe checkpoint state."""
        # Get checkpoint
        if request.checkpoint_id:
            checkpoint = self.state.load_checkpoint(request.checkpoint_id)
        else:
            checkpoint = self.state.get_latest_checkpoint()

        if not checkpoint:
            raise ValueError("No checkpoint found")

        session = self.state.load_session()
        worktree_path = Path(session.worktree_path) if session and session.worktree_path else None

        # Abort any in-progress rebase
        if worktree_path and worktree_path.exists():
            if await self.git.is_rebase_in_progress(cwd=worktree_path):
                await self.git.rebase_abort(cwd=worktree_path)

        # Clean up worktree if requested
        worktree_cleaned = False
        if request.cleanup_worktree and worktree_path and worktree_path.exists():
            try:
                await self.git.worktree_remove(str(worktree_path), force=True)
                worktree_cleaned = True
            except GitError:
                pass  # May already be removed

        # Reset main branch to backup
        await self.git.reset_hard(f"refs/freebase/{checkpoint.backup_ref}")

        # Update session
        self.state.update_session(
            status="aborted",
            current_step=0,
            worktree_path=None,
        )

        # Log operation
        self.state.log_operation(
            "abort_to_checkpoint",
            {"checkpoint_id": checkpoint.checkpoint_id},
            success=True,
        )

        # Get current branch state
        branch_state = await self.git.get_head_sha()

        return AbortToCheckpointResponse(
            restored_to=checkpoint.checkpoint_id,
            branch_state=branch_state,
            worktree_cleaned=worktree_cleaned,
            backup_ref=checkpoint.backup_ref,
        )

    # =========================================================================
    # Tool 8: git_range_diff_report
    # =========================================================================

    async def range_diff_report(self, request: RangeDiffRequest) -> RangeDiffResponse:
        """Show before/after equivalence to verify rebase correctness."""
        # Parse ranges
        orig_parts = request.original_range.split("..")
        rebased_parts = request.rebased_range.split("..")

        if len(orig_parts) != 2 or len(rebased_parts) != 2:
            raise ValueError("Ranges must be in format 'base..head'")

        # Get range-diff output
        range_diff = await self.git.get_range_diff(
            orig_parts[0], orig_parts[1],
            rebased_parts[0], rebased_parts[1],
        )

        # Get commit counts
        original_count = await self.git.get_commit_count(request.original_range)
        rebased_count = await self.git.get_commit_count(request.rebased_range)

        # Parse range-diff output to build mappings
        mappings, added, removed, modified = self._parse_range_diff(range_diff)

        # Determine equivalence
        # Equivalent if no content changes, only structural
        equivalent = len(modified) == 0 and len(added) == 0 and len(removed) == 0

        # Generate summary
        summary_parts = []
        if equivalent:
            summary_parts.append("Ranges are semantically equivalent")
        else:
            if added:
                summary_parts.append(f"{len(added)} commits added")
            if removed:
                summary_parts.append(f"{len(removed)} commits removed")
            if modified:
                summary_parts.append(f"{len(modified)} commits modified")

        summary = f"{original_count} → {rebased_count} commits. " + ", ".join(summary_parts)

        return RangeDiffResponse(
            equivalent=equivalent,
            original_count=original_count,
            rebased_count=rebased_count,
            mappings=mappings,
            added_commits=added,
            removed_commits=removed,
            modified_commits=modified,
            summary=summary,
        )

    def _parse_range_diff(
        self, range_diff: str
    ) -> tuple[list[CommitMapping], list[str], list[str], list[ModifiedCommit]]:
        """Parse git range-diff output."""
        mappings = []
        added = []
        removed = []
        modified = []

        # Range-diff format:
        # 1:  abc1234 = 1:  def5678 Subject line
        # 2:  abc1234 ! 2:  def5678 Subject line (modified)
        # -:  ------- > 3:  new1234 New commit
        # 3:  old1234 < -:  ------- Removed commit

        import re

        for line in range_diff.split("\n"):
            # Match the range-diff line format
            match = re.match(
                r"^\s*(\d+|-):?\s+([a-f0-9]+|-------)?\s*([=!<>])\s*(\d+|-):?\s+([a-f0-9]+|-------)?\s*(.*)",
                line,
            )
            if not match:
                continue

            left_num, left_sha, symbol, right_num, right_sha, subject = match.groups()

            if symbol == "=":
                # Identical
                mappings.append(
                    CommitMapping(
                        original_sha=left_sha or "",
                        rebased_sha=right_sha,
                        status="identical",
                    )
                )
            elif symbol == "!":
                # Modified
                mappings.append(
                    CommitMapping(
                        original_sha=left_sha or "",
                        rebased_sha=right_sha,
                        status="modified",
                    )
                )
                modified.append(
                    ModifiedCommit(
                        original_sha=left_sha or "",
                        rebased_sha=right_sha or "",
                        message_changed=True,  # Simplified
                        content_changed=True,
                        diff_summary=subject,
                    )
                )
            elif symbol == ">":
                # Added in rebased
                added.append(right_sha or "")
                mappings.append(
                    CommitMapping(
                        original_sha="",
                        rebased_sha=right_sha,
                        status="split",
                        notes="Added during rebase",
                    )
                )
            elif symbol == "<":
                # Removed from original
                removed.append(left_sha or "")
                mappings.append(
                    CommitMapping(
                        original_sha=left_sha or "",
                        rebased_sha=None,
                        status="dropped",
                        notes="Removed during rebase",
                    )
                )

        return mappings, added, removed, modified
