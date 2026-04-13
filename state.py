"""Session state persistence for Freebase.

All state is persisted to .git/freebase/ for crash recovery.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import (
    ExecutionCheckpoint,
    OperationLog,
    PlanGenerateResponse,
    SessionState,
    SessionStatus,
    Profile,
)


class StateManager:
    """Manages persistent state in .git/freebase/."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.freebase_dir = self.repo_path / ".git" / "freebase"
        self.plans_dir = self.freebase_dir / "plans"
        self.checkpoints_dir = self.freebase_dir / "checkpoints"
        self.state_file = self.freebase_dir / "state.json"
        self.log_file = self.freebase_dir / "log.jsonl"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.freebase_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Session State
    # =========================================================================

    def load_session(self) -> Optional[SessionState]:
        """Load the current session state."""
        if not self.state_file.exists():
            return None

        try:
            data = json.loads(self.state_file.read_text())
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None

    def save_session(self, state: SessionState) -> None:
        """Save the current session state."""
        self.state_file.write_text(state.model_dump_json(indent=2))

    def create_session(
        self,
        branch: str,
        base_branch: str = "main",
    ) -> SessionState:
        """Create a new session."""
        state = SessionState(
            session_id=str(uuid.uuid4()),
            started_at=datetime.utcnow(),
            repo_path=str(self.repo_path),
            branch=branch,
            base_branch=base_branch,
            status="idle",
        )
        self.save_session(state)
        return state

    def update_session(
        self,
        status: Optional[SessionStatus] = None,
        current_plan_id: Optional[str] = None,
        current_step: Optional[int] = None,
        worktree_path: Optional[str] = None,
        backup_ref: Optional[str] = None,
        profile: Optional[Profile] = None,
        last_error: Optional[str] = None,
    ) -> SessionState:
        """Update the current session state."""
        state = self.load_session()
        if state is None:
            raise ValueError("No active session")

        if status is not None:
            state.status = status
        if current_plan_id is not None:
            state.current_plan_id = current_plan_id
        if current_step is not None:
            state.current_step = current_step
        if worktree_path is not None:
            state.worktree_path = worktree_path
        if backup_ref is not None:
            state.backup_ref = backup_ref
        if profile is not None:
            state.profile = profile
        if last_error is not None:
            state.last_error = last_error

        self.save_session(state)
        return state

    def clear_session(self) -> None:
        """Clear the current session."""
        if self.state_file.exists():
            self.state_file.unlink()

    # =========================================================================
    # Plans
    # =========================================================================

    def save_plan(self, plan: PlanGenerateResponse) -> None:
        """Save a rebase plan."""
        plan_file = self.plans_dir / f"plan_{plan.plan_id}.json"
        plan_file.write_text(plan.model_dump_json(indent=2))

    def load_plan(self, plan_id: str) -> Optional[PlanGenerateResponse]:
        """Load a rebase plan."""
        plan_file = self.plans_dir / f"plan_{plan_id}.json"
        if not plan_file.exists():
            return None

        try:
            data = json.loads(plan_file.read_text())
            return PlanGenerateResponse.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None

    def list_plans(self) -> list[str]:
        """List all saved plan IDs."""
        plans = []
        for f in self.plans_dir.glob("plan_*.json"):
            plan_id = f.stem.replace("plan_", "")
            plans.append(plan_id)
        return plans

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan."""
        plan_file = self.plans_dir / f"plan_{plan_id}.json"
        if plan_file.exists():
            plan_file.unlink()
            return True
        return False

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def save_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save an execution checkpoint."""
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))

    def load_checkpoint(self, checkpoint_id: str) -> Optional[ExecutionCheckpoint]:
        """Load an execution checkpoint."""
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None

        try:
            data = json.loads(checkpoint_file.read_text())
            return ExecutionCheckpoint.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None

    def list_checkpoints(self, plan_id: Optional[str] = None) -> list[ExecutionCheckpoint]:
        """List all checkpoints, optionally filtered by plan."""
        checkpoints = []
        for f in self.checkpoints_dir.glob("checkpoint_*.json"):
            try:
                data = json.loads(f.read_text())
                checkpoint = ExecutionCheckpoint.model_validate(data)
                if plan_id is None or checkpoint.plan_id == plan_id:
                    checkpoints.append(checkpoint)
            except (json.JSONDecodeError, ValueError):
                continue

        # Sort by timestamp
        checkpoints.sort(key=lambda c: c.timestamp)
        return checkpoints

    def get_latest_checkpoint(self, plan_id: Optional[str] = None) -> Optional[ExecutionCheckpoint]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints(plan_id)
        return checkpoints[-1] if checkpoints else None

    def create_checkpoint(
        self,
        plan_id: str,
        step_index: int,
        branch_sha: str,
        backup_ref: str,
        worktree_path: Optional[str] = None,
        state: str = "clean",
    ) -> ExecutionCheckpoint:
        """Create a new checkpoint."""
        import time

        checkpoint = ExecutionCheckpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            timestamp=int(time.time()),
            plan_id=plan_id,
            step_index=step_index,
            branch_sha=branch_sha,
            backup_ref=backup_ref,
            worktree_path=worktree_path,
            state=state,
        )
        self.save_checkpoint(checkpoint)
        return checkpoint

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False

    # =========================================================================
    # Operation Log
    # =========================================================================

    def log_operation(
        self,
        operation: str,
        details: dict,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Append an operation to the log."""
        entry = OperationLog(
            timestamp=datetime.utcnow(),
            operation=operation,
            details=details,
            success=success,
            error=error,
        )

        with open(self.log_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_operation_log(self, limit: Optional[int] = None) -> list[OperationLog]:
        """Get operation log entries."""
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(OperationLog.model_validate(data))
                except (json.JSONDecodeError, ValueError):
                    continue

        if limit:
            entries = entries[-limit:]

        return entries

    def clear_operation_log(self) -> None:
        """Clear the operation log."""
        if self.log_file.exists():
            self.log_file.unlink()

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_all(self) -> None:
        """Remove all freebase state."""
        import shutil

        if self.freebase_dir.exists():
            shutil.rmtree(self.freebase_dir)

    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """Remove old checkpoints, keeping the most recent ones."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_count:
            return 0

        removed = 0
        for checkpoint in checkpoints[:-keep_count]:
            if self.delete_checkpoint(checkpoint.checkpoint_id):
                removed += 1

        return removed
