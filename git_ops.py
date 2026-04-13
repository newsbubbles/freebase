"""Low-level git subprocess operations for Freebase."""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GitResult:
    """Result of a git command."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def raise_on_error(self, context: str = "") -> None:
        if not self.success:
            msg = f"Git command failed: {self.stderr.strip()}"
            if context:
                msg = f"{context}: {msg}"
            raise GitError(msg, self.returncode, self.stderr)


class GitError(Exception):
    """Git operation error."""

    def __init__(self, message: str, returncode: int = 1, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class GitOps:
    """Low-level git operations using subprocess."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            # Check if it's a worktree
            git_file = self.repo_path / ".git"
            if not git_file.exists():
                raise GitError(f"Not a git repository: {repo_path}")

    async def run(self, *args: str, check: bool = False, cwd: Optional[Path] = None) -> GitResult:
        """Run a git command asynchronously."""
        cmd = ["git", *args]
        work_dir = cwd or self.repo_path

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        result = GitResult(
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

        if check:
            result.raise_on_error()

        return result

    def run_sync(self, *args: str, check: bool = False, cwd: Optional[Path] = None) -> GitResult:
        """Run a git command synchronously."""
        cmd = ["git", *args]
        work_dir = cwd or self.repo_path

        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
        )

        result = GitResult(
            returncode=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
        )

        if check:
            result.raise_on_error()

        return result

    # =========================================================================
    # Basic Operations
    # =========================================================================

    async def get_current_branch(self) -> str:
        """Get the current branch name."""
        result = await self.run("rev-parse", "--abbrev-ref", "HEAD", check=True)
        return result.stdout.strip()

    async def get_head_sha(self) -> str:
        """Get the current HEAD SHA."""
        result = await self.run("rev-parse", "HEAD", check=True)
        return result.stdout.strip()

    async def get_merge_base(self, ref1: str, ref2: str) -> str:
        """Get the merge base of two refs."""
        result = await self.run("merge-base", ref1, ref2, check=True)
        return result.stdout.strip()

    async def rev_parse(self, ref: str) -> str:
        """Resolve a ref to a SHA."""
        result = await self.run("rev-parse", ref, check=True)
        return result.stdout.strip()

    async def branch_exists(self, branch: str) -> bool:
        """Check if a branch exists."""
        result = await self.run("rev-parse", "--verify", f"refs/heads/{branch}")
        return result.success

    async def ref_exists(self, ref: str) -> bool:
        """Check if a ref exists."""
        result = await self.run("rev-parse", "--verify", ref)
        return result.success

    # =========================================================================
    # Commit Information
    # =========================================================================

    async def get_commit_log(
        self,
        range_spec: str,
        format_string: str = "%H%n%h%n%s%n%b%n%an%n%ae%n%at%n%P%n---COMMIT_END---",
    ) -> str:
        """Get commit log with custom format."""
        result = await self.run(
            "log",
            "--reverse",
            f"--format={format_string}",
            range_spec,
            check=True,
        )
        return result.stdout

    async def get_commit_files(self, sha: str) -> str:
        """Get files changed in a commit with stats."""
        result = await self.run(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            sha,
            check=True,
        )
        return result.stdout

    async def get_commit_numstat(self, sha: str) -> str:
        """Get numeric stats for a commit."""
        result = await self.run(
            "diff-tree",
            "--no-commit-id",
            "--numstat",
            "-r",
            sha,
            check=True,
        )
        return result.stdout

    async def get_commit_diff(self, sha: str) -> str:
        """Get the full diff for a commit."""
        result = await self.run("show", "--format=", sha, check=True)
        return result.stdout

    async def get_commit_count(self, range_spec: str) -> int:
        """Count commits in a range."""
        result = await self.run("rev-list", "--count", range_spec, check=True)
        return int(result.stdout.strip())

    # =========================================================================
    # Diff Operations
    # =========================================================================

    async def get_diff(self, ref1: str, ref2: str, path: Optional[str] = None) -> str:
        """Get diff between two refs."""
        args = ["diff", ref1, ref2]
        if path:
            args.extend(["--", path])
        result = await self.run(*args, check=True)
        return result.stdout

    async def get_range_diff(
        self, base1: str, head1: str, base2: str, head2: str
    ) -> str:
        """Get range-diff between two commit ranges."""
        result = await self.run(
            "range-diff",
            f"{base1}..{head1}",
            f"{base2}..{head2}",
            check=True,
        )
        return result.stdout

    # =========================================================================
    # Branch Operations
    # =========================================================================

    async def create_branch(self, name: str, start_point: Optional[str] = None) -> None:
        """Create a new branch."""
        args = ["branch", name]
        if start_point:
            args.append(start_point)
        await self.run(*args, check=True)

    async def delete_branch(self, name: str, force: bool = False) -> None:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        await self.run("branch", flag, name, check=True)

    async def checkout(self, ref: str) -> None:
        """Checkout a ref."""
        await self.run("checkout", ref, check=True)

    async def reset_hard(self, ref: str) -> None:
        """Hard reset to a ref."""
        await self.run("reset", "--hard", ref, check=True)

    # =========================================================================
    # Worktree Operations
    # =========================================================================

    async def worktree_add(
        self,
        path: str,
        branch: str,
        start_point: Optional[str] = None,
        create_branch: bool = True,
    ) -> None:
        """Add a new worktree."""
        args = ["worktree", "add"]
        if create_branch:
            args.extend(["-b", branch])
        args.append(path)
        if start_point:
            args.append(start_point)
        elif not create_branch:
            args.append(branch)
        await self.run(*args, check=True)

    async def worktree_remove(self, path: str, force: bool = False) -> None:
        """Remove a worktree."""
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(path)
        await self.run(*args, check=True)

    async def worktree_list(self) -> list[dict[str, str]]:
        """List all worktrees."""
        result = await self.run("worktree", "list", "--porcelain", check=True)
        worktrees = []
        current: dict[str, str] = {}

        for line in result.stdout.split("\n"):
            if not line:
                if current:
                    worktrees.append(current)
                    current = {}
            elif line.startswith("worktree "):
                current["path"] = line[9:]
            elif line.startswith("HEAD "):
                current["head"] = line[5:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]
            elif line == "bare":
                current["bare"] = "true"
            elif line == "detached":
                current["detached"] = "true"

        if current:
            worktrees.append(current)

        return worktrees

    # =========================================================================
    # Rebase Operations
    # =========================================================================

    async def rebase_interactive_start(
        self,
        onto: str,
        todo_content: str,
        cwd: Optional[Path] = None,
    ) -> GitResult:
        """Start an interactive rebase with custom todo.

        This creates a temp script that outputs the todo content.
        """
        work_dir = cwd or self.repo_path

        # Write todo to a temp file
        todo_file = work_dir / ".git" / "freebase-todo"
        todo_file.write_text(todo_content)

        # Create editor script that copies our todo
        editor_script = work_dir / ".git" / "freebase-editor.sh"
        editor_script.write_text(f'#!/bin/sh\ncat "{todo_file}" > "$1"\n')
        editor_script.chmod(0o755)

        # Run rebase with our editor
        env = os.environ.copy()
        env["GIT_SEQUENCE_EDITOR"] = str(editor_script)

        proc = await asyncio.create_subprocess_exec(
            "git",
            "rebase",
            "-i",
            onto,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        return GitResult(
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def rebase_continue(self, cwd: Optional[Path] = None) -> GitResult:
        """Continue a rebase in progress."""
        return await self.run("rebase", "--continue", cwd=cwd)

    async def rebase_abort(self, cwd: Optional[Path] = None) -> GitResult:
        """Abort a rebase in progress."""
        return await self.run("rebase", "--abort", cwd=cwd)

    async def rebase_skip(self, cwd: Optional[Path] = None) -> GitResult:
        """Skip the current commit in a rebase."""
        return await self.run("rebase", "--skip", cwd=cwd)

    async def is_rebase_in_progress(self, cwd: Optional[Path] = None) -> bool:
        """Check if a rebase is in progress."""
        work_dir = cwd or self.repo_path
        rebase_merge = work_dir / ".git" / "rebase-merge"
        rebase_apply = work_dir / ".git" / "rebase-apply"
        return rebase_merge.exists() or rebase_apply.exists()

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    async def get_conflicted_files(self, cwd: Optional[Path] = None) -> list[str]:
        """Get list of files with conflicts."""
        result = await self.run("diff", "--name-only", "--diff-filter=U", cwd=cwd)
        if not result.success:
            return []
        return [f for f in result.stdout.strip().split("\n") if f]

    async def get_file_content(self, path: str, ref: Optional[str] = None, cwd: Optional[Path] = None) -> str:
        """Get content of a file, optionally at a specific ref."""
        work_dir = cwd or self.repo_path
        if ref:
            result = await self.run("show", f"{ref}:{path}", cwd=work_dir)
        else:
            file_path = work_dir / path
            if file_path.exists():
                return file_path.read_text()
            return ""
        return result.stdout if result.success else ""

    async def get_conflict_content(
        self, path: str, cwd: Optional[Path] = None
    ) -> dict[str, str]:
        """Get ours, theirs, and base content for a conflicted file."""
        work_dir = cwd or self.repo_path

        # Get content from different stages
        # Stage 1 = base, Stage 2 = ours, Stage 3 = theirs
        ours = await self.run("show", f":2:{path}", cwd=work_dir)
        theirs = await self.run("show", f":3:{path}", cwd=work_dir)
        base = await self.run("show", f":1:{path}", cwd=work_dir)

        return {
            "ours": ours.stdout if ours.success else "",
            "theirs": theirs.stdout if theirs.success else "",
            "base": base.stdout if base.success else "",
        }

    # =========================================================================
    # Staging and Committing
    # =========================================================================

    async def add(self, *paths: str, cwd: Optional[Path] = None) -> None:
        """Stage files."""
        await self.run("add", *paths, cwd=cwd, check=True)

    async def add_all(self, cwd: Optional[Path] = None) -> None:
        """Stage all changes."""
        await self.run("add", "-A", cwd=cwd, check=True)

    async def commit(
        self,
        message: str,
        amend: bool = False,
        no_edit: bool = False,
        cwd: Optional[Path] = None,
    ) -> GitResult:
        """Create a commit."""
        args = ["commit", "-m", message]
        if amend:
            args.append("--amend")
        if no_edit:
            args.append("--no-edit")
        return await self.run(*args, cwd=cwd)

    # =========================================================================
    # Ref Management
    # =========================================================================

    async def create_backup_ref(self, branch: str, prefix: str = "freebase-backup") -> str:
        """Create a backup reference for a branch."""
        import time

        timestamp = int(time.time())
        backup_name = f"{prefix}-{branch}-{timestamp}"
        sha = await self.rev_parse(branch)
        await self.run("update-ref", f"refs/freebase/{backup_name}", sha, check=True)
        return backup_name

    async def get_backup_refs(self, prefix: str = "freebase-backup") -> list[str]:
        """List all backup refs."""
        result = await self.run("for-each-ref", "--format=%(refname:short)", "refs/freebase/")
        if not result.success:
            return []
        return [r for r in result.stdout.strip().split("\n") if r and r.startswith(prefix)]

    async def delete_backup_ref(self, name: str) -> None:
        """Delete a backup reference."""
        await self.run("update-ref", "-d", f"refs/freebase/{name}", check=True)

    # =========================================================================
    # Status
    # =========================================================================

    async def status_porcelain(self, cwd: Optional[Path] = None) -> str:
        """Get porcelain status."""
        result = await self.run("status", "--porcelain", cwd=cwd, check=True)
        return result.stdout

    async def is_clean(self, cwd: Optional[Path] = None) -> bool:
        """Check if working tree is clean."""
        status = await self.status_porcelain(cwd=cwd)
        return not status.strip()


def parse_commit_log(raw_log: str) -> list[dict[str, str]]:
    """Parse raw git log output into structured data."""
    commits = []
    entries = raw_log.split("---COMMIT_END---")

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        lines = entry.split("\n")
        if len(lines) < 7:
            continue

        # Parse format: SHA, short_sha, subject, body..., author, email, timestamp, parents
        sha = lines[0]
        short_sha = lines[1]
        subject = lines[2]

        # Body is everything between subject and the last 4 lines
        body_lines = lines[3:-4] if len(lines) > 7 else []
        body = "\n".join(body_lines).strip() if body_lines else None

        author = lines[-4]
        email = lines[-3]
        timestamp = lines[-2]
        parents = lines[-1].split() if lines[-1] else []

        commits.append(
            {
                "sha": sha,
                "short_sha": short_sha,
                "subject": subject,
                "body": body,
                "author": author,
                "author_email": email,
                "timestamp": int(timestamp) if timestamp.isdigit() else 0,
                "parents": parents,
            }
        )

    return commits


def parse_file_status(raw_status: str) -> list[dict[str, str]]:
    """Parse git diff-tree --name-status output."""
    files = []
    for line in raw_status.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        status_code = parts[0][0]  # First char is the status
        status_map = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
        }

        file_info = {
            "status": status_map.get(status_code, "modified"),
            "path": parts[-1],
        }

        # Handle renames/copies which have old and new paths
        if status_code in ("R", "C") and len(parts) >= 3:
            file_info["old_path"] = parts[1]
            file_info["path"] = parts[2]

        files.append(file_info)

    return files


def parse_numstat(raw_numstat: str) -> list[dict[str, int | str]]:
    """Parse git diff-tree --numstat output."""
    stats = []
    for line in raw_numstat.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        insertions = parts[0]
        deletions = parts[1]
        path = parts[2]

        # Binary files show "-" for insertions/deletions
        is_binary = insertions == "-" or deletions == "-"

        stats.append(
            {
                "path": path,
                "insertions": 0 if is_binary else int(insertions),
                "deletions": 0 if is_binary else int(deletions),
                "is_binary": is_binary,
            }
        )

    return stats


def parse_conflict_markers(content: str) -> list[dict[str, any]]:
    """Parse conflict markers from file content."""
    markers = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start_line = i + 1
            ours_lines = []
            theirs_lines = []
            base_lines = []
            in_ours = True
            in_base = False

            i += 1
            while i < len(lines):
                if lines[i].startswith("|||||||"):
                    in_ours = False
                    in_base = True
                elif lines[i].startswith("======="):
                    in_ours = False
                    in_base = False
                elif lines[i].startswith(">>>>>>>"):
                    end_line = i + 1
                    markers.append(
                        {
                            "start_line": start_line,
                            "end_line": end_line,
                            "ours_lines": ours_lines,
                            "theirs_lines": theirs_lines,
                            "base_lines": base_lines if base_lines else None,
                        }
                    )
                    break
                elif in_ours:
                    ours_lines.append(lines[i])
                elif in_base:
                    base_lines.append(lines[i])
                else:
                    theirs_lines.append(lines[i])
                i += 1
        i += 1

    return markers
