"""Commit clustering with embedding-based similarity for Freebase."""

import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import httpx

from models import CommitCluster, CommitNode, ClusterAction


# =============================================================================
# Embedding Client
# =============================================================================


class EmbeddingClient:
    """Client for generating text embeddings via OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/mistral-embed-2312",
        base_url: str = "https://openrouter.ai/api/v1",
        dimensions: int = 1024,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            # Fallback to simple hash-based pseudo-embedding
            return self._fallback_embedding(text)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": text,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Truncate to dimensions (Qwen3 supports Matryoshka Representation Learning)
            embedding = data["data"][0]["embedding"][:self.dimensions]

        self._cache[cache_key] = embedding
        return embedding

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        if not self.api_key:
            return [self._fallback_embedding(t) for t in texts]

        # Check cache first
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": uncached_texts,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                for item in data["data"]:
                    idx = item["index"]
                    # Truncate to dimensions (Qwen3 supports Matryoshka Representation Learning)
                    embedding = item["embedding"][:self.dimensions]
                    original_idx = uncached_indices[idx]
                    results[original_idx] = embedding
                    self._cache[self._cache_key(uncached_texts[idx])] = embedding

        return [r for r in results if r is not None]

    def _fallback_embedding(self, text: str, dim: Optional[int] = None) -> list[float]:
        """Generate a simple hash-based pseudo-embedding when no API key."""
        # This is NOT a real embedding, just for testing without API
        import struct

        if dim is None:
            dim = self.dimensions

        h = hashlib.sha256(text.lower().encode())
        # Generate dim floats from hash
        result = []
        for i in range(dim):
            # Use different parts of extended hash
            extended = hashlib.sha256(h.digest() + str(i).encode()).digest()
            # Convert first 4 bytes to float in [-1, 1]
            val = struct.unpack("f", extended[:4])[0]
            # Normalize to [-1, 1]
            result.append(max(-1.0, min(1.0, val / 1e38)))
        return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# =============================================================================
# Commit Analysis
# =============================================================================


# Patterns that suggest a fixup commit
FIXUP_PATTERNS = [
    r"^fix(?:up)?[:\s]",
    r"^wip[:\s]",
    r"^oops",
    r"^typo",
    r"^address(?:es)?\s+(?:review|comment|feedback)",
    r"^review\s+fix",
    r"^minor\s+fix",
    r"^quick\s+fix",
    r"^hotfix",
    r"^patch",
    r"^tweak",
    r"^adjust",
    r"^correct",
    r"^amend",
    r"^fixme",
    r"^todo",
    r"^temp",
    r"^checkpoint",
    r"^save\s+progress",
    r"^work\s+in\s+progress",
    r"^squash",
]

# Patterns that suggest a revert
REVERT_PATTERNS = [
    r"^revert",
    r"^undo",
    r"^rollback",
    r"^back\s*out",
]

# Patterns that suggest formatting only
FORMATTING_PATTERNS = [
    r"^format",
    r"^lint",
    r"^style",
    r"^prettier",
    r"^black",
    r"^eslint",
    r"^whitespace",
    r"^indent",
    r"^trailing",
    r"^cleanup\s+(?:format|style|lint)",
]

# Patterns that suggest test-only changes
TEST_PATTERNS = [
    r"^test",
    r"^add\s+test",
    r"^fix\s+test",
    r"^update\s+test",
    r"^spec",
]

# Patterns that suggest docs-only changes
DOCS_PATTERNS = [
    r"^doc",
    r"^readme",
    r"^comment",
    r"^jsdoc",
    r"^docstring",
    r"^changelog",
]

# Patterns that suggest dependency updates
DEPS_PATTERNS = [
    r"^(?:update|upgrade|bump)\s+(?:dep|package|version)",
    r"^npm\s+update",
    r"^yarn\s+upgrade",
    r"^pip\s+update",
    r"^poetry\s+update",
]


def detect_commit_tags(commit: CommitNode) -> list[str]:
    """Detect tags for a commit based on message and files."""
    tags = []
    subject_lower = commit.subject.lower()

    # Check message patterns
    for pattern in FIXUP_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("likely_fixup")
            break

    for pattern in REVERT_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("likely_revert")
            break

    for pattern in FORMATTING_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("formatting_only")
            break

    for pattern in TEST_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("test_only")
            break

    for pattern in DOCS_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("docs_only")
            break

    for pattern in DEPS_PATTERNS:
        if re.search(pattern, subject_lower):
            tags.append("dependency_update")
            break

    # Check file patterns
    file_paths = [f.path for f in commit.files]

    # Test files only
    test_patterns = [r"test", r"spec", r"__tests__", r"_test\."]
    all_test_files = all(
        any(re.search(p, f, re.IGNORECASE) for p in test_patterns) for f in file_paths
    )
    if all_test_files and file_paths and "test_only" not in tags:
        tags.append("test_only")

    # Docs files only
    doc_patterns = [r"\.md$", r"docs/", r"readme", r"changelog", r"\.rst$"]
    all_doc_files = all(
        any(re.search(p, f, re.IGNORECASE) for p in doc_patterns) for f in file_paths
    )
    if all_doc_files and file_paths and "docs_only" not in tags:
        tags.append("docs_only")

    # Generated code detection
    generated_patterns = [r"generated", r"auto-generated", r"\.lock$", r"package-lock", r"yarn\.lock"]
    all_generated = all(
        any(re.search(p, f, re.IGNORECASE) for p in generated_patterns) for f in file_paths
    )
    if all_generated and file_paths:
        tags.append("generated_code")

    # WIP detection from message
    if re.search(r"\bwip\b", subject_lower) and "likely_fixup" not in tags:
        tags.append("wip")

    # Checkpoint detection
    if re.search(r"checkpoint|save\s+point|backup", subject_lower):
        tags.append("checkpoint")

    return tags


def calculate_file_overlap(files1: list[str], files2: list[str]) -> float:
    """Calculate Jaccard similarity of file sets."""
    if not files1 or not files2:
        return 0.0

    set1 = set(files1)
    set2 = set(files2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def calculate_directory_overlap(files1: list[str], files2: list[str]) -> float:
    """Calculate overlap of directories touched."""
    if not files1 or not files2:
        return 0.0

    def get_dirs(files: list[str]) -> set[str]:
        dirs = set()
        for f in files:
            parts = f.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))
        return dirs

    dirs1 = get_dirs(files1)
    dirs2 = get_dirs(files2)

    if not dirs1 or not dirs2:
        return 0.0

    intersection = len(dirs1 & dirs2)
    union = len(dirs1 | dirs2)

    return intersection / union if union > 0 else 0.0


def calculate_temporal_proximity(ts1: int, ts2: int, max_gap: int = 3600) -> float:
    """Calculate temporal proximity score (1.0 = same time, 0.0 = far apart)."""
    gap = abs(ts1 - ts2)
    if gap >= max_gap:
        return 0.0
    return 1.0 - (gap / max_gap)


# =============================================================================
# Clustering Algorithm
# =============================================================================


@dataclass
class ClusterBuilder:
    """Builds clusters from commits using multiple signals."""

    embedding_client: EmbeddingClient
    similarity_threshold: float = 0.6
    file_overlap_weight: float = 0.3
    dir_overlap_weight: float = 0.2
    temporal_weight: float = 0.1
    embedding_weight: float = 0.4

    # Internal state
    commits: list[CommitNode] = field(default_factory=list)
    similarity_matrix: dict[tuple[str, str], float] = field(default_factory=dict)

    async def build_clusters(self, commits: list[CommitNode]) -> list[CommitCluster]:
        """Build clusters from a list of commits."""
        self.commits = commits

        if not commits:
            return []

        # Step 1: Get embeddings for all commit messages
        messages = [c.subject + ("\n" + c.body if c.body else "") for c in commits]
        embeddings = await self.embedding_client.get_embeddings(messages)

        # Store embeddings on commits
        for commit, embedding in zip(commits, embeddings):
            commit.embedding = embedding

        # Step 2: Build similarity matrix
        await self._build_similarity_matrix(commits)

        # Step 3: Group commits using connected components with threshold
        clusters = self._find_clusters(commits)

        # Step 4: Determine actions for each cluster
        result = []
        for i, cluster_commits in enumerate(clusters):
            cluster = self._create_cluster(i, cluster_commits)
            result.append(cluster)

        return result

    async def _build_similarity_matrix(self, commits: list[CommitNode]) -> None:
        """Build pairwise similarity matrix."""
        self.similarity_matrix = {}

        for i, c1 in enumerate(commits):
            for j, c2 in enumerate(commits):
                if i >= j:
                    continue

                similarity = self._calculate_similarity(c1, c2)
                self.similarity_matrix[(c1.sha, c2.sha)] = similarity
                self.similarity_matrix[(c2.sha, c1.sha)] = similarity

    def _calculate_similarity(self, c1: CommitNode, c2: CommitNode) -> float:
        """Calculate combined similarity score between two commits."""
        scores = []

        # File overlap
        files1 = [f.path for f in c1.files]
        files2 = [f.path for f in c2.files]
        file_sim = calculate_file_overlap(files1, files2)
        scores.append((file_sim, self.file_overlap_weight))

        # Directory overlap
        dir_sim = calculate_directory_overlap(files1, files2)
        scores.append((dir_sim, self.dir_overlap_weight))

        # Temporal proximity
        temporal_sim = calculate_temporal_proximity(c1.timestamp, c2.timestamp)
        scores.append((temporal_sim, self.temporal_weight))

        # Embedding similarity
        if c1.embedding and c2.embedding:
            embed_sim = cosine_similarity(c1.embedding, c2.embedding)
            # Normalize from [-1, 1] to [0, 1]
            embed_sim = (embed_sim + 1) / 2
            scores.append((embed_sim, self.embedding_weight))

        # Weighted average
        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in scores) / total_weight

    def _find_clusters(self, commits: list[CommitNode]) -> list[list[CommitNode]]:
        """Find clusters using union-find with similarity threshold."""
        # Union-Find data structure
        parent: dict[str, str] = {c.sha: c.sha for c in commits}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union commits that are similar enough
        for i, c1 in enumerate(commits):
            for j, c2 in enumerate(commits):
                if i >= j:
                    continue

                sim = self.similarity_matrix.get((c1.sha, c2.sha), 0)

                # Also consider explicit fixup relationships
                is_fixup_pair = self._is_fixup_pair(c1, c2)

                if sim >= self.similarity_threshold or is_fixup_pair:
                    union(c1.sha, c2.sha)

        # Group by root
        groups: dict[str, list[CommitNode]] = defaultdict(list)
        for commit in commits:
            root = find(commit.sha)
            groups[root].append(commit)

        # Sort groups by earliest commit timestamp
        sorted_groups = sorted(groups.values(), key=lambda g: min(c.timestamp for c in g))

        # Sort commits within each group by timestamp
        for group in sorted_groups:
            group.sort(key=lambda c: c.timestamp)

        return sorted_groups

    def _is_fixup_pair(self, c1: CommitNode, c2: CommitNode) -> bool:
        """Check if c2 is likely a fixup of c1."""
        # c2 should come after c1
        if c2.timestamp <= c1.timestamp:
            return False

        # c2 should have fixup-like tags
        if "likely_fixup" not in c2.tags:
            return False

        # Should touch overlapping files
        files1 = set(f.path for f in c1.files)
        files2 = set(f.path for f in c2.files)

        return bool(files1 & files2)

    def _create_cluster(
        self, index: int, commits: list[CommitNode]
    ) -> CommitCluster:
        """Create a cluster with suggested action."""
        cluster_id = f"cluster_{index}"

        # Assign cluster ID to commits
        for commit in commits:
            commit.cluster_id = cluster_id

        # Determine label
        label = self._generate_cluster_label(commits)

        # Determine suggested action
        action, target, reasoning, needs_input, question, confidence = self._suggest_action(commits)

        return CommitCluster(
            cluster_id=cluster_id,
            label=label,
            commits=[c.sha for c in commits],
            suggested_action=action,
            target_commits=target,
            reasoning=reasoning,
            needs_user_input=needs_input,
            question=question,
            confidence=confidence,
        )

    def _generate_cluster_label(self, commits: list[CommitNode]) -> str:
        """Generate a human-readable label for a cluster."""
        if len(commits) == 1:
            return commits[0].subject[:50]

        # Find common themes
        all_tags = [tag for c in commits for tag in c.tags]
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1

        # Most common tag
        if tag_counts:
            most_common = max(tag_counts.items(), key=lambda x: x[1])[0]
            tag_labels = {
                "likely_fixup": "Fixup commits",
                "formatting_only": "Formatting changes",
                "test_only": "Test changes",
                "docs_only": "Documentation",
                "wip": "Work in progress",
                "checkpoint": "Checkpoints",
            }
            if most_common in tag_labels:
                return f"{tag_labels[most_common]} ({len(commits)} commits)"

        # Use first commit subject as base
        return f"{commits[0].subject[:30]}... ({len(commits)} commits)"

    def _suggest_action(
        self, commits: list[CommitNode]
    ) -> tuple[ClusterAction, Optional[int], list[str], bool, Optional[str], float]:
        """Suggest an action for a cluster.

        Returns: (action, target_commits, reasoning, needs_input, question, confidence)
        """
        if len(commits) == 1:
            # Single commit - keep as is unless it's a revert of nothing
            if "likely_revert" in commits[0].tags:
                return (
                    "drop_all",
                    None,
                    ["Single revert commit with no corresponding commit to revert"],
                    True,
                    "This appears to be a standalone revert. Should it be dropped?",
                    0.6,
                )
            return ("keep_separate", None, ["Single commit"], False, None, 1.0)

        reasoning = []
        confidence = 0.8

        # Check for all fixups
        fixup_count = sum(1 for c in commits if "likely_fixup" in c.tags)
        if fixup_count == len(commits) - 1:
            reasoning.append(f"{fixup_count} fixup commits following main commit")
            return ("squash_all", None, reasoning, False, None, 0.95)

        # Check for formatting-only cluster
        formatting_count = sum(1 for c in commits if "formatting_only" in c.tags)
        if formatting_count == len(commits):
            reasoning.append("All commits are formatting changes")
            return ("squash_all", None, reasoning, False, None, 0.9)

        # Check for revert pairs
        revert_count = sum(1 for c in commits if "likely_revert" in c.tags)
        if revert_count > 0 and revert_count * 2 >= len(commits):
            reasoning.append(f"{revert_count} reverts that may cancel out other commits")
            return (
                "drop_all",
                None,
                reasoning,
                True,
                "This cluster contains commits and their reverts. Drop them all?",
                0.7,
            )

        # Check for WIP/checkpoint cluster
        wip_count = sum(1 for c in commits if "wip" in c.tags or "checkpoint" in c.tags)
        if wip_count > len(commits) / 2:
            reasoning.append(f"{wip_count} WIP/checkpoint commits")
            return ("squash_all", None, reasoning, False, None, 0.85)

        # Check for mixed-purpose cluster
        unique_purposes = set()
        for c in commits:
            if "test_only" in c.tags:
                unique_purposes.add("test")
            elif "docs_only" in c.tags:
                unique_purposes.add("docs")
            elif "formatting_only" in c.tags:
                unique_purposes.add("format")
            else:
                unique_purposes.add("code")

        if len(unique_purposes) > 1:
            reasoning.append(f"Mixed purposes: {', '.join(unique_purposes)}")
            return (
                "needs_split",
                None,
                reasoning,
                True,
                f"This cluster mixes {', '.join(unique_purposes)}. Should it be split?",
                0.6,
            )

        # Default: squash to reasonable number
        if len(commits) > 5:
            target = max(2, len(commits) // 3)
            reasoning.append(f"Large cluster of {len(commits)} related commits")
            return (
                "squash_to_n",
                target,
                reasoning,
                True,
                f"Squash {len(commits)} commits to {target}?",
                0.7,
            )

        # Small cluster - squash all
        reasoning.append(f"Small cluster of {len(commits)} related commits")
        return ("squash_all", None, reasoning, False, None, 0.8)
