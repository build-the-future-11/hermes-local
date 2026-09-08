from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hermes_local.substrate import (
    AbstainOutput,
    TextOutput,
    WorkspaceState,
    acl_labels_are_authorized,
    authorized_workspace_view,
    canonical_sha256,
    validate_envelope_against_workspace,
    validate_workspace_snapshot,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class HermesEvidence:
    evidence_id: str
    text: str
    source_uri: str
    content_hash: str
    acl_labels: tuple[str, ...] = ()

    @classmethod
    def from_text(
        cls,
        evidence_id: str,
        text: str,
        source_uri: str,
        *,
        acl_labels: tuple[str, ...] = (),
    ) -> HermesEvidence:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(evidence_id, text, source_uri, digest, acl_labels)


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    source_uri: str
    content_hash: str
    span_start: int
    span_end: int
    quote: str


@dataclass(frozen=True, slots=True)
class MemoryWriteProposal:
    key: str
    value: str
    evidence_ids: tuple[str, ...]
    requires_approval: bool = True


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer: str | None
    citations: tuple[Citation, ...]
    confidence: float
    abstention_reason: str | None
    memory_write_proposals: tuple[MemoryWriteProposal, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.answer is None


class HermesGroundedService:
    """Deterministic extractive baseline for the Hermes grounding contract.

    The service does not write memory. It can only return a proposal that another
    policy component may approve. Evidence is filtered by ACL before any content
    scoring takes place.
    """

    def __init__(self, *, minimum_support: float = 0.5) -> None:
        if not 0.0 < minimum_support <= 1.0:
            raise ValueError("minimum_support must be in (0, 1]")
        self.minimum_support = minimum_support

    def answer(
        self,
        request: str,
        evidence: tuple[HermesEvidence, ...] | list[HermesEvidence],
        *,
        granted_acl: frozenset[str] = frozenset(),
        propose_memory: bool = False,
    ) -> GroundedAnswer:
        query_tokens = _content_tokens(request)
        if not query_tokens:
            return self._abstain("request_has_no_groundable_terms")

        authorized = tuple(
            item for item in evidence if _is_authorized(item.acl_labels, granted_acl)
        )
        if not authorized:
            return self._abstain("no_authorized_evidence")

        candidates: list[tuple[float, str, int, int, HermesEvidence]] = []
        for item in authorized:
            if hashlib.sha256(item.text.encode("utf-8")).hexdigest() != item.content_hash:
                continue
            for match in _SENTENCE_RE.finditer(item.text):
                sentence = match.group().strip()
                if not sentence:
                    continue
                sentence_tokens = _content_tokens(sentence)
                overlap = query_tokens.intersection(sentence_tokens)
                support = len(overlap) / len(query_tokens)
                start = match.start() + len(match.group()) - len(match.group().lstrip())
                end = start + len(sentence)
                candidates.append((support, item.evidence_id, start, end, item))

        if not candidates:
            return self._abstain("no_integrity_valid_evidence")
        support, _, start, end, selected = max(candidates, key=lambda row: (row[0], row[1]))
        if support < self.minimum_support:
            return self._abstain("insufficient_claim_support", confidence=support)

        quote = selected.text[start:end]
        citation = Citation(
            evidence_id=selected.evidence_id,
            source_uri=selected.source_uri,
            content_hash=selected.content_hash,
            span_start=start,
            span_end=end,
            quote=quote,
        )
        proposals: tuple[MemoryWriteProposal, ...] = ()
        if propose_memory:
            key_digest = hashlib.sha256(request.strip().encode("utf-8")).hexdigest()
            proposals = (
                MemoryWriteProposal(
                    key=f"hermes:{key_digest}",
                    value=quote,
                    evidence_ids=(selected.evidence_id,),
                ),
            )
        return GroundedAnswer(
            answer=quote,
            citations=(citation,),
            confidence=support,
            abstention_reason=None,
            memory_write_proposals=proposals,
        )

    @staticmethod
    def _abstain(reason: str, confidence: float = 0.0) -> GroundedAnswer:
        return GroundedAnswer(
            answer=None,
            citations=(),
            confidence=confidence,
            abstention_reason=reason,
        )


class HermesWorkspaceRuntime:
    """Bind the deterministic Hermes baseline to the shared workspace protocol."""

    def __init__(self, service: HermesGroundedService | None = None) -> None:
        self.service = service or HermesGroundedService()

    def respond(
        self,
        workspace: WorkspaceState,
        request: str,
        *,
        allowed_evidence_ids: Collection[str] | None = None,
    ) -> TextOutput | AbstainOutput:
        workspace = validate_workspace_snapshot(workspace)
        view = authorized_workspace_view(workspace)
        allowed = set(allowed_evidence_ids) if allowed_evidence_ids is not None else None
        available = [
            HermesEvidence(
                evidence_id=item.evidence_id,
                text=item.text,
                source_uri=item.source_uri,
                content_hash=item.content_sha256,
                acl_labels=tuple(sorted(item.acl_labels)),
            )
            for item in view.evidence
            if item.text is not None
            and (allowed is None or item.evidence_id in allowed)
        ]
        grounded = self.service.answer(
            request,
            available,
            granted_acl=frozenset(workspace.permissions.granted_acl_labels),
        )
        workspace_sha256 = workspace.sha256
        model_identity_sha256 = canonical_sha256(workspace.model_identity)
        if grounded.abstained:
            envelope: TextOutput | AbstainOutput = AbstainOutput(
                workspace_sha256=workspace_sha256,
                model_identity_sha256=model_identity_sha256,
                reason=grounded.abstention_reason or "insufficient_grounding",
                unresolved=[request],
            )
        else:
            envelope = TextOutput(
                workspace_sha256=workspace_sha256,
                model_identity_sha256=model_identity_sha256,
                text=grounded.answer or "",
                citation_evidence_ids=[
                    citation.evidence_id for citation in grounded.citations
                ],
                confidence=grounded.confidence,
            )
        validate_envelope_against_workspace(envelope, workspace)
        return envelope

    def respond_after_execution(
        self,
        workspace: WorkspaceState,
        request: str,
        *,
        execution_succeeded: bool,
        failure_code: str | None = None,
        upstream_component_failure: str | None = None,
        allowed_evidence_ids: Collection[str] | None = None,
    ) -> TextOutput | AbstainOutput:
        """Ground a final response without masking an upstream tool failure."""

        workspace = validate_workspace_snapshot(workspace)
        if execution_succeeded:
            if failure_code is not None:
                raise ValueError("successful execution cannot include a failure code")
            if upstream_component_failure is None:
                return self.respond(
                    workspace,
                    request,
                    allowed_evidence_ids=allowed_evidence_ids,
                )
        elif not failure_code:
            raise ValueError("failed execution requires a sanitized failure code")
        if upstream_component_failure not in {None, "kronos"}:
            raise ValueError("unsupported upstream component failure")
        reason = (
            f"upstream_execution_failed:{failure_code}"
            if not execution_succeeded
            else f"upstream_component_failed:{upstream_component_failure}"
        )
        envelope = AbstainOutput(
            workspace_sha256=workspace.sha256,
            model_identity_sha256=canonical_sha256(workspace.model_identity),
            reason=reason,
            unresolved=[request],
        )
        validate_envelope_against_workspace(envelope, workspace)
        return envelope


class HermesForwardOutput(TypedDict, total=False):
    attribution_logits: Tensor
    confidence_logit: Tensor
    memory_write_logit: Tensor
    loss: Tensor


class HermesGroundingModule(nn.Module):
    """Trainable attribution, confidence, and memory-proposal heads.

    This is deliberately a small component model, not a released Hermes
    checkpoint. It exposes measurable losses so family training cannot silently
    degrade into a non-differentiable placeholder.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        if vocab_size < 2 or hidden_size < 2:
            raise ValueError("vocab_size and hidden_size must be at least 2")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        pair_size = hidden_size * 3
        self.attribution_head = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        self.confidence_head = nn.Linear(hidden_size * 2, 1)
        self.memory_write_head = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        query_ids: Tensor,
        evidence_ids: Tensor,
        *,
        query_mask: Tensor | None = None,
        evidence_mask: Tensor | None = None,
        attribution_targets: Tensor | None = None,
        correctness_targets: Tensor | None = None,
        memory_targets: Tensor | None = None,
    ) -> HermesForwardOutput:
        if query_ids.ndim != 2 or evidence_ids.ndim != 3:
            raise ValueError("expected query [batch, tokens] and evidence [batch, items, tokens]")
        if query_ids.shape[0] != evidence_ids.shape[0]:
            raise ValueError("query and evidence batch sizes differ")
        if query_ids.shape[0] == 0 or query_ids.shape[1] == 0:
            raise ValueError("query batch and token dimensions must be non-empty")
        if evidence_ids.shape[1] == 0 or evidence_ids.shape[2] == 0:
            raise ValueError("evidence item and token dimensions must be non-empty")
        _validate_token_ids(query_ids, self.embedding.num_embeddings, "query_ids")
        _validate_token_ids(evidence_ids, self.embedding.num_embeddings, "evidence_ids")
        if query_mask is not None and query_mask.shape != query_ids.shape:
            raise ValueError("query_mask must match query_ids")
        if evidence_mask is not None and evidence_mask.shape != evidence_ids.shape:
            raise ValueError("evidence_mask must match evidence_ids")
        query_mask = query_ids.ne(0) if query_mask is None else query_mask.bool()
        evidence_mask = evidence_ids.ne(0) if evidence_mask is None else evidence_mask.bool()
        if torch.any(query_mask.sum(dim=1) == 0):
            raise ValueError("each query requires at least one unmasked token")
        if torch.any(evidence_mask.any(dim=-1).sum(dim=1) == 0):
            raise ValueError("each query requires at least one evidence item")

        query = _masked_mean(self.embedding(query_ids), query_mask)
        embedded_evidence = self.embedding(evidence_ids)
        evidence = _masked_mean(embedded_evidence, evidence_mask)
        expanded_query = query.unsqueeze(1).expand_as(evidence)
        pairs = torch.cat((expanded_query, evidence, expanded_query * evidence), dim=-1)
        attribution_logits = self.attribution_head(pairs)

        support_weights = attribution_logits.softmax(dim=-1)[..., 0]
        present = evidence_mask.any(dim=-1)
        support_weights = support_weights.masked_fill(~present, 0.0)
        normalizer = support_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        context = (evidence * (support_weights / normalizer).unsqueeze(-1)).sum(dim=1)
        state = torch.cat((query, context), dim=-1)
        confidence_logit = self.confidence_head(state).squeeze(-1)
        memory_write_logit = self.memory_write_head(state).squeeze(-1)

        output: HermesForwardOutput = {
            "attribution_logits": attribution_logits,
            "confidence_logit": confidence_logit,
            "memory_write_logit": memory_write_logit,
        }
        losses: list[Tensor] = []
        if attribution_targets is not None:
            if attribution_targets.shape != evidence_ids.shape[:2]:
                raise ValueError("attribution targets must match batch and evidence items")
            if attribution_targets.dtype not in {torch.int32, torch.int64}:
                raise ValueError("attribution targets must use an integer tensor dtype")
            if torch.any((attribution_targets < 0) | (attribution_targets > 2)):
                raise ValueError("attribution targets must use classes zero through two")
            losses.append(
                F.cross_entropy(
                    attribution_logits[present],
                    attribution_targets[present],
                )
            )
        if correctness_targets is not None:
            _validate_binary_targets(
                correctness_targets, query_ids.shape[0], "correctness_targets"
            )
            losses.append(
                F.binary_cross_entropy_with_logits(
                    confidence_logit, correctness_targets.float()
                )
            )
        if memory_targets is not None:
            _validate_binary_targets(memory_targets, query_ids.shape[0], "memory_targets")
            losses.append(
                F.binary_cross_entropy_with_logits(memory_write_logit, memory_targets.float())
            )
        if losses:
            output["loss"] = torch.stack(losses).sum()
        return output


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in (item.lower() for item in _TOKEN_RE.findall(text))
        if token not in _STOP_WORDS
    )


def _is_authorized(required: tuple[str, ...], granted: frozenset[str]) -> bool:
    return acl_labels_are_authorized(required, granted)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _validate_token_ids(token_ids: Tensor, vocab_size: int, name: str) -> None:
    if token_ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"{name} must use an integer tensor dtype")
    if torch.any(token_ids < 0) or torch.any(token_ids >= vocab_size):
        raise ValueError(f"{name} contains a token outside the configured vocabulary")


def _validate_binary_targets(targets: Tensor, batch_size: int, name: str) -> None:
    if targets.shape != (batch_size,):
        raise ValueError(f"{name} must contain one value per query")
    if not torch.isfinite(targets).all() or torch.any((targets != 0) & (targets != 1)):
        raise ValueError(f"{name} must contain finite binary labels")
