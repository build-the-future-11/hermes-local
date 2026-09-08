"""Shared typed evidence/action substrate and adapter-capable decoder.

The contracts in this module are enforcement inputs for family components. A
valid envelope is a proposal bound to a workspace and model identity; it is not
authorization to mutate external state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Sequence
from datetime import datetime
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Self, cast

import torch
from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from hermes_local.core import ComputeBudget, StrictModel

_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256 = r"^[0-9a-f]{64}$"
_ADAPTER_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ModelLifecycle(StrEnum):
    SPECIFIED = "specified"
    IMPLEMENTED = "implemented"
    EXPERIMENTAL = "experimental"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class PlanStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ModelIdentity(StrictModel):
    model_id: str = Field(pattern=_IDENTIFIER)
    family: str = Field(pattern=_IDENTIFIER)
    version: str = Field(min_length=1, max_length=128)
    base_model_id: str = Field(min_length=1, max_length=500)
    base_revision: str = Field(min_length=1, max_length=200)
    base_sha256: str = Field(pattern=_SHA256)
    checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256)
    lifecycle: ModelLifecycle = ModelLifecycle.SPECIFIED

    @model_validator(mode="after")
    def promoted_identity_requires_checkpoint(self) -> Self:
        if self.lifecycle in {ModelLifecycle.EVALUATED, ModelLifecycle.PROMOTED}:
            if self.checkpoint_sha256 is None:
                raise ValueError("evaluated or promoted models require a checkpoint hash")
        return self


class EvidenceItem(StrictModel):
    evidence_id: str = Field(pattern=_IDENTIFIER)
    source_uri: str = Field(min_length=1, max_length=4_000)
    content_sha256: str = Field(pattern=_SHA256)
    acquired_at: datetime
    license_id: str = Field(min_length=1, max_length=200)
    acl_labels: set[str] = Field(default_factory=set)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    text: str | None = Field(default=None, max_length=100_000)

    @model_validator(mode="after")
    def validate_span_and_content(self) -> Self:
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("evidence spans require both start and end")
        if self.span_start is not None and self.span_end is not None:
            if self.span_end <= self.span_start:
                raise ValueError("evidence span end must be greater than start")
            if self.text is not None and self.span_end > len(self.text):
                raise ValueError("evidence span exceeds provided text")
        if self.text is not None:
            digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if digest != self.content_sha256:
                raise ValueError("evidence text does not match its content hash")
        return self


class Claim(StrictModel):
    claim_id: str = Field(pattern=_IDENTIFIER)
    text: str = Field(min_length=1, max_length=20_000)
    status: ClaimStatus = ClaimStatus.PROPOSED
    support_evidence_ids: list[str] = Field(default_factory=list)
    refute_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def evidence_sets_do_not_overlap(self) -> Self:
        overlap = set(self.support_evidence_ids) & set(self.refute_evidence_ids)
        if overlap:
            raise ValueError(f"evidence cannot both support and refute a claim: {sorted(overlap)}")
        if self.status is ClaimStatus.SUPPORTED and not self.support_evidence_ids:
            raise ValueError("supported claims require supporting evidence")
        if self.status is ClaimStatus.REFUTED and not self.refute_evidence_ids:
            raise ValueError("refuted claims require refuting evidence")
        return self


class PlanStep(StrictModel):
    step_id: str = Field(pattern=_IDENTIFIER)
    description: str = Field(min_length=1, max_length=4_000)
    depends_on: list[str] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    tool_id: str | None = Field(default=None, pattern=_IDENTIFIER)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)


class ToolSpec(StrictModel):
    tool_id: str = Field(pattern=_IDENTIFIER)
    schema_version: str = Field(min_length=1, max_length=128)
    argument_schema: dict[str, Any]
    required_capabilities: set[str] = Field(default_factory=set)
    material_effect: bool = False
    approval_required: bool = False

    @model_validator(mode="after")
    def validate_argument_schema_subset(self) -> ToolSpec:
        if self.material_effect and not self.approval_required:
            raise ValueError("material-effect tools must require explicit approval")
        if not self.argument_schema:
            raise ValueError("tool argument schema must explicitly declare an object schema")
        _validate_schema_definition(self.argument_schema, path="$", root=True)
        return self


class ToolInvocation(StrictModel):
    tool_id: str = Field(pattern=_IDENTIFIER)
    schema_version: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
    approval_id: str | None = Field(default=None, pattern=_IDENTIFIER)

    @model_validator(mode="after")
    def arguments_are_strict_json(self) -> ToolInvocation:
        ensure_finite_json(self.arguments, label="tool arguments")
        return self


class PermissionSet(StrictModel):
    granted_acl_labels: set[str] = Field(default_factory=set)
    allowed_tool_ids: set[str] = Field(default_factory=set)
    capabilities: set[str] = Field(default_factory=set)
    approval_bindings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_approval_bindings(self) -> PermissionSet:
        for approval_id, invocation_sha256 in self.approval_bindings.items():
            if not re.fullmatch(_IDENTIFIER, approval_id):
                raise ValueError("approval binding IDs must use the shared identifier format")
            if not re.fullmatch(_SHA256, invocation_sha256):
                raise ValueError("approval bindings must contain lowercase SHA-256 values")
        return self


class WorkspaceEvent(StrictModel):
    event_id: str = Field(pattern=_IDENTIFIER)
    sequence: int = Field(ge=1)
    kind: str = Field(pattern=_IDENTIFIER)
    actor: str = Field(min_length=1, max_length=200)
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_sha256: str = Field(pattern=_SHA256)
    previous_event_sha256: str = Field(pattern=_SHA256)
    event_sha256: str = Field(pattern=_SHA256)

    @classmethod
    def from_payload(
        cls,
        *,
        event_id: str,
        sequence: int,
        kind: str,
        actor: str,
        occurred_at: datetime,
        payload: dict[str, Any],
        previous_event_sha256: str,
    ) -> WorkspaceEvent:
        payload_sha256 = canonical_sha256(payload)
        core = {
            "event_id": event_id,
            "sequence": sequence,
            "kind": kind,
            "actor": actor,
            "occurred_at": occurred_at,
            "payload": payload,
            "payload_sha256": payload_sha256,
            "previous_event_sha256": previous_event_sha256,
        }
        return cls(
            event_id=event_id,
            sequence=sequence,
            kind=kind,
            actor=actor,
            occurred_at=occurred_at,
            payload=payload,
            payload_sha256=payload_sha256,
            previous_event_sha256=previous_event_sha256,
            event_sha256=canonical_sha256(core),
        )

    @model_validator(mode="after")
    def verify_event_hashes(self) -> WorkspaceEvent:
        if self.occurred_at.utcoffset() is None:
            raise ValueError("workspace event timestamps must be timezone-aware")
        ensure_finite_json(self.payload, label="workspace event payload")
        if canonical_sha256(self.payload) != self.payload_sha256:
            raise ValueError("workspace event payload hash is invalid")
        core = self.model_dump(exclude={"event_sha256"})
        if canonical_sha256(core) != self.event_sha256:
            raise ValueError("workspace event hash is invalid")
        return self


class WorkspaceState(StrictModel):
    workspace_id: str = Field(pattern=_IDENTIFIER)
    objective: str = Field(min_length=1, max_length=20_000)
    constraints: list[str] = Field(default_factory=list)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    permissions: PermissionSet = Field(default_factory=PermissionSet)
    events: list[WorkspaceEvent] = Field(default_factory=list)
    model_identity: ModelIdentity
    event_sequence: int = Field(default=0, ge=0)
    parent_state_sha256: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def enforce_referential_integrity(self) -> Self:
        _require_unique("evidence", [item.evidence_id for item in self.evidence])
        _require_unique("claim", [item.claim_id for item in self.claims])
        _require_unique("plan step", [item.step_id for item in self.plan])
        _require_unique("tool", [f"{item.tool_id}:{item.schema_version}" for item in self.tools])
        _require_unique("workspace event", [item.event_id for item in self.events])

        evidence_ids = {item.evidence_id for item in self.evidence}
        for claim in self.claims:
            referenced = set(claim.support_evidence_ids) | set(claim.refute_evidence_ids)
            missing = referenced - evidence_ids
            if missing:
                raise ValueError(f"claim references unknown evidence: {sorted(missing)}")

        plan_ids = {item.step_id for item in self.plan}
        graph: dict[str, set[str]] = {}
        tool_ids = {item.tool_id for item in self.tools}
        for step in self.plan:
            missing = set(step.depends_on) - plan_ids
            if missing:
                raise ValueError(f"plan step references unknown dependencies: {sorted(missing)}")
            if step.step_id in step.depends_on:
                raise ValueError("plan steps cannot depend on themselves")
            if step.tool_id is not None and step.tool_id not in tool_ids:
                raise ValueError("plan step references an undeclared tool")
            graph[step.step_id] = set(step.depends_on)
        _assert_acyclic(graph)
        if self.event_sequence != len(self.events):
            raise ValueError("event_sequence must equal the hash-linked event count")
        previous = "0" * 64
        previous_time: datetime | None = None
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        claims_by_id = {item.claim_id: item for item in self.claims}
        plan_by_id = {item.step_id: item for item in self.plan}
        seen_added_evidence: set[str] = set()
        seen_added_claims: set[str] = set()
        seen_added_plan: set[str] = set()
        for sequence, event in enumerate(self.events, start=1):
            if event.sequence != sequence:
                raise ValueError("workspace event sequence must be contiguous")
            if event.previous_event_sha256 != previous:
                raise ValueError("workspace event hash chain is discontinuous")
            if previous_time is not None and event.occurred_at < previous_time:
                raise ValueError("workspace event timestamps must be monotonic")
            if event.payload.get("transition_schema") == "workspace.append.v1":
                _verify_workspace_append_event(
                    event,
                    evidence_by_id=evidence_by_id,
                    claims_by_id=claims_by_id,
                    plan_by_id=plan_by_id,
                    seen_added_evidence=seen_added_evidence,
                    seen_added_claims=seen_added_claims,
                    seen_added_plan=seen_added_plan,
                )
            previous = event.event_sha256
            previous_time = event.occurred_at
        return self

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)


class AuthorizedWorkspaceView(StrictModel):
    """Minimum model-input projection with inaccessible evidence removed.

    The source workspace hash is intentionally absent: a commitment over hidden,
    low-entropy plaintext can itself become a guessing oracle. Trusted runtime
    code, rather than learned code, binds outputs back to the source workspace.
    """

    workspace_id: str = Field(pattern=_IDENTIFIER)
    objective: str = Field(min_length=1, max_length=20_000)
    constraints: list[str] = Field(default_factory=list)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    model_identity: ModelIdentity

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)


def authorized_workspace_view(workspace: WorkspaceState) -> AuthorizedWorkspaceView:
    """Project the trusted workspace before any model tokenization or scoring."""

    workspace = validate_workspace_snapshot(workspace)
    evidence = [
        item.model_copy(deep=True)
        for item in workspace.evidence
        if evidence_is_authorized(item, workspace.permissions)
    ]
    tools = [
        item.model_copy(deep=True)
        for item in workspace.tools
        if item.tool_id in workspace.permissions.allowed_tool_ids
        and item.required_capabilities.issubset(workspace.permissions.capabilities)
    ]
    return AuthorizedWorkspaceView(
        workspace_id=workspace.workspace_id,
        objective=workspace.objective,
        constraints=list(workspace.constraints),
        budget=workspace.budget.model_copy(deep=True),
        evidence=evidence,
        tools=tools,
        model_identity=workspace.model_identity.model_copy(deep=True),
    )


def validate_workspace_snapshot(workspace: WorkspaceState) -> WorkspaceState:
    """Deeply revalidate mutable containers at every authority/runtime boundary."""

    return WorkspaceState.model_validate(workspace.model_dump(mode="python"))


def append_workspace_event(
    workspace: WorkspaceState,
    *,
    event_id: str,
    kind: str,
    actor: str,
    occurred_at: datetime,
    payload: dict[str, Any] | None = None,
    evidence: Sequence[EvidenceItem] = (),
    claims: Sequence[Claim] = (),
    plan_steps: Sequence[PlanStep] = (),
) -> WorkspaceState:
    """Return a new hash-linked workspace snapshot with append-only additions.

    This is a narrow, process-local transition helper. It never mutates the
    supplied snapshot and deliberately cannot replace or delete authority,
    evidence, claims, plans, or tools. Callers that need those operations must
    define a separately reviewed transition protocol.
    """

    snapshot = validate_workspace_snapshot(workspace)
    added_evidence = [
        EvidenceItem.model_validate(item.model_dump(mode="python")) for item in evidence
    ]
    added_claims = [Claim.model_validate(item.model_dump(mode="python")) for item in claims]
    added_plan = [
        PlanStep.model_validate(item.model_dump(mode="python")) for item in plan_steps
    ]
    event_payload = {} if payload is None else payload
    ensure_finite_json(event_payload, label="workspace transition payload")
    # A JSON round trip both snapshots caller-owned containers and preserves the
    # exact finite-JSON value whose digest is put into the event chain.
    payload_snapshot = cast(
        dict[str, Any],
        json.loads(
            json.dumps(
                event_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        ),
    )
    additions = {
        "evidence": [item.model_dump(mode="python") for item in added_evidence],
        "claims": [item.model_dump(mode="python") for item in added_claims],
        "plan_steps": [item.model_dump(mode="python") for item in added_plan],
    }
    transition_payload = {
        "transition_schema": "workspace.append.v1",
        "data": payload_snapshot,
        "added_evidence_ids": [item.evidence_id for item in added_evidence],
        "added_claim_ids": [item.claim_id for item in added_claims],
        "added_plan_step_ids": [item.step_id for item in added_plan],
        "additions_sha256": canonical_sha256(additions),
    }
    previous_event_sha256 = (
        snapshot.events[-1].event_sha256 if snapshot.events else "0" * 64
    )
    event = WorkspaceEvent.from_payload(
        event_id=event_id,
        sequence=snapshot.event_sequence + 1,
        kind=kind,
        actor=actor,
        occurred_at=occurred_at,
        payload=transition_payload,
        previous_event_sha256=previous_event_sha256,
    )
    updated = snapshot.model_dump(mode="python")
    updated.update(
        {
            "evidence": [*snapshot.evidence, *added_evidence],
            "claims": [*snapshot.claims, *added_claims],
            "plan": [*snapshot.plan, *added_plan],
            "events": [*snapshot.events, event],
            "event_sequence": snapshot.event_sequence + 1,
            "parent_state_sha256": snapshot.sha256,
        }
    )
    return WorkspaceState.model_validate(updated)


def _verify_workspace_append_event(
    event: WorkspaceEvent,
    *,
    evidence_by_id: dict[str, EvidenceItem],
    claims_by_id: dict[str, Claim],
    plan_by_id: dict[str, PlanStep],
    seen_added_evidence: set[str],
    seen_added_claims: set[str],
    seen_added_plan: set[str],
) -> None:
    expected_keys = {
        "transition_schema",
        "data",
        "added_evidence_ids",
        "added_claim_ids",
        "added_plan_step_ids",
        "additions_sha256",
    }
    if set(event.payload) != expected_keys or not isinstance(event.payload["data"], dict):
        raise ValueError("workspace append event has an invalid transition payload")

    def identifiers(key: str) -> list[str]:
        value = event.payload[key]
        if (
            not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
            or len(value) != len(set(value))
        ):
            raise ValueError("workspace append event IDs must be unique string lists")
        return value

    evidence_ids = identifiers("added_evidence_ids")
    claim_ids = identifiers("added_claim_ids")
    plan_ids = identifiers("added_plan_step_ids")
    if seen_added_evidence.intersection(evidence_ids):
        raise ValueError("workspace evidence was appended more than once")
    if seen_added_claims.intersection(claim_ids):
        raise ValueError("workspace claim was appended more than once")
    if seen_added_plan.intersection(plan_ids):
        raise ValueError("workspace plan step was appended more than once")
    try:
        additions = {
            "evidence": [
                evidence_by_id[item].model_dump(mode="python") for item in evidence_ids
            ],
            "claims": [claims_by_id[item].model_dump(mode="python") for item in claim_ids],
            "plan_steps": [plan_by_id[item].model_dump(mode="python") for item in plan_ids],
        }
    except KeyError as error:
        raise ValueError("workspace append event references a missing added object") from error
    additions_sha256 = event.payload["additions_sha256"]
    if not isinstance(additions_sha256, str) or canonical_sha256(additions) != additions_sha256:
        raise ValueError("workspace append event additions hash is invalid")
    seen_added_evidence.update(evidence_ids)
    seen_added_claims.update(claim_ids)
    seen_added_plan.update(plan_ids)


class BoundEnvelope(StrictModel):
    workspace_sha256: str = Field(pattern=_SHA256)
    model_identity_sha256: str = Field(pattern=_SHA256)


class TextOutput(BoundEnvelope):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=100_000)
    citation_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimOutput(BoundEnvelope):
    kind: Literal["claim"] = "claim"
    claims: list[Claim] = Field(min_length=1, max_length=1_000)


class ActionOutput(BoundEnvelope):
    kind: Literal["action"] = "action"
    invocation: ToolInvocation


class AbstainOutput(BoundEnvelope):
    kind: Literal["abstain"] = "abstain"
    reason: str = Field(min_length=1, max_length=10_000)
    unresolved: list[str] = Field(default_factory=list)
    required_authority: str | None = Field(default=None, max_length=500)


OutputEnvelope = Annotated[
    TextOutput | ClaimOutput | ActionOutput | AbstainOutput,
    Field(discriminator="kind"),
]
_OUTPUT_ADAPTER: TypeAdapter[OutputEnvelope] = TypeAdapter(OutputEnvelope)


def canonical_json_bytes(value: Any) -> bytes:
    payload = _canonical_value(value)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def ensure_finite_json(value: Any, *, label: str = "value") -> None:
    """Reject Python-only values and ambiguous/non-finite JSON structures."""

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} JSON object keys must be strings")
            ensure_finite_json(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            ensure_finite_json(item, label=f"{label}[{index}]")
        return
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{label} must contain only finite JSON values")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tool_invocation_sha256(invocation: ToolInvocation) -> str:
    """Hash the immutable action intent without its separately supplied approval ID."""

    return canonical_sha256(invocation.model_dump(exclude={"approval_id"}))


def parse_output_envelope(payload: str | bytes | dict[str, Any]) -> OutputEnvelope:
    if isinstance(payload, bytes):
        return _OUTPUT_ADAPTER.validate_json(payload)
    if isinstance(payload, str):
        return _OUTPUT_ADAPTER.validate_json(payload)
    return _OUTPUT_ADAPTER.validate_python(payload)


def validate_envelope_against_workspace(
    envelope: OutputEnvelope, workspace: WorkspaceState
) -> None:
    # Pydantic assignment validation cannot observe in-place mutation of a
    # nested list/dict. Reconstruct the discriminated union before treating an
    # already-instantiated envelope as an authority-bound object.
    envelope = _OUTPUT_ADAPTER.validate_python(envelope.model_dump(mode="python"))
    workspace = validate_workspace_snapshot(workspace)
    if envelope.workspace_sha256 != workspace.sha256:
        raise ValueError("output is not bound to the active workspace")
    expected_model = canonical_sha256(workspace.model_identity)
    if envelope.model_identity_sha256 != expected_model:
        raise ValueError("output is not bound to the active model identity")

    evidence_by_id = {item.evidence_id: item for item in workspace.evidence}
    if isinstance(envelope, TextOutput):
        _validate_evidence_access(
            envelope.citation_evidence_ids, evidence_by_id, workspace.permissions
        )
    elif isinstance(envelope, ClaimOutput):
        for claim in envelope.claims:
            _validate_evidence_access(
                [*claim.support_evidence_ids, *claim.refute_evidence_ids],
                evidence_by_id,
                workspace.permissions,
            )
    elif isinstance(envelope, ActionOutput):
        invocation = envelope.invocation
        matching = [
            tool
            for tool in workspace.tools
            if tool.tool_id == invocation.tool_id
            and tool.schema_version == invocation.schema_version
        ]
        if not matching:
            raise ValueError("action references an undeclared tool or schema version")
        tool = matching[0]
        if tool.tool_id not in workspace.permissions.allowed_tool_ids:
            raise PermissionError("tool is not allowed by the workspace permission set")
        missing = tool.required_capabilities - workspace.permissions.capabilities
        if missing:
            raise PermissionError(f"missing tool capabilities: {sorted(missing)}")
        _validate_json_object(invocation.arguments, tool.argument_schema)
        if tool.approval_required:
            approval_id = invocation.approval_id
            expected = tool_invocation_sha256(invocation)
            if (
                approval_id is None
                or workspace.permissions.approval_bindings.get(approval_id) != expected
            ):
                raise PermissionError(
                    "tool invocation requires an action-bound granted approval"
                )


def _validate_evidence_access(
    references: list[str],
    evidence_by_id: dict[str, EvidenceItem],
    permissions: PermissionSet,
) -> None:
    missing = set(references) - evidence_by_id.keys()
    if missing:
        raise ValueError(f"output references unknown evidence: {sorted(missing)}")
    for reference in references:
        if not evidence_is_authorized(evidence_by_id[reference], permissions):
            raise PermissionError(f"output cites unauthorized evidence: {reference}")


def evidence_is_authorized(item: EvidenceItem, permissions: PermissionSet) -> bool:
    """Apply the one shared ACL rule used at model-family boundaries."""

    return acl_labels_are_authorized(item.acl_labels, permissions.granted_acl_labels)


def acl_labels_are_authorized(
    required_labels: Collection[str], granted_labels: Collection[str]
) -> bool:
    required = set(required_labels)
    return not required or required.issubset(set(granted_labels) | {"public"})


_JSON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}
_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
}


def _validate_schema_definition(
    schema: dict[str, Any], *, path: str, root: bool = False, depth: int = 0
) -> None:
    if not schema:
        raise ValueError(f"schema at {path} cannot be empty")
    if depth > 16:
        raise ValueError("tool argument schema exceeds the supported nesting depth")
    unknown = schema.keys() - _SCHEMA_KEYS
    if unknown:
        raise ValueError(f"unsupported schema keywords at {path}: {sorted(unknown)}")
    kind = schema.get("type")
    if kind not in _JSON_TYPES:
        raise ValueError(f"schema at {path} requires one supported explicit type")
    if root and kind != "object":
        raise ValueError("tool argument schema root type must be object")
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise ValueError(f"schema enum at {path} must be a non-empty list")
        for option in enum:
            if not _matches_json_type(option, kind):
                raise ValueError(f"schema enum value at {path} does not match {kind}")
            try:
                canonical_json_bytes(option)
            except (TypeError, ValueError) as error:
                raise ValueError(f"schema enum at {path} must contain finite JSON") from error
    for keyword in ("minimum", "maximum"):
        if keyword in schema:
            bound = schema[keyword]
            if kind not in {"integer", "number"}:
                raise ValueError(f"{keyword} at {path} requires a numeric type")
            if (
                isinstance(bound, bool)
                or not isinstance(bound, int | float)
                or not math.isfinite(float(bound))
            ):
                raise ValueError(f"{keyword} at {path} must be finite")
    if "minimum" in schema and "maximum" in schema:
        if schema["minimum"] > schema["maximum"]:
            raise ValueError(f"minimum exceeds maximum at {path}")
    for keyword in ("minLength", "maxLength"):
        if keyword in schema:
            if kind != "string" or isinstance(schema[keyword], bool):
                raise ValueError(f"{keyword} at {path} requires a string type")
            if not isinstance(schema[keyword], int) or schema[keyword] < 0:
                raise ValueError(f"{keyword} at {path} must be a non-negative integer")
    if "minLength" in schema and "maxLength" in schema:
        if schema["minLength"] > schema["maxLength"]:
            raise ValueError(f"minLength exceeds maxLength at {path}")

    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", False)
        if not isinstance(properties, dict):
            raise ValueError(f"schema properties at {path} must be an object")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError(f"schema required at {path} must be a string list")
        if len(required) != len(set(required)) or not set(required).issubset(properties):
            raise ValueError(f"schema required entries at {path} must uniquely name properties")
        if not isinstance(additional, bool):
            raise ValueError(f"additionalProperties at {path} must be boolean")
        for name, nested in properties.items():
            if not isinstance(name, str) or not isinstance(nested, dict):
                raise ValueError(f"schema properties at {path} must map strings to schemas")
            _validate_schema_definition(
                nested, path=f"{path}.{name}", depth=depth + 1
            )
    elif any(key in schema for key in ("properties", "required", "additionalProperties")):
        raise ValueError(f"object-only schema keyword used at non-object path {path}")

    if kind == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            raise ValueError(f"array schema at {path} requires one items schema")
        _validate_schema_definition(items, path=f"{path}[]", depth=depth + 1)
    elif "items" in schema:
        raise ValueError(f"items schema used at non-array path {path}")


def _validate_json_object(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    _validate_json_value(arguments, schema, path="arguments")
    canonical_json_bytes(arguments)


def _validate_json_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    kind = schema["type"]
    if not _matches_json_type(value, kind):
        raise ValueError(f"{path} has the wrong type")
    if kind == "number" and not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed enum value")
    if kind in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} exceeds its maximum")
    if kind == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is longer than maxLength")
    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{path} is missing required properties: {sorted(missing)}")
        if schema.get("additionalProperties", False) is False:
            unknown = value.keys() - properties.keys()
            if unknown:
                raise ValueError(f"{path} contains unknown properties: {sorted(unknown)}")
        for name, item in value.items():
            nested = properties.get(name)
            if nested is not None:
                _validate_json_value(item, nested, path=f"{path}.{name}")
    if kind == "array":
        for index, item in enumerate(value):
            _validate_json_value(item, schema["items"], path=f"{path}[{index}]")


def _matches_json_type(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, _JSON_TYPES[kind])


def _canonical_value(value: Any) -> Any:
    if isinstance(value, StrictModel):
        return _canonical_value(value.model_dump(mode="python", exclude_none=False))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON objects require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def _require_unique(label: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} identifiers must be unique")


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("plan dependencies must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


class DecoderConfig(StrictModel):
    vocab_size: int = Field(default=260, ge=8, le=1_000_000)
    width: int = Field(default=64, ge=8, le=8_192)
    layers: int = Field(default=2, ge=1, le=128)
    heads: int = Field(default=4, ge=1, le=128)
    max_sequence_tokens: int = Field(default=256, ge=8, le=1_000_000)
    adapter_rank: int = Field(default=16, ge=1, le=2_048)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def width_matches_heads(self) -> Self:
        if self.width % self.heads:
            raise ValueError("decoder width must be divisible by attention heads")
        if self.adapter_rank > self.width:
            raise ValueError("adapter rank cannot exceed decoder width")
        return self


class BottleneckAdapter(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.scale = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, self.scale * self.up(F.gelu(self.down(hidden))))


class AdapterDecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.width * 4),
            nn.GELU(),
            nn.Linear(config.width * 4, config.width),
        )
        self.adapters = nn.ModuleDict()

    def add_adapter(self, name: str, width: int, rank: int) -> None:
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        self.adapters[name] = BottleneckAdapter(width, rank)

    def forward(
        self,
        hidden: Tensor,
        *,
        causal_mask: Tensor,
        key_padding_mask: Tensor | None,
        adapter_name: str | None,
    ) -> Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        if adapter_name is not None:
            hidden = hidden + self.adapters[adapter_name](hidden)
        return hidden


class DecoderOutput(StrictModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    logits: Tensor
    hidden_states: Tensor
    pooled_state: Tensor


class AdapterCausalDecoder(nn.Module):
    """Compact decoder with independently selectable residual adapters."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence_tokens, config.width)
        self.blocks = nn.ModuleList(AdapterDecoderBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.active_adapter: str | None = None
        self._base_frozen = False
        self._train_active_adapter = False

    def add_adapter(self, name: str) -> None:
        if not _ADAPTER_NAME.fullmatch(name):
            raise ValueError("adapter name must match ^[a-z][a-z0-9_-]{0,63}$")
        for block in self.blocks:
            typed_block = cast(AdapterDecoderBlock, block)
            typed_block.add_adapter(name, self.config.width, self.config.adapter_rank)
        if self._base_frozen:
            self._sync_adapter_trainability()

    def activate_adapter(self, name: str | None) -> None:
        if name is not None:
            for block in self.blocks:
                if name not in cast(AdapterDecoderBlock, block).adapters:
                    raise KeyError(f"unknown adapter: {name}")
        self.active_adapter = name
        if self._base_frozen:
            self._sync_adapter_trainability()

    def freeze_base(self, *, train_active_adapter: bool = True) -> None:
        self._base_frozen = True
        self._train_active_adapter = train_active_adapter
        for parameter in self.parameters():
            parameter.requires_grad = False
        self._sync_adapter_trainability()

    def _sync_adapter_trainability(self) -> None:
        for block in self.blocks:
            typed_block = cast(AdapterDecoderBlock, block)
            for adapter_name, adapter in typed_block.adapters.items():
                trainable = (
                    self._train_active_adapter and adapter_name == self.active_adapter
                )
                for parameter in adapter.parameters():
                    parameter.requires_grad = trainable

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def adapter_state_dict(self, name: str) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for index, block in enumerate(self.blocks):
            typed_block = cast(AdapterDecoderBlock, block)
            if name not in typed_block.adapters:
                raise KeyError(f"unknown adapter: {name}")
            for key, value in typed_block.adapters[name].state_dict().items():
                state[f"blocks.{index}.adapters.{name}.{key}"] = (
                    value.detach().cpu().clone()
                )
        return state

    def load_adapter_state_dict(self, name: str, state: dict[str, Tensor]) -> None:
        for block in self.blocks:
            if name not in cast(AdapterDecoderBlock, block).adapters:
                raise KeyError(f"unknown adapter: {name}")
        expected = self.adapter_state_dict(name)
        if set(state) != set(expected):
            raise ValueError("adapter state keys do not match the target adapter")
        for index, block in enumerate(self.blocks):
            typed_block = cast(AdapterDecoderBlock, block)
            prefix = f"blocks.{index}.adapters.{name}."
            local = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            typed_block.adapters[name].load_state_dict(local, strict=True)

    def forward(
        self, token_ids: Tensor, attention_mask: Tensor | None = None
    ) -> DecoderOutput:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one sequence and token")
        if token_ids.shape[1] > self.config.max_sequence_tokens:
            raise ValueError("input exceeds decoder sequence limit")
        if token_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("token_ids must use an integer tensor dtype")
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        elif attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        else:
            attention_mask = attention_mask.to(torch.bool)
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("each sequence requires at least one unmasked token")
        if token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError("token IDs are outside the configured vocabulary")

        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(
                token_ids.shape[1],
                token_ids.shape[1],
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask
        for block in self.blocks:
            typed_block = cast(AdapterDecoderBlock, block)
            hidden = typed_block(
                hidden,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                adapter_name=self.active_adapter,
            )
        hidden = self.output_norm(hidden)
        token_positions = torch.arange(hidden.shape[1], device=hidden.device)
        last_indices = torch.where(attention_mask, token_positions, -1).max(dim=1).values
        pooled = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), last_indices
        ]
        return DecoderOutput(
            logits=cast(Tensor, self.lm_head(hidden)),
            hidden_states=hidden,
            pooled_state=pooled,
        )


def causal_language_model_loss(
    output: DecoderOutput, targets: Tensor, attention_mask: Tensor | None = None
) -> Tensor:
    if targets.shape != output.logits.shape[:2]:
        raise ValueError("language-model targets must match output token dimensions")
    if targets.shape[1] < 2:
        raise ValueError("language-model loss requires at least two tokens")
    shift_logits = output.logits[:, :-1].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    if attention_mask is not None:
        if attention_mask.shape != targets.shape:
            raise ValueError("loss attention mask must match targets")
        shift_targets = shift_targets.masked_fill(~attention_mask[:, 1:].to(torch.bool), -100)
        if not torch.any(shift_targets != -100):
            raise ValueError("language-model loss requires at least one supervised token")
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_targets.reshape(-1)
    )
