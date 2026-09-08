import hashlib
from datetime import UTC, datetime

import pytest
import torch

from hermes_local.grounding import (
    HermesEvidence,
    HermesGroundedService,
    HermesGroundingModule,
    HermesWorkspaceRuntime,
)
from hermes_local.substrate import (
    AbstainOutput,
    EvidenceItem,
    ModelIdentity,
    TextOutput,
    WorkspaceState,
)


def test_hermes_returns_exact_citation_and_only_proposes_memory() -> None:
    text = "Unrelated preface. Olympus stores approved facts with their source identifiers."
    evidence = HermesEvidence.from_text("doc-1", text, "file:///policy.txt")

    result = HermesGroundedService(minimum_support=0.6).answer(
        "How does Olympus store approved facts and source identifiers?",
        [evidence],
        propose_memory=True,
    )

    assert not result.abstained
    assert result.answer == "Olympus stores approved facts with their source identifiers."
    assert result.citations[0].quote == text[
        result.citations[0].span_start : result.citations[0].span_end
    ]
    assert result.citations[0].content_hash == evidence.content_hash
    assert len(result.memory_write_proposals) == 1
    assert result.memory_write_proposals[0].requires_approval


def test_hermes_abstains_on_unsupported_or_unauthorized_evidence() -> None:
    protected = HermesEvidence.from_text(
        "secret",
        "The launch code is helios seven.",
        "vault:///launch",
        acl_labels=("mission-control",),
    )
    service = HermesGroundedService(minimum_support=0.5)

    unauthorized = service.answer("What is the launch code?", [protected])
    unsupported = service.answer(
        "What is the launch code?",
        [HermesEvidence.from_text("public", "Bird migration is seasonal.", "file:///birds")],
    )

    assert unauthorized.abstention_reason == "no_authorized_evidence"
    assert unsupported.abstention_reason == "insufficient_claim_support"
    assert unauthorized.citations == ()
    assert unsupported.memory_write_proposals == ()


def test_hermes_trainable_heads_receive_gradient_and_update() -> None:
    torch.manual_seed(4)
    model = HermesGroundingModule(vocab_size=32, hidden_size=12)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    attribution_head = model.attribution_head[0]
    assert isinstance(attribution_head, torch.nn.Linear)
    before = attribution_head.weight.detach().clone()

    output = model(
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        torch.tensor([[[1, 2, 0], [8, 9, 0]], [[4, 6, 0], [10, 11, 12]]]),
        attribution_targets=torch.tensor([[0, 2], [0, 1]]),
        correctness_targets=torch.tensor([1.0, 0.0]),
        memory_targets=torch.tensor([1.0, 0.0]),
    )
    loss = output["loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    torch.autograd.backward(loss)
    optimizer.step()

    assert not torch.equal(before, attribution_head.weight.detach())


def test_hermes_rejects_all_padding_and_misaligned_targets() -> None:
    model = HermesGroundingModule(vocab_size=16, hidden_size=8)
    query = torch.tensor([[1, 2]])
    evidence = torch.zeros((1, 2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="evidence item"):
        model(query, evidence, attribution_targets=torch.tensor([[0, 1]]))

    evidence[0, 0, 0] = 3
    with pytest.raises(ValueError, match="attribution targets"):
        model(query, evidence, attribution_targets=torch.tensor([[0]]))
    with pytest.raises(ValueError, match="integer tensor"):
        model(query, evidence, attribution_targets=torch.tensor([[0.1, 1.9]]))


def test_hermes_workspace_runtime_returns_a_bound_validated_envelope() -> None:
    text = "Mars has two moons."
    workspace = WorkspaceState(
        workspace_id="hermes-runtime-test",
        objective="Answer from the supplied evidence.",
        evidence=[
            EvidenceItem(
                evidence_id="mars",
                source_uri="urn:test:mars",
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
                license_id="test-only",
                acl_labels={"public"},
                text=text,
            )
        ],
        model_identity=ModelIdentity(
            model_id="hermes-runtime",
            family="Hermes",
            version="test",
            base_model_id="local/test",
            base_revision="test",
            base_sha256="0" * 64,
        ),
    )
    runtime = HermesWorkspaceRuntime(HermesGroundedService(minimum_support=0.5))
    answer = runtime.respond(workspace, "Mars moons")
    abstention = runtime.respond(workspace, "unrelated quantum question")

    assert isinstance(answer, TextOutput)
    assert answer.citation_evidence_ids == ["mars"]
    assert answer.workspace_sha256 == workspace.sha256
    assert isinstance(abstention, AbstainOutput)
    protocol_restricted = runtime.respond(
        workspace,
        "Mars moons",
        allowed_evidence_ids={"different-evidence"},
    )
    assert isinstance(protocol_restricted, AbstainOutput)
    assert protocol_restricted.reason == "no_authorized_evidence"

    execution_failure = runtime.respond_after_execution(
        workspace,
        "Mars moons",
        execution_succeeded=False,
        failure_code="compensatable_execution_failure",
    )
    assert isinstance(execution_failure, AbstainOutput)
    assert (
        execution_failure.reason
        == "upstream_execution_failed:compensatable_execution_failure"
    )
    upstream_failure = runtime.respond_after_execution(
        workspace,
        "Mars moons",
        execution_succeeded=True,
        upstream_component_failure="kronos",
        allowed_evidence_ids={"mars"},
    )
    assert isinstance(upstream_failure, AbstainOutput)
    assert upstream_failure.reason == "upstream_component_failed:kronos"
    with pytest.raises(ValueError, match="successful execution"):
        runtime.respond_after_execution(
            workspace,
            "Mars moons",
            execution_succeeded=True,
            failure_code="impossible",
        )
