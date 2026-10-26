# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import (
    DuplexContextLedger,
    DuplexContextLevel,
    DuplexContextNode,
    DuplexContextNodeRejectedError,
    source_refs,
)
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

L1 = DuplexContextLevel.MEDIA_DROPPED
SUMMARY = DuplexContextLevel.SUMMARY


@pytest.fixture
def session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="ledger-test", config=DuplexSessionConfig(model="test-model"))


def _user(text: str) -> dict[str, object]:
    return {"role": "user", "content": text}


def _l1(ledger: DuplexContextLedger, snapshot: DuplexHistorySnapshot, index: int) -> DuplexContextNode:
    return ledger.add_node(
        snapshot,
        level=L1,
        sources=source_refs(snapshot, snapshot.source_ids[index]),
        message=dict(snapshot.messages[index]),
    )


def _summary(
    ledger: DuplexContextLedger,
    snapshot: DuplexHistorySnapshot,
    *parents: DuplexContextNode,
    source_index: int | None = None,
) -> DuplexContextNode:
    sources = () if source_index is None else source_refs(snapshot, snapshot.source_ids[source_index])
    return ledger.add_node(
        snapshot,
        level=SUMMARY,
        sources=sources,
        parents=tuple(parent.node_id for parent in parents),
        message={"role": "system", "content": "summary"},
    )


def test_node_valid_while_source_revisions_match(session):
    session.register_history_item("u1", _user("hi"))
    ledger = DuplexContextLedger()
    node = _l1(ledger, session.history_snapshot(), 0)

    assert ledger.refresh(session.history_snapshot()) == ()
    assert ledger.valid(node.node_id)
    assert ledger.nodes() == (node,)


def test_l1_batch_uses_snapshot_revisions_and_preserves_order(session):
    session.register_history_item("u1", _user("one"))
    session.register_history_item("u2", _user("two"))
    snapshot = session.history_snapshot()
    first, second = snapshot.source_ids
    ledger = DuplexContextLedger()

    nodes = ledger.add_l1_nodes(snapshot, messages={second: _user("derived two"), first: _user("derived one")})

    assert [node.sources for node in nodes] == [source_refs(snapshot, second), source_refs(snapshot, first)]
    assert [node.message["content"] for node in nodes] == ["derived two", "derived one"]
    assert [node.node_id for node in nodes] == [1, 2]
    assert ledger.refresh(snapshot) == ()
    assert session.history_snapshot() == snapshot
    session.register_history_item("u2", _user("edited"))
    assert ledger.refresh(session.history_snapshot()) == (nodes[0].node_id,)
    assert ledger.valid(nodes[1].node_id)


def test_l1_batch_rejects_unknown_source_before_admitting_nodes(session):
    session.register_history_item("u1", _user("one"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()

    with pytest.raises(DuplexContextNodeRejectedError, match="unknown_source_id"):
        ledger.add_l1_nodes(snapshot, messages={snapshot.source_ids[0]: _user("one"), -1: _user("unknown")})

    assert ledger.nodes() == ()


def test_l1_batch_keeps_individual_duplicate_admission_rule(session):
    session.register_history_item("u1", _user("one"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    existing = _l1(ledger, snapshot, 0)

    with pytest.raises(DuplexContextNodeRejectedError, match="source_already_has_media_dropped_node"):
        ledger.add_l1_nodes(snapshot, messages={snapshot.source_ids[0]: _user("another")})

    assert ledger.nodes() == (existing,)


def _setup_replacement(session: DuplexEngineSession) -> Callable[[], object]:
    session.register_history_item("u1", _user("hi"))
    return lambda: session.register_history_item("u1", _user("edited"))


def _setup_transcript_update(session: DuplexEngineSession) -> Callable[[], object]:
    committed = session.commit_audio_input()
    session.register_history_item("u1", committed.message)
    part = session.input_audio_part("u1")
    return lambda: session.update_input_transcript("u1", "heard words", audio_part=part)


def _setup_late_item_id(session: DuplexEngineSession) -> Callable[[], object]:
    committed = session.commit_audio_input(transcript="hello")
    return lambda: session.register_history_item("u1", committed.message)


@pytest.mark.parametrize(
    "setup",
    [_setup_replacement, _setup_transcript_update, _setup_late_item_id],
    ids=["content_replaced", "transcript_updated", "late_item_id"],
)
def test_source_edit_invalidates_node(session, setup):
    edit = setup(session)
    ledger = DuplexContextLedger()
    node = _l1(ledger, session.history_snapshot(), 0)

    edit()

    assert ledger.refresh(session.history_snapshot()) == (node.node_id,)
    assert not ledger.valid(node.node_id)
    assert ledger.nodes() == ()


def test_deleted_source_invalidates_node(session):
    session.register_history_item("u1", _user("gone"))
    session.register_history_item("u2", _user("kept"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    gone = _l1(ledger, snapshot, 0)
    kept = _l1(ledger, snapshot, 1)

    assert session.delete_history_item("u1")

    assert ledger.refresh(session.history_snapshot()) == (gone.node_id,)
    assert ledger.valid(kept.node_id)


def test_invalidation_propagates_three_levels(session):
    session.register_history_item("s1", _user("one"))
    session.register_history_item("s2", _user("two"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    a = _l1(ledger, snapshot, 0)
    b = _summary(ledger, snapshot, a)
    c = _summary(ledger, snapshot, b)
    d = _l1(ledger, snapshot, 1)

    session.register_history_item("s1", _user("edited"))

    assert ledger.refresh(session.history_snapshot()) == (a.node_id, b.node_id, c.node_id)
    assert not any(ledger.valid(node.node_id) for node in (a, b, c))
    assert ledger.valid(d.node_id)
    assert ledger.nodes() == (d,)


def test_diamond_dependency_invalidates_only_dependents(session):
    session.register_history_item("s1", _user("one"))
    session.register_history_item("s2", _user("two"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    a = _l1(ledger, snapshot, 0)
    b = _l1(ledger, snapshot, 1)
    c = _summary(ledger, snapshot, a, b)
    d = _summary(ledger, snapshot, b)

    session.register_history_item("s1", _user("edited"))

    assert ledger.refresh(session.history_snapshot()) == (a.node_id, c.node_id)
    assert ledger.nodes() == (b, d)


def test_add_node_rejects_stale_source_revision(session):
    session.register_history_item("u1", _user("hi"))
    session.register_history_item("u2", _user("bye"))
    old = session.history_snapshot()
    session.register_history_item("u1", _user("edited"))
    assert session.delete_history_item("u2")
    current = session.history_snapshot()
    ledger = DuplexContextLedger()

    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(current, level=L1, sources=source_refs(old, old.source_ids[0]), message=_user("hi"))
    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(current, level=L1, sources=source_refs(old, old.source_ids[1]), message=_user("bye"))
    assert ledger.nodes() == ()


def test_add_node_rejects_missing_or_invalid_parent(session):
    session.register_history_item("s1", _user("one"))
    session.register_history_item("s2", _user("two"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()

    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(snapshot, level=SUMMARY, sources=(), parents=(99,), message=_user("x"))

    pruned = _l1(ledger, snapshot, 0)
    session.register_history_item("s1", _user("edited"))
    current = session.history_snapshot()
    assert ledger.refresh(current) == (pruned.node_id,)

    with pytest.raises(DuplexContextNodeRejectedError):
        _summary(ledger, current, pruned, source_index=1)
    assert ledger.nodes() == ()


def test_add_node_rejects_malformed_nodes(session):
    session.register_history_item("s1", _user("one"))
    session.register_history_item("s2", _user("two"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    a = _l1(ledger, snapshot, 0)

    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(snapshot, level=SUMMARY, sources=(), parents=(), message=_user("x"))
    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(snapshot, level=L1, sources=source_refs(snapshot, *snapshot.source_ids), message=_user("x"))
    with pytest.raises(DuplexContextNodeRejectedError):
        ledger.add_node(
            snapshot,
            level=L1,
            sources=source_refs(snapshot, snapshot.source_ids[1]),
            parents=(a.node_id,),
            message=_user("x"),
        )
    with pytest.raises(DuplexContextNodeRejectedError):
        _l1(ledger, snapshot, 0)
    assert ledger.nodes() == (a,)


def test_node_ids_never_reused_after_prune(session):
    session.register_history_item("u1", _user("hi"))
    ledger = DuplexContextLedger()
    first = _l1(ledger, session.history_snapshot(), 0)
    session.register_history_item("u1", _user("edited"))
    ledger.refresh(session.history_snapshot())

    second = _l1(ledger, session.history_snapshot(), 0)
    assert session.delete_history_item("u1")
    ledger.refresh(session.history_snapshot())
    session.register_history_item("u1", _user("recreated"))
    third = _l1(ledger, session.history_snapshot(), 0)

    assert first.node_id < second.node_id < third.node_id


def test_l1_node_for_returns_only_valid_l1(session):
    session.register_history_item("s1", _user("one"))
    session.register_history_item("s2", _user("two"))
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    a = _l1(ledger, snapshot, 0)
    _summary(ledger, snapshot, source_index=1)

    assert ledger.l1_node_for(snapshot.source_ids[0]) == a
    assert ledger.l1_node_for(snapshot.source_ids[1]) is None

    session.register_history_item("s1", _user("edited"))
    ledger.refresh(session.history_snapshot())

    assert ledger.l1_node_for(snapshot.source_ids[0]) is None
