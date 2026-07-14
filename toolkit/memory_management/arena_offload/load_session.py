"""Generic production checkpoint-to-arena load session."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


PENDING_CANONICAL_BUILD_ATTR = "_arena_pending_canonical_build"
_CURRENT_SESSION = ContextVar("arena_direct_load_session", default=None)


@dataclass
class _DirectLoadSession:
    device: object
    block_names: tuple[str, ...]
    pending: dict[int, tuple[object, object]] = field(default_factory=dict)
    unsupported_reason: str | None = None

    def publish(self, model, build) -> None:
        if getattr(model, PENDING_CANONICAL_BUILD_ATTR, None) is not None:
            raise RuntimeError("arena_direct_load_duplicate_pending_build")
        setattr(model, PENDING_CANONICAL_BUILD_ATTR, build)
        self.pending[id(model)] = (model, build)

    def claim(self, model):
        build = getattr(model, PENDING_CANONICAL_BUILD_ATTR, None)
        if build is None:
            return None
        delattr(model, PENDING_CANONICAL_BUILD_ATTR)
        self.pending.pop(id(model), None)
        return build

    def rollback_pending(self) -> None:
        for model, build in tuple(self.pending.values()):
            if getattr(model, PENDING_CANONICAL_BUILD_ATTR, None) is build:
                delattr(model, PENDING_CANONICAL_BUILD_ATTR)
            build.rollback()
        self.pending.clear()


def _arena_load_enabled(base_model) -> bool:
    config = getattr(base_model, "model_config", None)
    return bool(
        config is not None
        and getattr(config, "layer_offloading", False)
        and getattr(config, "layer_offloading_smart", False)
        and not getattr(base_model, "te_only", False)
    )


@contextmanager
def model_load_arena_session(base_model):
    """Offer generic direct arena ingestion during ``base_model.load_model``.

    Unsupported or non-inferable state schemas leave the state mapping intact
    and use the ordinary assignment path. Once a mapping has been consumed,
    the model must claim it through ``prepare_arena_offload`` before returning.
    """
    if not _arena_load_enabled(base_model):
        yield None
        return
    if _CURRENT_SESSION.get() is not None:
        raise RuntimeError("nested_arena_direct_load_session")
    block_names = ()
    provider = getattr(base_model, "get_transformer_block_names", None)
    if callable(provider):
        block_names = tuple(provider() or ())
    session = _DirectLoadSession(
        device=getattr(base_model, "device_torch", None),
        block_names=block_names,
    )
    token = _CURRENT_SESSION.set(session)
    try:
        yield session
        if session.pending:
            session.rollback_pending()
            raise RuntimeError("arena_direct_load_build_not_claimed")
    except BaseException:
        session.rollback_pending()
        raise
    finally:
        _CURRENT_SESSION.reset(token)


def try_prepare_canonical_from_state_dict(model, state_dict):
    """Consume inferable managed state into a pending canonical build."""
    session = _CURRENT_SESSION.get()
    if session is None:
        return None
    from .api import prepare_canonical_storage_from_state_dict
    from .construction import CanonicalStateInferenceError
    from .discovery import BlockDiscoveryError

    try:
        build = prepare_canonical_storage_from_state_dict(
            model,
            state_dict,
            block_names=session.block_names,
            device=session.device,
        )
    except (BlockDiscoveryError, CanonicalStateInferenceError) as error:
        session.unsupported_reason = str(error)
        return None
    try:
        session.publish(model, build)
    except BaseException:
        build.rollback()
        raise
    return build


def claim_pending_canonical_build(model):
    """Take a generic loader build at the normal arena-attach boundary."""
    session = _CURRENT_SESSION.get()
    if session is not None:
        return session.claim(model)
    build = getattr(model, PENDING_CANONICAL_BUILD_ATTR, None)
    if build is not None:
        delattr(model, PENDING_CANONICAL_BUILD_ATTR)
    return build


def discard_pending_canonical_build(model) -> None:
    """Rollback a pending build after residual assignment fails."""
    build = claim_pending_canonical_build(model)
    if build is not None:
        build.rollback()
