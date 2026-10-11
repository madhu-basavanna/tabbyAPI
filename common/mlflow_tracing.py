"""
Optional MLflow tracing for tabbyAPI.

Enables full observability of what happens inside the server: per-request
spans for every generation (prompt tokens, cache reuse, prefill/decode times,
tokens/sec, speculative-decoding accept/reject, finish reason), nested spans
for each parsed tool call (name + arguments), and a span per API request
with the full request payload and response.

Tracing is opt-in and optional:
  * `mlflow` is not a hard dependency. It is imported lazily at setup time.
  * When disabled (the default) every span helper is a cheap no-op, so the
    instrumented code paths run with zero overhead and the server behaves
    exactly as before.
  * When enabled but the `mlflow` package is missing, setup logs a warning
    and tracing stays off instead of crashing the server.

Enable it in config.yml:

    monitoring:
      mlflow: true
      mlflow_tracking_uri: sqlite:///mlflow.db
      mlflow_experiment: tabbyapi

Every generation request becomes one MLflow trace; the spans inside it are
the Exllama generation step (token/timing metrics) and one span per parsed
tool call. Traces land in the configured tracking backend (SQLite, a hosted
MLflow server, etc.) and can be browsed in the MLflow UI.
"""

import functools
import inspect
from contextlib import contextmanager
from typing import Callable

from common.logger import xlogger
from common.tabby_config import config

# Set once by setup(); every helper checks this before touching mlflow
_enabled = False
_setup_done = False


def _cfg():
    """The monitoring config section, tolerating it being absent (old configs)."""

    return getattr(config, "monitoring", None)


def is_enabled() -> bool:
    return _enabled


def is_setup_done() -> bool:
    return _setup_done


def _tracking_server_reachable(tracking_uri: str, timeout: float = 3.0) -> bool:
    """
    Fast pre-flight check that an http(s) tracking server is reachable.

    MLflow's client uses generous defaults (120 s timeout, 7 retries), so a
    misconfigured or unreachable server would otherwise block server startup
    for many minutes. This check bounds the wait to a few seconds.
    """

    from urllib.parse import urlparse
    from urllib.request import urlopen

    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("http", "https"):
        return True  # file-based backends (sqlite, local paths): nothing to check

    target = f"{parsed.scheme}://{parsed.netloc}"
    try:
        urlopen(target, timeout=timeout).close()
        return True
    except Exception:
        return False


def setup() -> bool:
    """
    Initialize MLflow tracing from the monitoring config.

    Idempotent: a second call is a no-op. Returns True when tracing is active.
    Never raises and never blocks startup: a missing mlflow install, an
    unreachable tracking server, or a bad tracking URI just leaves tracing
    disabled so the server still starts.
    """

    global _enabled, _setup_done
    if _setup_done:
        return _enabled

    _setup_done = True

    section = _cfg()
    if section is None or not getattr(section, "mlflow", False):
        return False

    try:
        import mlflow
    except ImportError:
        xlogger.warning(
            "monitoring.mlflow is enabled but the `mlflow` package is not installed. "
            "Install it with `pip install 'mlflow>=3.0'` to enable tracing."
        )
        return False

    tracking_uri = getattr(section, "mlflow_tracking_uri", None) or "sqlite:///mlflow.db"
    experiment = getattr(section, "mlflow_experiment", None) or "tabbyapi"

    if not _tracking_server_reachable(tracking_uri):
        xlogger.warning(
            f"monitoring.mlflow is enabled but the tracking server at {tracking_uri} "
            "is not reachable. Is the MLflow server running and is the hostname/port "
            "right (use the tracking base URL, not the browser UI URL)? "
            "Tracing stays disabled; the server will still start."
        )
        return False

    # Bound MLflow's HTTP timeout/retries so a server that becomes unreachable
    # later can't stall the event loop for minutes.
    import os

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
    except Exception as exc:
        xlogger.warning(
            f"monitoring.mlflow is enabled but MLflow could not be initialized "
            f"(tracking_uri={tracking_uri}): {exc}. Tracing stays disabled."
        )
        return False

    _enabled = True
    xlogger.info(f"MLflow tracing enabled (experiment '{experiment}', tracking uri {tracking_uri})")
    return True


class _NoopSpan:
    """Drop-in stand-in for a LiveSpan used while tracing is disabled."""

    name = None

    def set_inputs(self, *args, **kwargs):
        pass

    def set_outputs(self, *args, **kwargs):
        pass

    def set_attributes(self, *args, **kwargs):
        pass

    def set_attribute(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def add_event(self, *args, **kwargs):
        pass


def current_span():
    """
    The active span, or a no-op span when tracing is off (or when called
    outside any span). Instrumented code always calls this rather than
    touching mlflow directly, so the disabled path costs one attribute check.
    """

    if not _enabled:
        return _NoopSpan()
    try:
        import mlflow

        return mlflow.get_current_active_span() or _NoopSpan()
    except Exception:
        return _NoopSpan()


def safe_set(span, method: str, *args, **kwargs):
    """
    Call a span method (set_inputs/set_outputs/set_attributes/...) ignoring
    any error: metrics are best-effort and must never break a generation.
    """

    try:
        getattr(span, method)(*args, **kwargs)
    except Exception:
        pass


def record_metrics(span, metrics: dict):
    """
    Attach a backend metrics dict to a span as attributes, dropping None
    values. Errors are swallowed: metrics are best-effort and must never
    break a generation.
    """

    if not metrics:
        return
    clean = {key: value for key, value in metrics.items() if value is not None}
    if clean:
        safe_set(span, "set_attributes", clean)


@contextmanager
def span(name: str, span_type: str = "CHAIN"):
    """
    Context manager that opens an MLflow span when tracing is enabled and
    yields a no-op span otherwise. Exceptions are recorded on the span and
    re-raised; the span is always closed.
    """

    if not _enabled:
        yield _NoopSpan()
        return

    try:
        import mlflow

        cm = mlflow.start_span(name=name, span_type=span_type)
        active = cm.__enter__()
    except Exception:
        # mlflow present but in a bad state: degrade to a no-op span so the
        # request still succeeds
        yield _NoopSpan()
        return

    try:
        yield active
    except Exception as exc:
        safe_set(active, "record_exception", exc)
        safe_set(active, "set_status", "ERROR")
        raise
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def trace(name: str, span_type: str = "CHAIN") -> Callable:
    """
    Decorator that opens one span around a function, supporting plain,
    async, and async-generator functions. The span is only opened when
    tracing is enabled, so the disabled path adds no span overhead.

    The wrapped function records its own inputs/outputs/attributes through
    `current_span()`; the decorator only provides the span's lifetime.
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def async_gen_wrapper(*args, **kwargs):
                with span(name, span_type):
                    async for item in fn(*args, **kwargs):
                        yield item

            return async_gen_wrapper

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with span(name, span_type):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            with span(name, span_type):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def request_tag_attributes(request) -> dict:
    """
    JSON-safe attributes identifying the current HTTP request (method, path,
    client). Used to tag request spans; returns an empty dict when the
    request object has no usable scope.
    """

    try:
        scope = getattr(request, "scope", None) or {}
        client = scope.get("client") or []
        return {
            "http.method": scope.get("method"),
            "http.path": scope.get("path"),
            "http.client": f"{client[0]}:{client[1]}" if len(client) >= 2 else None,
        }
    except Exception:
        return {}
