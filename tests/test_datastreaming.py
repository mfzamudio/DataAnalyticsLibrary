"""
Tests for analytics.datastreaming.

Only the SIMULATED source and the consumption helpers run here — the Kafka and
RabbitMQ adapters need a live broker and external client libraries, so they are
out of scope for unit tests (we only assert their import-guard behavior).
"""

import pandas as pd
import pytest

from analytics import datastreaming as ds


def test_simulated_source_is_bounded():
    src = ds.SimulatedStreamSource(n_messages=5)
    records = list(src.messages())
    assert len(records) == 5
    # Each record is a dict with the expected event schema.
    assert set(records[0]) == {
        "event_id", "event_time", "region", "product_category",
        "channel", "units_sold", "unit_price", "sales",
    }


def test_simulated_source_is_reproducible():
    a = list(ds.SimulatedStreamSource(n_messages=3, seed=7).messages())
    b = list(ds.SimulatedStreamSource(n_messages=3, seed=7).messages())
    # Same seed => identical categorical/numeric fields (event_time aside).
    assert [r["region"] for r in a] == [r["region"] for r in b]
    assert [r["sales"] for r in a] == [r["sales"] for r in b]


def test_stream_to_dataframe_respects_limit():
    # Infinite source, bounded by limit -> must terminate.
    src = ds.SimulatedStreamSource()
    df = ds.stream_to_dataframe(src, limit=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_stream_to_dataframe_empty():
    src = ds.SimulatedStreamSource(n_messages=0)
    df = ds.stream_to_dataframe(src, limit=10)
    assert df.empty


def test_micro_batch_sizes():
    src = ds.SimulatedStreamSource(n_messages=250)
    sizes = [len(b) for b in ds.micro_batch(src, batch_size=100)]
    # Last batch holds the remainder.
    assert sizes == [100, 100, 50]


def test_micro_batch_max_batches_bounds_infinite_stream():
    src = ds.SimulatedStreamSource()  # infinite
    batches = list(ds.micro_batch(src, batch_size=10, max_batches=3))
    assert len(batches) == 3
    assert all(len(b) == 10 for b in batches)


def test_consume_counts_messages_and_calls_handler():
    src = ds.SimulatedStreamSource(n_messages=55)
    seen = []
    total = ds.consume(src, handler=lambda df: seen.append(len(df)), batch_size=20)
    assert total == 55
    assert seen == [20, 20, 15]


def test_context_manager_closes(monkeypatch):
    closed = {"value": False}

    class Dummy(ds.StreamSource):
        def messages(self):
            yield {"x": 1}

        def close(self):
            closed["value"] = True

    with Dummy() as src:
        list(src.messages())
    assert closed["value"] is True


def test_kafka_adapter_requires_library(monkeypatch):
    """If kafka-python is absent, consuming raises a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kafka":
            raise ImportError("simulated missing kafka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    src = ds.KafkaStreamSource("topic")
    with pytest.raises(ImportError, match="kafka-python"):
        next(src.messages())


def test_spark_template_is_runnable_text():
    template = ds.spark_structured_streaming_template()
    assert "readStream" in template
    assert "kafka" in template
    # It should at least be syntactically valid Python source.
    compile(template, "<template>", "exec")
