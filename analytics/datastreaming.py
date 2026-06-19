"""
datastreaming
=============

STAGE 1b of the analytics lifecycle: **Streaming Ingestion**.

Where this fits
---------------
:mod:`analytics.dataloading` handles BOUNDED (batch) data: a file or a query
that has an end. This module handles UNBOUNDED (streaming) data: a continuous
flow of messages from a broker. The bridge back to the rest of the library is
**micro-batching** — we group the never-ending stream into small DataFrames so
every downstream stage (cleansing, exploration, modeling) keeps working
unchanged.

Architecture
------------
::

    [ broker ] --messages()--> StreamSource --micro_batch()--> DataFrame(s)
     Kafka / RabbitMQ / sim                                    -> downstream

* ``StreamSource``           : abstract interface; ``messages()`` yields dict
                               records one at a time.
* ``SimulatedStreamSource``  : FULLY FUNCTIONAL, dependency-free generator of
                               synthetic events. Use it to develop and test a
                               streaming pipeline with no broker installed.
* ``KafkaStreamSource``      : adapter for Apache Kafka (the de-facto industry
                               standard for event streaming).
* ``RabbitMQStreamSource``   : adapter for RabbitMQ (the most common message
                               queue / broker).
* ``micro_batch`` / ``stream_to_dataframe`` / ``consume`` : turn any
  ``StreamSource`` into DataFrames or drive a callback.

IMPORTANT — honesty about the broker adapters
---------------------------------------------
The ``KafkaStreamSource`` and ``RabbitMQStreamSource`` classes use lazy imports
and the public APIs of ``kafka-python`` and ``pika`` respectively. They CANNOT
be exercised without (a) the matching client library installed and (b) a
running broker. They are provided as production-shaped, documented adapters —
verify connection details against your infrastructure and the official client
docs before relying on them. Only the SIMULATED source runs out of the box and
is covered by the test suite.

Spark / Flink note
------------------
Spark Structured Streaming and Apache Flink are *processing engines*, not
brokers — they typically READ FROM Kafka/RabbitMQ and process micro-batches or
true streams. The ``micro_batch`` function here mirrors Spark's micro-batch
model conceptually. Driving PySpark/PyFlink requires their own runtimes; see
``spark_structured_streaming_template`` for a documented, copy-pasteable
starting point (returned as a string so this module has no PySpark dependency).
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd


# ===========================================================================
# Core abstraction
# ===========================================================================

class StreamSource(ABC):
    """
    Abstract base class for a streaming source.

    Subclasses implement :meth:`messages`, a generator that yields one record
    (a ``dict``) at a time. Keeping the unit a plain ``dict`` lets micro-batching
    build a DataFrame from any source uniformly.

    Subclasses MAY override :meth:`close` to release broker connections.
    Implements the context-manager protocol so callers can use ``with``.
    """

    @abstractmethod
    def messages(self) -> Iterator[dict]:
        """Yield records (dicts) from the stream, one at a time."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources. No-op by default."""
        return None

    def __enter__(self) -> "StreamSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ===========================================================================
# Simulated source (functional, no dependencies)
# ===========================================================================

class SimulatedStreamSource(StreamSource):
    """
    Generate a synthetic stream of "sales event" records.

    Designed to mimic what a real broker would deliver, so you can build and
    test a streaming pipeline locally with zero infrastructure. Each emitted
    record is a dict shaped like a transaction event.

    Parameters
    ----------
    n_messages : int, optional
        Total number of messages to emit, then stop. ``None`` (default) means
        an INFINITE stream — pair it with a bound in ``micro_batch`` /
        ``stream_to_dataframe`` so consumption terminates.
    rate_per_sec : float, optional
        If set, throttle emission to roughly this many messages per second
        (simulates real-time arrival via ``time.sleep``). ``None`` emits as
        fast as possible (best for tests).
    seed : int, default 42
        Seed for reproducible synthetic data.

    Examples
    --------
    >>> src = SimulatedStreamSource(n_messages=10)
    >>> df = stream_to_dataframe(src, limit=10)
    >>> len(df)
    10
    """

    # Static dimension values used to fabricate events.
    _REGIONS = ("North", "South", "East", "West")
    _CATEGORIES = ("Electronics", "Furniture", "Office", "Software")
    _CHANNELS = ("Online", "Retail")

    def __init__(
        self,
        *,
        n_messages: Optional[int] = None,
        rate_per_sec: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self.n_messages = n_messages
        self.rate_per_sec = rate_per_sec
        self._rng = np.random.default_rng(seed)

    def _make_event(self, event_id: int) -> dict:
        """Fabricate a single transaction event as a dict."""
        units = int(self._rng.integers(1, 20))
        unit_price = round(float(self._rng.normal(150, 40)), 2)
        sales = round(units * max(unit_price, 1.0), 2)
        return {
            "event_id": event_id,
            # ISO-8601 UTC timestamp, like a real event envelope.
            "event_time": datetime.now(timezone.utc).isoformat(),
            "region": str(self._rng.choice(self._REGIONS)),
            "product_category": str(self._rng.choice(self._CATEGORIES)),
            "channel": str(self._rng.choice(self._CHANNELS)),
            "units_sold": units,
            "unit_price": unit_price,
            "sales": sales,
        }

    def messages(self) -> Iterator[dict]:
        """Yield synthetic events, optionally throttled and/or bounded."""
        # ``count`` gives an infinite counter; we stop ourselves when bounded.
        counter = itertools.count(start=1)
        for event_id in counter:
            if self.n_messages is not None and event_id > self.n_messages:
                return
            yield self._make_event(event_id)
            # Throttle to simulate real-time arrival, if requested.
            if self.rate_per_sec:
                time.sleep(1.0 / self.rate_per_sec)


# ===========================================================================
# Broker adapters (require external libraries + a running broker)
# ===========================================================================

class KafkaStreamSource(StreamSource):
    """
    Consume a Kafka topic as a :class:`StreamSource` (Apache Kafka).

    Uses ``kafka-python`` (``pip install kafka-python``). Requires a reachable
    Kafka broker. Messages are assumed to be UTF-8 JSON and are decoded to
    dicts.

    Parameters
    ----------
    topic : str
        Topic to subscribe to.
    bootstrap_servers : str or list of str, default "localhost:9092"
        Broker address(es).
    group_id : str, optional
        Consumer group id (enables offset tracking across restarts).
    auto_offset_reset : {"earliest", "latest"}, default "earliest"
        Where to start when no committed offset exists.
    consumer_timeout_ms : int, default 10000
        Stop iterating after this many ms of no new messages (so the generator
        terminates instead of blocking forever). Set higher for long-running
        consumers.

    Notes
    -----
    Verify these options against the installed ``kafka-python`` version and
    your cluster's security settings (SASL/SSL) before production use.
    """

    def __init__(
        self,
        topic: str,
        *,
        bootstrap_servers="localhost:9092",
        group_id: Optional[str] = None,
        auto_offset_reset: str = "earliest",
        consumer_timeout_ms: int = 10_000,
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer_timeout_ms = consumer_timeout_ms
        self._consumer = None  # created lazily in messages()

    def messages(self) -> Iterator[dict]:
        import json

        try:
            # Optional dependency; the ImportError is handled below.
            from kafka import KafkaConsumer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "KafkaStreamSource requires kafka-python. "
                "Install it with `pip install kafka-python`."
            ) from exc

        # ``value_deserializer`` turns raw bytes into a dict per message.
        self._consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            consumer_timeout_ms=self.consumer_timeout_ms,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        # Iterating a KafkaConsumer yields messages until the timeout elapses.
        for message in self._consumer:
            yield message.value

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


class RabbitMQStreamSource(StreamSource):
    """
    Consume a RabbitMQ queue as a :class:`StreamSource`.

    Uses ``pika`` (``pip install pika``). Requires a reachable RabbitMQ broker.
    Message bodies are assumed to be UTF-8 JSON and decoded to dicts.

    Parameters
    ----------
    queue : str
        Queue name to consume from.
    host : str, default "localhost"
        Broker host.
    port : int, default 5672
        Broker port.
    durable : bool, default True
        Declare the queue as durable (survives broker restart). Must match how
        the queue was originally declared.
    inactivity_timeout : float, default 5.0
        Seconds to wait for a message before the generator ends (so it does not
        block forever on an empty queue).

    Notes
    -----
    ``BlockingChannel.consume`` yields ``(method, properties, body)`` and yields
    ``(None, None, None)`` when ``inactivity_timeout`` elapses. Verify against
    your ``pika`` version and broker credentials/vhost before production use.
    """

    def __init__(
        self,
        queue: str,
        *,
        host: str = "localhost",
        port: int = 5672,
        durable: bool = True,
        inactivity_timeout: float = 5.0,
    ) -> None:
        self.queue = queue
        self.host = host
        self.port = port
        self.durable = durable
        self.inactivity_timeout = inactivity_timeout
        self._connection = None

    def messages(self) -> Iterator[dict]:
        import json

        try:
            import pika  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "RabbitMQStreamSource requires pika. "
                "Install it with `pip install pika`."
            ) from exc

        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.host, port=self.port)
        )
        channel = self._connection.channel()
        # Declaring is idempotent; ensures the queue exists with our settings.
        channel.queue_declare(queue=self.queue, durable=self.durable)

        for method, _properties, body in channel.consume(
            self.queue, auto_ack=True, inactivity_timeout=self.inactivity_timeout
        ):
            # On inactivity timeout pika yields (None, None, None) -> we stop.
            if body is None:
                break
            yield json.loads(body.decode("utf-8"))

    def close(self) -> None:
        if self._connection is not None and self._connection.is_open:
            self._connection.close()
            self._connection = None


# ===========================================================================
# Consuming a stream -> DataFrames (the bridge to the rest of the library)
# ===========================================================================

def stream_to_dataframe(source: StreamSource, *, limit: int) -> pd.DataFrame:
    """
    Collect the first ``limit`` messages from a stream into ONE DataFrame.

    Use for bounded snapshots ("give me the next 1,000 events as a table").
    Always pass a ``limit`` — an unbounded source would otherwise never return.

    Parameters
    ----------
    source : StreamSource
    limit : int
        Maximum number of messages to collect.

    Returns
    -------
    pandas.DataFrame
        One row per message. Empty DataFrame if the stream produced nothing.
    """
    # ``islice`` safely takes at most ``limit`` items, even from infinite streams.
    records = list(itertools.islice(source.messages(), limit))
    return pd.DataFrame(records)


def micro_batch(
    source: StreamSource,
    *,
    batch_size: int = 100,
    max_batches: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """
    Consume a stream and yield it as a sequence of DataFrame micro-batches.

    This is the core streaming-to-batch bridge (the Spark Structured Streaming
    micro-batch model). Each yielded DataFrame can be fed straight into
    :mod:`analytics.datacleansing`, :mod:`analytics.descriptiveanalysis`, etc.

    Parameters
    ----------
    source : StreamSource
    batch_size : int, default 100
        Number of messages per micro-batch (the last batch may be smaller).
    max_batches : int, optional
        Stop after yielding this many batches. ``None`` runs until the source
        is exhausted (use only with a bounded source, or break out yourself).

    Yields
    ------
    pandas.DataFrame
        One micro-batch at a time.

    Examples
    --------
    >>> src = SimulatedStreamSource(n_messages=250)
    >>> sizes = [len(df) for df in micro_batch(src, batch_size=100)]
    >>> sizes
    [100, 100, 50]
    """
    buffer: list[dict] = []
    batches_yielded = 0

    for record in source.messages():
        buffer.append(record)
        if len(buffer) >= batch_size:
            yield pd.DataFrame(buffer)
            buffer = []
            batches_yielded += 1
            if max_batches is not None and batches_yielded >= max_batches:
                return

    # Flush any remaining records as a final (smaller) batch.
    if buffer and (max_batches is None or batches_yielded < max_batches):
        yield pd.DataFrame(buffer)


def consume(
    source: StreamSource,
    handler: Callable[[pd.DataFrame], None],
    *,
    batch_size: int = 100,
    max_batches: Optional[int] = None,
) -> int:
    """
    Drive a stream, calling ``handler`` on each micro-batch (push model).

    Convenience wrapper over :func:`micro_batch` for "process and forget"
    pipelines (e.g. write each batch to a warehouse, update a dashboard).

    Parameters
    ----------
    source : StreamSource
    handler : callable
        Function invoked with each micro-batch DataFrame. Its return value is
        ignored.
    batch_size : int, default 100
    max_batches : int, optional

    Returns
    -------
    int
        Total number of messages processed across all batches.
    """
    total = 0
    for batch in micro_batch(source, batch_size=batch_size, max_batches=max_batches):
        total += len(batch)
        handler(batch)
    return total


def spark_structured_streaming_template() -> str:
    """
    Return a documented PySpark Structured Streaming snippet (as a string).

    Provided as text so this module has NO PySpark dependency. Copy it into a
    Spark job, adapt the connection options, and verify against your Spark
    version's docs. This is the industry pattern for consuming Kafka at scale
    and processing it in micro-batches — the same model :func:`micro_batch`
    mirrors locally.

    Returns
    -------
    str
        A runnable PySpark template.
    """
    return (
        # NOTE: template only. Requires a Spark runtime + Kafka package.
        'from pyspark.sql import SparkSession\n'
        'from pyspark.sql.functions import from_json, col\n'
        'from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType\n'
        '\n'
        'spark = SparkSession.builder.appName("analytics-stream").getOrCreate()\n'
        '\n'
        '# 1. Read an unbounded stream from Kafka.\n'
        'raw = (spark.readStream.format("kafka")\n'
        '       .option("kafka.bootstrap.servers", "localhost:9092")\n'
        '       .option("subscribe", "sales-events")\n'
        '       .option("startingOffsets", "earliest")\n'
        '       .load())\n'
        '\n'
        '# 2. Parse the JSON payload (Kafka delivers key/value as bytes).\n'
        'schema = (StructType()\n'
        '          .add("region", StringType())\n'
        '          .add("units_sold", IntegerType())\n'
        '          .add("sales", DoubleType()))\n'
        'events = (raw.selectExpr("CAST(value AS STRING) AS json")\n'
        '          .select(from_json(col("json"), schema).alias("e"))\n'
        '          .select("e.*"))\n'
        '\n'
        '# 3. Aggregate per micro-batch and write to the console sink.\n'
        'agg = events.groupBy("region").sum("sales")\n'
        'query = (agg.writeStream.outputMode("complete").format("console").start())\n'
        'query.awaitTermination()\n'
    )
