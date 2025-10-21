# gateway/mem0/sinks/neo4j_sink.py
from __future__ import annotations

import os
from typing import List

from loguru import logger
from neo4j import GraphDatabase, basic_auth

from .base import BaseSink
from ..models import MemoryRecord

_CYPHER_BOOTSTRAP = [
    "CREATE CONSTRAINT unique_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;",
    "CREATE INDEX topic_title_index IF NOT EXISTS FOR (t:Topic) ON (t.title);",
]


class Neo4jSink(BaseSink):
    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD")
        if not pwd:
            raise RuntimeError("Neo4j not configured (NEO4J_PASSWORD missing)")

        self.database = os.getenv("NEO4J_DB", "neo4j")
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, pwd))
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self.driver.session(database=self.database) as sess:
            for cmd in _CYPHER_BOOTSTRAP:
                try:
                    sess.run(cmd)
                except Exception:
                    pass

    def upsert(self, records: List[MemoryRecord]) -> int:
        if not records:
            return 0

        ok, fail = 0, 0
        with self.driver.session(database=self.database) as sess:
            tx = sess.begin_transaction()
            for r in records:
                try:
                    topic_id = (r.metadata or {}).get("topic_id")
                    title = (r.metadata or {}).get("topic_title") or "Untitled"
                    source = (r.metadata or {}).get("source")

                    if topic_id:
                        tx.run(
                            """
                            MERGE (t:Topic {topic_id: $topic_id})
                            ON CREATE SET t.title = $title
                            ON MATCH SET  t.title = coalesce(t.title, $title)
                            MERGE (c:Chunk {id: $cid})
                            SET c.text = $text, c.source = $source
                            MERGE (t)-[:HAS_CHUNK]->(c)
                            """,
                            topic_id=topic_id, title=title, cid=r.id, text=r.text, source=source,
                        )
                    else:
                        tx.run(
                            """
                            MERGE (c:Chunk {id: $cid})
                            SET c.text = $text, c.source = $source
                            """,
                            cid=r.id, text=r.text, source=source,
                        )
                    ok += 1
                except Exception as e:
                    logger.warning(f"neo4j upsert failed for {r.id}: {e}")
                    fail += 1
            tx.commit()

        logger.info(f"neo4j sink: {ok} ok, {fail} failed")
        return ok
