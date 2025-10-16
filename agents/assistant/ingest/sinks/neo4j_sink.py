from __future__ import annotations
from typing import List
from loguru import logger
from neo4j import GraphDatabase, basic_auth  # type: ignore
from agents.assistant.ingest.sinks.base import BaseSink
from agents.assistant.ingest.models import MemoryRecord
from agents.assistant.ingest.config import settings

_CYPHER_BOOTSTRAP = [
    "CREATE CONSTRAINT unique_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;",
    "CREATE INDEX topic_title_index IF NOT EXISTS FOR (t:Topic) ON (t.title);",
]

class Neo4jSink(BaseSink):
    def __init__(self) -> None:
        if not (settings.NEO4J_URL and settings.NEO4J_PASSWORD):
            raise RuntimeError("Neo4j not configured (NEO4J_URL, NEO4J_PASSWORD)")
        self.driver = GraphDatabase.driver(settings.NEO4J_URL, auth=basic_auth(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self.driver.session(database=settings.NEO4J_DB) as sess:
            for cmd in _CYPHER_BOOTSTRAP:
                try:
                    sess.run(cmd)
                except Exception:
                    pass

    def upsert(self, records: List[MemoryRecord]) -> None:
        if not records:
            return
        with self.driver.session(database=settings.NEO4J_DB) as sess:
            tx = sess.begin_transaction()
            ok, fail = 0, 0
            for r in records:
                topic_id = (r.metadata or {}).get("topic_id")
                title = (r.metadata or {}).get("topic_title") or "Untitled"
                try:
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
                            topic_id=topic_id, title=title, cid=r.id, text=r.text, source=r.metadata.get("source"),
                        )
                    else:
                        tx.run("""MERGE (c:Chunk {id: $cid}) SET c.text=$text, c.source=$source""",
                               cid=r.id, text=r.text, source=r.metadata.get("source"))
                    ok += 1
                except Exception as e:
                    logger.warning(f"neo4j upsert failed for {r.id}: {e}")
                    fail += 1
            tx.commit()
            logger.info(f"neo4j sink: {ok} ok, {fail} failed")
