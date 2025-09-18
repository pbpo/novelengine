from __future__ import annotations

import asyncio
import datetime
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

from story_engine.config import EngineConfig, load_config
from story_engine.engine.core import Engine, build_engine_services
from story_engine.factory import (
    create_autonomy_services,
    create_canon_services,
    create_llm_cache,
    create_rhythm_mixer,
)


@dataclass
class RunOptions:
    config_path: str
    rag_db: str = "story_rag.sqlite"
    out_path: Optional[str] = None
    target_chars: int = 100000
    segments: int = 3
    seg_tokens: int = 1600
    bible_path: str = "bible.txt"


class StoryEngineRunner:
    """Convenience wrapper to run the modern engine with CLI options."""

    def __init__(self, options: RunOptions) -> None:
        self.options = options
        self.cfg: Optional[EngineConfig] = None
        self._resources: dict[str, object] = {}

    def load(self) -> None:
        opts = self.options
        cfg = load_config(opts.config_path)
        self.cfg = cfg
        self._resources = {
            "llm_cache": create_llm_cache(cfg),
            "autonomy": create_autonomy_services(cfg),
            "canon": create_canon_services(cfg),
            "rhythm": create_rhythm_mixer(),
        }

    def run(self) -> None:
        if self.cfg is None:
            self.load()
        assert self.cfg is not None
        opts = self.options
        out_path = opts.out_path or f"novel_story_engine_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        services = build_engine_services(self.cfg, self._resources)
        engine = Engine(
            self.cfg,
            opts.rag_db,
            out_path,
            opts.target_chars,
            opts.segments,
            opts.seg_tokens,
            services=services,
        )
        try:
            if not self._rag_has_bible(opts.rag_db):
                self._load_bible(engine)
            asyncio.run(engine.autorun(opts.target_chars))
        finally:
            engine.close()
        print(f"[저장 완료] {out_path} / {opts.rag_db}")

    def _load_bible(self, engine: Engine) -> None:
        opts = self.options
        if os.path.exists(opts.bible_path):
            with open(opts.bible_path, "r", encoding="utf-8") as bible_file:
                engine.ingest_bible(bible_file.read())
        else:
            engine.ingest_bible(
                "배경: 익명 도시. 선형 진행, 과설명 자제, 감정선–인과 중심. 새 조직/모티프는 최소화하고 이름은 가명 처리."
            )

    @staticmethod
    def _rag_has_bible(path: str) -> bool:
        try:
            con = sqlite3.connect(path)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = cur.execute("SELECT value FROM meta WHERE key='bible'").fetchone()
            con.commit()
            con.close()
            return bool(row and row[0])
        except Exception:
            return False
