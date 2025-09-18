from __future__ import annotations

import asyncio
import datetime
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

from story_engine.config import EngineConfig, load_config
from story_engine.engine.full_engine import Engine
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
    """Convenience runner that reuses the legacy Engine with refactored modules."""

    def __init__(self, options: RunOptions) -> None:
        self.options = options
        self.cfg: Optional[EngineConfig] = None

    def load(self) -> None:
        opts = self.options
        cfg = load_config(opts.config_path)
        self.cfg = cfg
        create_llm_cache(cfg)
        create_autonomy_services(cfg)
        create_canon_services(cfg)
        create_rhythm_mixer()

    def run(self) -> None:
        if self.cfg is None:
            self.load()
        assert self.cfg is not None
        cfg_dict = self.cfg.as_dict()
        opts = self.options
        out_path = opts.out_path or f"novel_story_engine_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        engine = Engine(
            cfg_dict,
            opts.rag_db,
            out_path,
            opts.target_chars,
            opts.segments,
            opts.seg_tokens,
        )
        try:
            con = sqlite3.connect(opts.rag_db).cursor()
            exist = con.execute(
                "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='rag'"
            ).fetchone()[0]
            con.connection.close()
        except Exception:
            exist = 0
        if not exist:
            if os.path.exists(opts.bible_path):
                engine.ingest_bible(open(opts.bible_path, "r", encoding="utf-8").read())
            else:
                engine.ingest_bible(
                    "배경: 익명 도시. 선형 진행, 과설명 자제, 감정선–인과 중심. 새 조직/모티프는 최소화하고 이름은 가명 처리."
                )
        asyncio.run(engine.autorun(opts.target_chars))
        print(f"[저장 완료] {out_path} / {opts.rag_db}")
