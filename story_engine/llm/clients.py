from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, Optional

try:
    from anthropic import Anthropic, APIStatusError
except Exception:  # pragma: no cover - anthropic optional
    Anthropic = None
    APIStatusError = Exception  # type: ignore[misc]

try:
    from openai import OpenAI as _OpenAI
except Exception:  # pragma: no cover - openai optional
    _OpenAI = None

LLM_LOG_PATH = os.environ.get("LLM_LOG_PATH", "llm_calls.log")


def make_sha(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def log_llm_call(
    model: str,
    kind: str,
    system: str,
    user: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not LLM_LOG_PATH:
        return
    try:
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "kind": kind,
            "system": (system[:160] if system else ""),
            "user": (user[:320] if user else ""),
            "meta": meta or {},
        }
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


class LLMClientError(RuntimeError):
    pass


class LLMCache:
    def __init__(
        self,
        path: str = "llm_cache.sqlite",
        *,
        enable: bool = True,
        ttl_hours: Optional[float] = 168.0,
    ) -> None:
        self.enable = bool(enable)
        self.ttl = float(ttl_hours) if ttl_hours is not None else None
        self.con = sqlite3.connect(path)
        cur = self.con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, ts REAL, v TEXT)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_cache_ts ON cache(ts)")
        self.con.commit()

    def get(self, key: str) -> Optional[str]:
        if not self.enable:
            return None
        row = self.con.cursor().execute(
            "SELECT ts, v FROM cache WHERE k=?",
            (key,),
        ).fetchone()
        if not row:
            return None
        if self.ttl is not None:
            age_h = (time.time() - float(row[0])) / 3600.0
            if age_h > self.ttl:
                try:
                    self.con.cursor().execute("DELETE FROM cache WHERE k=?", (key,))
                    self.con.commit()
                except Exception:
                    pass
                return None
        return row[1]

    def put(self, key: str, value: str) -> None:
        if not self.enable:
            return
        cur = self.con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cache(k, ts, v) VALUES(?,?,?)",
            (key, time.time(), value),
        )
        self.con.commit()


GLOBAL_CACHE: Optional[LLMCache] = None


def set_global_cache(cache: Optional[LLMCache]) -> None:
    global GLOBAL_CACHE
    GLOBAL_CACHE = cache


def get_global_cache() -> Optional[LLMCache]:
    return GLOBAL_CACHE


class LLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.ok = bool(Anthropic and os.environ.get("ANTHROPIC_API_KEY")) and model != "none"
        self.cli = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) if self.ok else None

    def gen(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1400,
        temperature: float = 0.85,
        top_p: float = 0.95,
    ) -> str:
        if not self.cli:
            return "[생성 비활성화]"
        cache = GLOBAL_CACHE
        cache_key = make_sha(
            f"ANTH|{self.model}|{max_tokens}|{temperature}|{top_p}|{system}|{user}"
        )
        if cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        last_error: Optional[Exception] = None
        for _ in range(5):
            try:
                response = self.cli.messages.create(
                    model=self.model,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    messages=[{"role": "user", "content": user}],
                )
                output = "".join(getattr(block, "text", "") for block in response.content)
                log_llm_call(
                    self.model,
                    "anthropic",
                    system,
                    user,
                    meta={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
                )
                if cache:
                    cache.put(cache_key, output)
                return output
            except APIStatusError as err:  # type: ignore[misc]
                last_error = err
        if last_error:
            raise LLMClientError(str(last_error)) from last_error
        return ""


class LLM_OAI:
    def __init__(self, model: str) -> None:
        self.model = model
        self.cli = _OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if (_OpenAI and os.environ.get("OPENAI_API_KEY")) else None

    def gen(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        force_json: bool = False,
    ) -> str:
        if not self.cli:
            return "{}"
        cache = GLOBAL_CACHE
        cache_key = make_sha(
            f"OAI|{self.model}|{max_tokens}|{temperature}|{int(force_json)}|{system}|{user}"
        )
        if cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            if force_json:
                response = self.cli.responses.create(
                    model=self.model,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                output = response.output_text if response else "{}"
            else:
                response = self.cli.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                output = response.choices[0].message.content if response else ""
        except Exception as exc:  # pragma: no cover - network
            raise LLMClientError(str(exc)) from exc
        if cache and output:
            cache.put(cache_key, output)
        log_llm_call(
            self.model,
            "openai",
            system,
            user,
            meta={"max_tokens": max_tokens, "temperature": temperature, "force_json": force_json},
        )
        return output
