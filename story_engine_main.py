from __future__ import annotations

import argparse

from story_engine.engine.runner import RunOptions, StoryEngineRunner


def parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(description="Story Engine modular runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rag-db", default="story_rag.sqlite")
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-chars", type=int, default=100000)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--seg-tokens", type=int, default=1600)
    parser.add_argument("--bible", default="bible.txt")
    args = parser.parse_args()
    return RunOptions(
        config_path=args.config,
        rag_db=args.rag_db,
        out_path=args.out,
        target_chars=args.target_chars,
        segments=args.segments,
        seg_tokens=args.seg_tokens,
        bible_path=args.bible,
    )


def main() -> None:
    options = parse_args()
    runner = StoryEngineRunner(options)
    runner.run()


if __name__ == "__main__":
    main()
