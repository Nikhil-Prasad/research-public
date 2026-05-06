from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class RunStore:
    def __init__(self, root: Path = Path("runs")) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = root / ts
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._event_counter = 0

    def write_json(self, name: str, obj: Any) -> None:
        with self._lock:
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    def write_text(self, name: str, text: str) -> None:
        with self._lock:
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def write_event(self, event_type: str, obj: Any) -> None:
        with self._lock:
            self._event_counter += 1
            target = self.path / f"event_{self._event_counter:04d}_{event_type}.json"
            target.write_text(
                json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
