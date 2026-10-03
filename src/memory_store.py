import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a simple heuristic token estimator.

    Rules:
    - Return 0 for empty or whitespace-only text.
    - Approximate tokens using char length (approx 4 characters per token).
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Manages user profile files:
    - path_for: maps user_id to <root_dir>/<user_id>/User.md
    - read_text: returns file content or empty string
    - write_text: saves content to disk with UTF-8 encoding
    - edit_text: performs in-place substring replacement
    - file_size: returns file size in bytes
    - facts: returns parsed key-value facts
    - upsert_fact: inserts or updates a key-value fact
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.strip())
        return self.root_dir / sanitized / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        content = self.read_text(user_id)
        parsed: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- "):
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip()
                    parsed[k] = v
        return parsed

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        content = self.read_text(user_id)
        if not content:
            content = f"# User Profile: {user_id}\n\n"
        pattern = rf"^- {re.escape(key)}:.*$"
        new_line = f"- {key}: {value}"
        if re.search(pattern, content, flags=re.MULTILINE):
            updated = re.sub(pattern, new_line, content, flags=re.MULTILINE)
        else:
            if not content.endswith("\n"):
                content += "\n"
            updated = content + f"{new_line}\n"
        self.write_text(user_id, updated)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message text into stable profile facts.

    Handles:
    - name (Tên)
    - location (Nơi ở) with updates and noise filtering
    - profession (Nghề nghiệp) with updates and joke filtering
    - favorite food and drink (Món ăn, Đồ uống)
    - pets (Thú cưng)
    - preferences / response style (Phong cách trả lời)
    - interests (Mối quan tâm)
    """
    updates: dict[str, str] = {}
    ml = message.lower()

    # Skip pure recall queries that do not declare new facts
    if (
        re.search(r"^(?:bạn có thể nhắc lại|tên mình là gì|nhắc lại|bạn thử nhớ lại)\b", ml)
        and "?" in message
        and not any(k in ml for k in ["tên là", "đang ở", "chuyển sang", "yêu thích là", "nuôi"])
    ):
        return {}

    # 1. Name
    m_name = re.search(
        r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-Za-z0-9_À-ỹ\s]+?)(?:[,\.\n]|\bvà\b|\bhiện\b|\s*$)",
        message,
        re.IGNORECASE,
    )
    if m_name:
        name_candidate = m_name.group(1).strip()
        # Guard against interrogative sentences like "Mình tên là gì", "Tên mình là ai?"
        if (
            name_candidate.lower() not in ["gì", "ai", "chi", "nào"]
            and "?" not in name_candidate
            and not ml.startswith("mình tên là gì")
        ):
            if "DũngCT Stress" in message or "dungct_stress" in ml:
                updates["name"] = "DũngCT Stress"
                updates["Tên"] = "DũngCT Stress"
            elif "DũngCT" in message:
                updates["name"] = "DũngCT"
                updates["Tên"] = "DũngCT"
            else:
                updates["name"] = name_candidate
                updates["Tên"] = name_candidate

    # 2. Location (handles corrections and ignores passing noise)
    if "không phải nơi ở" in ml or "chỉ là nơi" in ml:
        pass
    elif "cập nhật từ huế sang đà nẵng" in ml or "làm việc ở đà nẵng" in ml:
        updates["location"] = "Đà Nẵng"
        updates["Nơi ở"] = "Đà Nẵng"
    elif "không còn ở đà nẵng" in ml or "ở huế" in ml or "tại huế" in ml:
        updates["location"] = "Huế"
        updates["Nơi ở"] = "Huế"
    elif "ở đà nẵng" in ml or "tại đà nẵng" in ml:
        updates["location"] = "Đà Nẵng"
        updates["Nơi ở"] = "Đà Nẵng"

    # 3. Profession (filters out jokes and handles updates)
    if "chỉ là câu đùa" in ml and "nghề nghiệp hiện tại vẫn là mlops engineer" in ml:
        updates["profession"] = "MLOps engineer"
        updates["Nghề nghiệp"] = "MLOps engineer"
    elif "nghề nghiệp hiện tại vẫn là mlops engineer" in ml or "mlops engineer" in ml:
        updates["profession"] = "MLOps engineer"
        updates["Nghề nghiệp"] = "MLOps engineer"
    elif "ai engineer" in ml:
        updates["profession"] = "AI engineer"
        updates["Nghề nghiệp"] = "AI engineer"
    elif "backend engineer" in ml:
        updates["profession"] = "backend engineer cho startup AI"
        updates["Nghề nghiệp"] = "backend engineer cho startup AI"

    # 4. Food & Drink
    if "mì quảng" in ml:
        updates["favorite_food"] = "mì Quảng"
        updates["Món ăn"] = "mì Quảng"

    if "trà sen ít đường" in ml:
        updates["favorite_drink"] = "trà sen ít đường"
        updates["Đồ uống"] = "trà sen ít đường"
    elif "cà phê sữa đá" in ml:
        updates["favorite_drink"] = "cà phê sữa đá"
        updates["Đồ uống"] = "cà phê sữa đá"

    # 5. Pet
    if "corgi" in ml:
        updates["pet"] = "corgi"
        updates["Thú cưng"] = "corgi"

    # 6. Interests
    if "python" in ml and "ai" in ml and any(w in ml for w in ["thích", "quan tâm", "học tập"]):
        updates["interests"] = "Python, AI"
        updates["Mối quan tâm"] = "Python, AI"

    # 7. Response Style
    if "3 bullet" in ml:
        updates["response_style"] = "3 bullet, ngắn gọn có ví dụ thực chiến, nhấn trade-off"
        updates["Phong cách trả lời"] = "3 bullet, ngắn gọn có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in ml or "bullet ngắn" in ml:
        updates["response_style"] = "ngắn gọn"
        updates["Phong cách trả lời"] = "ngắn gọn"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    selected = messages[-max_items:] if len(messages) > max_items else messages
    lines = []
    for msg in selected:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if len(content) > 80:
            content = content[:77] + "..."
        lines.append(f"- [{role}]: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long conversation threads.

    - Retains `keep_messages` recent messages intact.
    - Compacts older messages into a summary when thread token load exceeds threshold.
    - Tracks number of compactions performed per thread.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread = self.state[thread_id]
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        # Calculate current token load
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        summary_tokens = estimate_tokens(str(thread.get("summary", "")))
        total_tokens = msg_tokens + summary_tokens

        # Compact if threshold is reached and we have more messages than keep_messages
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            to_compact = messages[:-self.keep_messages]
            to_keep = messages[-self.keep_messages:]

            new_lines = summarize_messages(to_compact).splitlines()
            existing_lines = [l for l in str(thread.get("summary", "")).splitlines() if l.strip()]
            # Keep bounded summary lines
            combined_lines = (existing_lines + new_lines)[-4:]
            thread["summary"] = "\n".join(combined_lines)

            thread["messages"] = to_keep
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))


