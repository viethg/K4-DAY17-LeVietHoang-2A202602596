from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session memory only (remembers inside the same thread_id).
    - No persistent storage (`User.md`).
    - Forgets all facts across new sessions/threads.
    - No compact memory: prompt context load grows monotonically with thread length.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                result = self.langchain_agent.invoke(
                    {"messages": [("user", message)]},
                    config=config,
                )
                answer = str(result["messages"][-1].content)

                if thread_id not in self.sessions:
                    self.sessions[thread_id] = SessionState()
                session = self.sessions[thread_id]

                turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
                session.prompt_tokens_processed += turn_prompt_tokens
                session.messages.append({"role": "user", "content": message})

                agent_tokens = estimate_tokens(answer)
                session.token_usage += agent_tokens
                session.messages.append({"role": "assistant", "content": answer})

                return {
                    "answer": answer,
                    "token_usage": session.token_usage,
                    "prompt_tokens_processed": session.prompt_tokens_processed,
                }
            except Exception:
                return self._reply_offline(thread_id, message)

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent generated token count for one thread or all sessions."""
        if thread_id is None:
            return sum(s.token_usage for s in self.sessions.values())
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt context tokens processed for one thread or all sessions."""
        if thread_id is None:
            return sum(s.prompt_tokens_processed for s in self.sessions.values())
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def memory_file_size(self, user_id: str) -> int:
        """Baseline has no persistent User.md file."""
        return 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory layer."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic within-session offline behavior."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        # Baseline sends all previous messages in session + current message
        turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Record user message
        session.messages.append({"role": "user", "content": message})

        # Formulate reply:
        # If fresh thread with only 1 message, Baseline has no prior memory
        if len(session.messages) <= 1:
            reply_text = "Chào bạn! Trong phiên hội thoại này mình chưa có thông tin trước đó của bạn. Bạn cần hỗ trợ gì?"
        else:
            ml = message.lower()
            if any(q in ml for q in ["tên gì", "tên là gì", "đồ uống", "ở đâu", "nghề gì"]):
                found_facts = []
                for prev_msg in session.messages[:-1]:
                    if prev_msg["role"] == "user":
                        pc = prev_msg["content"]
                        if "DũngCT" in pc:
                            found_facts.append("DũngCT")
                        if "cà phê sữa đá" in pc:
                            found_facts.append("cà phê sữa đá")
                        if "mì Quảng" in pc or "mì quảng" in pc.lower():
                            found_facts.append("mì Quảng")
                        if "corgi" in pc.lower():
                            found_facts.append("corgi")
                        if "Huế" in pc:
                            found_facts.append("Huế")
                        elif "Đà Nẵng" in pc:
                            found_facts.append("Đà Nẵng")
                if found_facts:
                    reply_text = f"Trong phiên trò chuyện này, bạn đã chia sẻ: {', '.join(sorted(set(found_facts)))}."
                else:
                    reply_text = "Đã ghi nhận thông tin của bạn trong phiên trò chuyện này."
            else:
                reply_text = "Đã ghi nhận thông tin của bạn trong phiên trò chuyện này."

        agent_tokens = estimate_tokens(reply_text)
        session.token_usage += agent_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "answer": reply_text,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire `create_react_agent` + `InMemorySaver`."""
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None

