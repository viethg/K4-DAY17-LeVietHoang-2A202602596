from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    cfg = load_config(base_dir=tmp_path)
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # 1. Create and write
    initial_content = "# User Profile: test_user\n\n- Tên: DũngCT\n- Nơi ở: Đà Nẵng\n"
    file_path = store.write_text(user_id, initial_content)
    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # 2. Read
    content = store.read_text(user_id)
    assert "DũngCT" in content
    assert "Đà Nẵng" in content

    # 3. Edit
    success = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert success is True
    updated = store.read_text(user_id)
    assert "Huế" in updated
    assert "Đà Nẵng" not in updated

    # 4. Upsert fact
    store.upsert_fact(user_id, "Nghề nghiệp", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("Nghề nghiệp") == "MLOps engineer"
    assert facts.get("Tên") == "DũngCT"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "stress_test_thread"

    assert manager.compaction_count(thread_id) == 0

    # Send long messages exceeding the 40 token threshold
    manager.append(thread_id, "user", "Tin nhắn 1 dài dằng dặc nhằm mục đích làm tăng số lượng token để kiểm tra nén.")
    manager.append(thread_id, "assistant", "Phản hồi 1 cũng dài tương ứng để đẩy tổng context vượt quá ngưỡng quy định.")
    manager.append(thread_id, "user", "Tin nhắn 2 tiếp tục đẩy token lên cao hơn ngưỡng 40 tokens của cấu hình.")

    ctx = manager.context(thread_id)
    assert manager.compaction_count(thread_id) >= 1
    assert len(ctx["messages"]) == 2
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "recall_user"
    thread_1 = "session_01"
    thread_2 = "session_02"

    # Turn 1 in thread 1
    adv.reply(user_id, thread_1, "Chào bạn, mình tên là DũngCT và thích cà phê sữa đá.")
    base.reply(user_id, thread_1, "Chào bạn, mình tên là DũngCT và thích cà phê sữa đá.")

    # Turn 2 in thread 2 (new session)
    r_adv = adv.reply(user_id, thread_2, "Mình tên là gì và thích đồ uống gì?")
    r_base = base.reply(user_id, thread_2, "Mình tên là gì và thích đồ uống gì?")

    # Advanced remembers via User.md
    assert "DũngCT" in r_adv["answer"]
    assert "cà phê sữa đá" in r_adv["answer"]

    # Baseline forgets across new threads
    assert "DũngCT" not in r_base["answer"]
    assert "cà phê sữa đá" not in r_base["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "load_user"
    thread_id = "very_long_thread"

    # Send a sequence of long turns to both agents
    for i in range(10):
        message = (
            f"Lượt thứ {i}: Đây là một đoạn văn bản rất dài được lặp đi lặp lại nhiều thông tin kỹ thuật "
            f"về hệ thống phân tán, xử lý bất đồng bộ và kiến trúc bộ nhớ."
        )
        adv.reply(user_id, thread_id, message)
        base.reply(user_id, thread_id, message)

    adv_prompt = adv.prompt_token_usage(thread_id)
    base_prompt = base.prompt_token_usage(thread_id)

    # Advanced agent compacts older messages, bounding prompt load
    assert adv.compaction_count(thread_id) > 0
    assert base.compaction_count(thread_id) == 0
    assert adv_prompt < base_prompt


def test_conflict_handling_and_question_guardrails(tmp_path: Path) -> None:
    """Bonus: Verify conflict resolution, update handling, and question guardrails."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    user_id = "guardrail_user"

    # Turn 1: Initial fact
    adv.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và làm backend engineer.")
    facts = adv.profile_store.facts(user_id)
    assert facts.get("Tên") == "DũngCT"
    assert facts.get("Nơi ở") == "Đà Nẵng"
    assert "backend engineer" in facts.get("Nghề nghiệp", "")

    # Turn 2: Correction / update fact (Đà Nẵng -> Huế, backend -> MLOps)
    adv.reply(user_id, "t1", "Giờ mình đã chuyển sang ở Huế và làm MLOps engineer chứ không còn ở Đà Nẵng nữa.")
    facts = adv.profile_store.facts(user_id)
    assert facts.get("Nơi ở") == "Huế"
    assert facts.get("Nghề nghiệp") == "MLOps engineer"
    # Ensure old location is not retained as active location
    assert facts.get("Nơi ở") != "Đà Nẵng"

    # Turn 3: Question guardrail - user asks question "Mình tên là gì?"
    # Must NOT overwrite name with 'gì'
    adv.reply(user_id, "t2", "Mình tên là gì và hiện tại làm nghề gì?")
    facts = adv.profile_store.facts(user_id)
    assert facts.get("Tên") == "DũngCT"
    assert facts.get("Tên") != "gì"

    # Turn 4: Noise / joke guardrail - joke about becoming product manager
    adv.reply(user_id, "t2", "Có lúc mình đùa chuyển sang product manager, nhưng chỉ là câu đùa thôi. Nghề nghiệp hiện tại vẫn là MLOps engineer.")
    facts = adv.profile_store.facts(user_id)
    assert facts.get("Nghề nghiệp") == "MLOps engineer"
    assert "product manager" not in facts.get("Nghề nghiệp", "").lower()


