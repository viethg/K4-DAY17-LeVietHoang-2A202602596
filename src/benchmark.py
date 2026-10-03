from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return fraction of expected facts present in answer (0.0 to 1.0)."""
    if not expected:
        return 0.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return round(matches / len(expected), 2)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Calculate a lightweight quality score for offline mode.

    Scoring criteria (0.0 to 1.0):
    - Fact recall completeness (up to 0.70)
    - Response length and substance (up to 0.20)
    - Structure and punctuation polish (up to 0.10)
    """
    if not answer or not answer.strip():
        return 0.0

    score = 0.0
    r_pts = recall_points(answer, expected)
    score += r_pts * 0.70

    ans_len = len(answer.strip())
    if 20 <= ans_len <= 500:
        score += 0.20
    elif ans_len > 0:
        score += 0.10

    if any(p in answer for p in [":", "-", ".", ","]):
        score += 0.10

    return round(min(1.0, score), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall queries."""
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]
        recall_questions = conv.get("recall_questions", [])

        # 1. Feed conversation turns into the agent
        for turn in turns:
            agent.reply(user_id, conv_id, turn)

        # 2. Ask recall questions in fresh threads (measuring cross-session recall)
        for q_idx, q in enumerate(recall_questions):
            recall_thread = f"{conv_id}_recall_{q_idx}"
            question_text = q["question"]
            expected = q.get("expected_contains", [])

            res = agent.reply(user_id, recall_thread, question_text)
            ans = res["answer"]

            r_score = recall_points(ans, expected)
            q_score = heuristic_quality(ans, expected)

            all_recall_scores.append(r_score)
            all_quality_scores.append(q_score)

    avg_recall = round(sum(all_recall_scores) / len(all_recall_scores), 2) if all_recall_scores else 0.0
    avg_quality = round(sum(all_quality_scores) / len(all_quality_scores), 2) if all_quality_scores else 0.0

    # Aggregate token accounting
    total_agent_tokens = agent.token_usage()
    total_prompt_tokens = agent.prompt_token_usage()

    # Memory file size for all tested users
    user_ids = {c["user_id"] for c in conversations}
    total_mem_bytes = sum(agent.memory_file_size(uid) for uid in user_ids)

    # Total compactions across all conversation threads
    total_compactions = sum(agent.compaction_count(c["id"]) for c in conversations)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_mem_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print tabulated markdown output for benchmark rows."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    try:
        from tabulate import tabulate

        table_data = [
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality:.2f}",
                f"{r.memory_growth_bytes:,}",
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback to pure markdown table
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in rows:
            row_str = (
                f"| {r.agent_name} | {r.agent_tokens_only:,} | {r.prompt_tokens_processed:,} | "
                f"{r.recall_score * 100:.1f}% | {r.response_quality:.2f} | {r.memory_growth_bytes:,} | {r.compactions} |"
            )
            lines.append(row_str)
        return "\n".join(lines)


def main() -> None:
    """Run both standard and long-context stress benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("1. STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 80)
    std_convs = load_conversations(std_path)

    # Clean state between runs to ensure fair baseline vs advanced test
    base_agent_std = BaselineAgent(config=config, force_offline=True)
    adv_agent_std = AdvancedAgent(config=config, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline", base_agent_std, std_convs, config)
    row_adv_std = run_agent_benchmark("Advanced", adv_agent_std, std_convs, config)
    print(format_rows([row_base_std, row_adv_std]))

    print("\n" + "=" * 80)
    print("2. LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 80)
    stress_convs = load_conversations(stress_path)

    base_agent_stress = BaselineAgent(config=config, force_offline=True)
    adv_agent_stress = AdvancedAgent(config=config, force_offline=True)

    row_base_stress = run_agent_benchmark("Baseline", base_agent_stress, stress_convs, config)
    row_adv_stress = run_agent_benchmark("Advanced", adv_agent_stress, stress_convs, config)
    print(format_rows([row_base_stress, row_adv_stress]))


if __name__ == "__main__":
    main()

