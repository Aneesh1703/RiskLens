"""
Risk Fusion Stress Test
========================
Thorough validation of risk_fusion logic.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import random
import pandas as pd
from risk_fusion import fuse_scores


def print_case(title, anomaly, sequence, rule):
    result = fuse_scores(
        anomaly_score=anomaly,
        sequence_score=sequence,
        rule_score=rule,
        triggered_rules=["test_rule"] if rule > 0 else []
    )

    print(f"\n=== {title} ===")
    print(f"Inputs      : A={anomaly:.2f}, S={sequence:.2f}, R={rule:.2f}")
    print(f"Final Risk  : {result.final_risk:.3f}")
    print(f"Level       : {result.risk_level}")
    print(f"Confidence  : {result.confidence:.2f}")
    print("-" * 50)


def deterministic_tests():
    print("\n" + "="*60)
    print("  DETERMINISTIC EDGE CASE TESTS")
    print("="*60)

    print_case("All zeros", 0.0, 0.0, 0.0)
    print_case("Only anomaly high", 0.9, 0.1, 0.0)
    print_case("Only sequence high", 0.1, 0.9, 0.0)
    print_case("Only rules high", 0.1, 0.1, 0.95)
    print_case("All medium", 0.5, 0.5, 0.5)
    print_case("All high", 0.9, 0.85, 0.8)
    print_case("Conflicting signals", 0.9, 0.2, 0.1)


def random_batch_test(n=1000):
    print("\n" + "="*60)
    print("  RANDOM DISTRIBUTION TEST")
    print("="*60)

    results = []

    for _ in range(n):
        anomaly = random.random()
        sequence = random.random()
        rule = random.random()

        result = fuse_scores(
            anomaly_score=anomaly,
            sequence_score=sequence,
            rule_score=rule,
        )

        results.append(result.final_risk)

    df = pd.Series(results)

    print(f"Total samples: {n}")
    print(f"Mean risk   : {df.mean():.3f}")
    print(f"Std risk    : {df.std():.3f}")
    print(f"Min risk    : {df.min():.3f}")
    print(f"Max risk    : {df.max():.3f}")

    print("\nRisk level distribution:")
    bins = pd.cut(df, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["low", "medium", "high", "critical"])
    print(bins.value_counts().sort_index())


def escalation_test():
    print("\n" + "="*60)
    print("  ESCALATION TESTS")
    print("="*60)

    print_case("Rule override scenario", 0.2, 0.2, 0.99)
    print_case("Dual ML spike", 0.9, 0.9, 0.1)


if __name__ == "__main__":
    deterministic_tests()
    escalation_test()
    random_batch_test(2000)