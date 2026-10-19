import pandas as pd


# --------------------------------------------------
# 1. Load screening results
# --------------------------------------------------

combined = pd.read_csv(
    "outputs/combined_duplicate_results.csv"
)

expected = pd.read_csv(
    "data/expected_results.csv"
)


# --------------------------------------------------
# 2. Identify the strongest candidate for each
#    incoming case
# --------------------------------------------------

top_candidates = (
    combined
    .sort_values(
        ["incoming_case_id", "combined_score"],
        ascending=[True, False]
    )
    .groupby("incoming_case_id")
    .first()
    .reset_index()
)


# --------------------------------------------------
# 3. Display expected-result structure
# --------------------------------------------------

print("Expected-results columns:")
print(expected.columns.tolist())

print("\nExpected results:")
print(expected.to_string(index=False))


# --------------------------------------------------
# 4. Merge expected and predicted results
# --------------------------------------------------

evaluation = pd.merge(
    expected,
    top_candidates,
    on="incoming_case_id",
    how="left"
)


# --------------------------------------------------
# 5. Determine prediction
# --------------------------------------------------
#
# Only a high-confidence candidate is considered
# an automatic duplicate prediction.
#
# Potential duplicates remain for human review.
#

evaluation["predicted_duplicate"] = (
    evaluation["classification"]
    == "High-confidence duplicate candidate"
)


# --------------------------------------------------
# 6. Compare prediction with expected result
# --------------------------------------------------

evaluation["correct"] = (
    evaluation["predicted_duplicate"]
    == evaluation["expected_duplicate"]
)


# --------------------------------------------------
# 7. Calculate confusion-matrix values
# --------------------------------------------------

true_positive = (
    (evaluation["expected_duplicate"] == True)
    & (evaluation["predicted_duplicate"] == True)
).sum()

false_positive = (
    (evaluation["expected_duplicate"] == False)
    & (evaluation["predicted_duplicate"] == True)
).sum()

false_negative = (
    (evaluation["expected_duplicate"] == True)
    & (evaluation["predicted_duplicate"] == False)
).sum()

true_negative = (
    (evaluation["expected_duplicate"] == False)
    & (evaluation["predicted_duplicate"] == False)
).sum()


# --------------------------------------------------
# 8. Calculate performance metrics
# --------------------------------------------------

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

accuracy = (
    (true_positive + true_negative)
    / len(evaluation)
    if len(evaluation) > 0
    else 0
)


# --------------------------------------------------
# 9. Display evaluation summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE DETECTION EVALUATION")
print("=" * 60)

print(f"True Positives : {true_positive}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")
print(f"True Negatives : {true_negative}")

print(f"\nPrecision: {precision:.2%}")
print(f"Recall   : {recall:.2%}")
print(f"Accuracy : {accuracy:.2%}")


# --------------------------------------------------
# 10. Display case-level assessment
# --------------------------------------------------

print("\nCase-level assessment:")

display_columns = [
    "incoming_case_id",
    "candidate_case_id",
    "rule_score",
    "semantic_score",
    "combined_score",
    "classification",
    "expected_duplicate",
    "expected_case_id",
    "duplicate_type",
    "predicted_duplicate",
    "correct",
]

print(
    evaluation[display_columns]
    .to_string(index=False)
)


# --------------------------------------------------
# 11. Save evaluation results
# --------------------------------------------------

output_path = (
    "outputs/duplicate_detection_evaluation.csv"
)

evaluation.to_csv(
    output_path,
    index=False
)

print(
    f"\nEvaluation saved to: {output_path}"
)