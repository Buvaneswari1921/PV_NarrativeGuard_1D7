import pandas as pd


# --------------------------------------------------
# 1. Load screening results
# --------------------------------------------------

rule_results = pd.read_csv(
    "outputs/duplicate_screening_results.csv"
)

# The rule-based module stores its score as similarity_score.
# Rename it to rule_score for the combined scoring layer.
if "similarity_score" in rule_results.columns:
    rule_results = rule_results.rename(
        columns={"similarity_score": "rule_score"}
    )

semantic_results = pd.read_csv(
    "outputs/semantic_similarity_results.csv"
)


# --------------------------------------------------
# 2. Combine rule and semantic scores
# --------------------------------------------------

combined = pd.merge(
    rule_results,
    semantic_results,
    on=["incoming_case_id", "candidate_case_id"],
    how="inner"
)


# --------------------------------------------------
# 3. Calculate combined score
# --------------------------------------------------

combined["combined_score"] = (
    0.40 * (combined["rule_score"] / 100)
    + 0.60 * combined["semantic_score"]
)


# --------------------------------------------------
# 4. Initial investigation classification
# --------------------------------------------------

def classify_candidate(row):

    rule_score = row["rule_score"]
    semantic_score = row["semantic_score"]

    # Strong agreement between structured
    # and semantic evidence
    if rule_score >= 80 and semantic_score >= 0.90:
        return "High-confidence duplicate candidate"

    # Moderate evidence
    elif rule_score >= 60 or semantic_score >= 0.80:
        return "Potential duplicate - human review"

    # Weak evidence
    else:
        return "Low similarity"


combined["classification"] = combined.apply(
    classify_candidate,
    axis=1
)


# --------------------------------------------------
# 5. Sort candidates
# --------------------------------------------------

combined = combined.sort_values(
    ["incoming_case_id", "combined_score"],
    ascending=[True, False]
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

output_path = (
    "outputs/combined_duplicate_results.csv"
)

combined.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("Combined duplicate scoring completed.")
print(f"Results saved to: {output_path}")

print("\nCombined duplicate assessment:")
print(
    combined[
        [
            "incoming_case_id",
            "candidate_case_id",
            "rule_score",
            "semantic_score",
            "combined_score",
            "classification",
        ]
    ].to_string(index=False)
)