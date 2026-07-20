# HTU AI Advisor — Working Seed Data

Generated: 2026-07-19T09:55:58

## Included files

- `htu_majors_interests.csv`
  - 5,000 balanced synthetic student-interest records.
  - 34 binary features plus `label`.
  - Feature names exactly match `FEATURE_QUESTIONS` in `advisor_system_fixed.py`.
  - Labels match the current code so model training can run immediately.

- `dept_labeled_chunks_cleaned_Policy.csv`
  - Small working knowledge corpus for the current Bag-of-Words and Hugging Face QA pipeline.

- `majors.json`
  - Starter content for website major cards, recommendations, explanations, and a future knowledge graph.

- `admission_rules.json`
  - Prototype admission-rule structure.
  - Tawjihi values are based on the supplied policy.
  - Verify the latest official policy before public deployment.

- `courses_seed.csv`
  - Representative course records for testing website pages and APIs.
  - It is intentionally incomplete.

- `major_label_mapping.json`
  - Maps internal prototype labels such as `CS` to user-facing names.

## Important limitations

1. The recommendation dataset is synthetic, not collected from real HTU students.
2. It is suitable for a working prototype and model demonstration.
3. Do not claim that it proves real-world recommendation accuracy.
4. For the AI Expo proposal, describe it as a rule-guided synthetic dataset derived from programme themes and official study plans.
5. AI and Data Science have now been merged into one label: `Data Science and Artificial Intelligence`.
6. The Architectural Engineering content is a temporary prototype because no official study plan was included in the supplied files.

## How to use

Place these files beside `advisor_system_fixed.py`, then run:

```bash
python advisor_system_fixed.py
```

To retrain the models first:

```python
from advisor_system_fixed import train_major_predictor
train_major_predictor("htu_majors_interests.csv")
```

The training function should generate the model artifacts and label encoder.


## Required code change

Use `advisor_system_merged.py` instead of the older `advisor_system_fixed.py`.

Delete old trained artifacts before retraining:

- `label_encoder.pkl`
- `best_major_predictor.pkl`
- `best_major_predictor.h5`
- `best_major_predictor_meta.pkl`

Then retrain so the label encoder and model use the merged class.
