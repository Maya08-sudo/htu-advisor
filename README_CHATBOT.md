# HTU Advisor chatbot build

## First run

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python knowledge_base_builder.py
python -m streamlit run app.py
```

## Main chatbot files

- `rag.py`: orchestrates rules, retrieval, context and response structure.
- `rule_engine.py`: deterministic Tawjihi admission checks.
- `entity_extractor.py`: detects language, major, certificate, degree and grades.
- `intent_classifier.py`: detects question category.
- `retriever.py`: hybrid word and character TF-IDF retrieval.
- `knowledge_base_builder.py`: rebuilds the combined searchable knowledge base.
- `knowledge_base/processed/htu_knowledge_base.csv`: generated index source.

## Updating sources

1. Update `dept_labeled_chunks_cleaned_Policy.csv`, `courses_seed.csv`, or files under `knowledge_base/source/`.
2. Run:

```powershell
python knowledge_base_builder.py
```

3. Restart Streamlit.

The chatbot answers deterministic Tawjihi requirement and eligibility questions through rules. Other questions use grounded retrieval and display their sources.
