HTU automatic model-update system

Files:
- train_model.py: validates the updated dataset, compares several models, evaluates them, backs up the previous model, and replaces it safely.
- model_manager.py: checks whether the dataset changed and retrains when needed.
- app_integration.txt: exact app.py integration lines.
- requirements-model-update.txt: required packages.

Installation:
    python -m pip install -r requirements-model-update.txt

Manual retraining after updating the CSV:
    python train_model.py

Require at least 70% test accuracy:
    python train_model.py --minimum-accuracy 0.70

Specify the target column when automatic detection fails:
    python train_model.py --target major

Force replacement even when the new score is lower:
    python train_model.py --force

Recommended workflow:
1. Replace or edit htu_majors_interests.csv.
2. Run python train_model.py.
3. Review models/training_metadata.json.
4. Start Streamlit.

Automatic workflow:
Integrate model_manager.py using the instructions in app_integration.txt.
The app will compare the dataset SHA-256 hash with the saved model metadata.
When the file content changes, it retrains once and then loads the new model.

Accuracy note:
The system selects the model by cross-validated macro F1, not training accuracy.
This is more appropriate when different majors have unequal sample counts.
