# 2026-04-06

Initial commit of the homework assignment, including project documentation, poetry dependency management, and refactored pipeline structure.

## dev-04:18
### Initial project structure and pipeline refactoring
- Added `pyproject.toml` and `poetry.lock` for dependency management.
- Included homework assignment outline in `assignment.txt`, `README.md`, and `1st Assignment.pdf`.
- Refactored `main.py` to break down the monolithic pipeline into modular functions.
- Extracted preprocessing logic into an independent pipeline inside `src/preprocessing.py`.
- Created tests in `tests/test_main.py` and `tests/test_preprocessing.py`.
# 2026-04-07

Implemeted complete unit test coverage for the machine learning pipeline, added target variable encoding to the preprocessing pipeline, verified the framework with comprehensive unit tests, refined the codebase with explicit return type hints, and further refactored the training module into smaller focused functions for improved readability and maintainability.

## dev-00:24
### Classical ML training and preprocessing enhancements
- Created `src/train_classical.py`: Implemented Grid Search for Decision Tree, Random Forest, XGBoost, Logistic Regression, and SVM.
- Added `tests/test_train_classical.py`: verified the new training and evaluation functions.
- Updated `src/preprocessing.py`: Added `LabelEncoder` for target variables and integrated it into the pipeline via `fit_target_encoder` and `apply_target_encoder`.
- Refactored `main.py`: Updated the preprocessing orchestration to include the newly implemented target encoding.
- Dependency Management: Integrated `xgboost` into the project environment using Poetry.
- Automated Visualizations: Added generation of metrics bar plots and confusion matrices for all evaluated models.

## dev-00:49
### Enhanced type safety with function return hints
- Updated `main.py`: Added return type hints to core pipeline functions.
- Updated `src/preprocessing.py`: Defined return types for all data manipulation functions.
- Updated `src/train_classical.py`: Added return type hints for evaluation and search components.
- Updated `tests/`: Added return type hints to all test functions and fixtures across the integration and unit test suite.
## dev-01:05
### Modular refactoring of classical ML training pipeline
- Extracted `run_grid_search_for_model` from `train_classical_models` to isolate per-model grid search logic into a dedicated, testable function.
- Extracted `save_best_model` to handle model persistence independently, with a configurable `models_dir` parameter.
- Extracted `report_feature_importances` to cleanly separate feature importance printing from the main training loop.
- Updated `train_classical_models` to accept `models_dir` and `visuals_dir` parameters instead of hardcoded paths.
- Cleaned up `main.py` to align with refactored function signatures.

## dev-01:14
### Comprehensive test coverage for ML pipeline
- Added tests for preprocessing utilities: `apply_target_encoder`, `apply_imputation`, `apply_scaling`, `save_outlier_histograms`, and `generate_pca_insights`.
- Added tests for modular training components: `build_model_grid`, `run_grid_search_for_model`, `report_feature_importances`, and `save_best_model`.
- Enhanced `test_main.py` with more robust integration tests for the `run_pipeline` orchestration.
