# 2026-06-03

Implemented a Retrieval-Augmented Generation (RAG) document ingestion pipeline. Processed 8 PDF domain documents, generating over 5,000 vector chunks, and persisted them locally to avoid rebuilding the database on every startup.

## dev-22:02
### Agent Tool Expansion and Streaming Support (Tasks 5 & 6)
- **Task 5 (Additional Tools):**
  - Added `dataset_stats` tool in `hw2/src/tools.py` to fetch summary statistics and value counts directly from the HECATE dataset.
  - Implemented `calculator` tool for evaluating numerical expressions and unit conversions using Python's `math` module.
  - Implemented `csv_lookup` tool using Pandas `query()` to allow the agent to filter and retrieve specific galaxy records.
  - Bound all new tools to the `gemini-2.0-flash` LangGraph agent and updated the system prompt to instruct the agent on their specific use-cases.
- **Task 6 (Streaming Output):**
  - Added `stream_agent` async generator in `hw2/src/agent.py` using LangGraph's `.astream_events(version="v2")` to yield real-time LLM tokens.
  - Implemented a `POST /chat/stream` endpoint in `hw2/src/api.py` utilizing FastAPI's `StreamingResponse` to serve tokens as Server-Sent Events (SSE).

## dev-01:42
### Conversational Agent (Task 3) and REST API (Task 4)
- **Task 3 (Conversational Agent):**
  - Created `hw2/src/agent.py` using `langgraph` for state graph coordination.
  - Linked the RAG retrieval function as a `@tool` (`retrieve_domain_knowledge`) and the galaxy classifier (`predict_galaxy_class`).
  - Set up `ChatGoogleGenerativeAI` bound with tools using model `"gemini-2.5-flash"`.
  - Added session history tracking using LangGraph's `MemorySaver()` for cross-turn memory.
  - Resolved `sys.path` import issues to support executing scripts from the repository root.
  - Fixed a response structure validation issue (where Gemini returned a list of dictionary blocks instead of a string) by introducing extraction logic.
- **Task 4 (REST API Integration):**
  - Implemented `hw2/src/api.py` with FastAPI exposing a `POST /chat` endpoint and a `GET /health` endpoint.
  - Modified `hw2/main.py` as the uvicorn entrypoint to launch the server.
  - Verified the API using curl requests, successfully validating output structure and end-to-end tool calling.

## dev-00:38
### RAG Document Ingestion and Persistent Vector Store
- Initialized `hw2` project structure including `main.py` and `src/` modules.
- Added RAG dependencies via Poetry: `langchain`, `langchain-community`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, and `pypdf`.
- Implemented `hw2/src/rag.py` with `ingest_documents` using `RecursiveCharacterTextSplitter` and `Chroma` vector store.
- Successfully loaded, chunked (size 1000, overlap 200), and embedded 1321 document pages using the local `all-MiniLM-L6-v2` model.
- Saved embeddings to a persistent `ChromaDB` index on disk (`hw2/data/vector_store`).
- Configured a `.gitignore` to prevent tracking of the large vector store and `.venv` directory.

## dev-01:14
### RAG Context Retrieval and HW1 Model Tool Integration
- **Task 1.3:** Added `retrieve_context(query, k)` to `hw2/src/rag.py` to fetch and concatenate the most relevant chunks into a single string for the LangGraph agent.
- **Deprecation Fixes:** Fixed `langchain_community` warnings by migrating to the standalone `langchain-chroma` package and configured the `logging` module to suppress harmless `MacExpertEncoding` warnings from `pypdf`.
- **Model Migration:** Copied `best_model.pkl`, `scaler.pkl`, `iqr_bounds.pkl`, `label_encoder.pkl`, and `imputation_pipeline.pkl` from HW1 to `hw2/models/`.
- **Environment Fix:** Downgraded `scikit-learn` via Poetry to `1.7.2` to resolve unpickling version mismatches without needing to retrain the HW1 model.
- **Task 2:** Implemented `predict_galaxy_class` in `hw2/src/tools.py` using a strict Pydantic `GalaxyPredictionInput` schema for LangGraph. The tool dynamically constructs a Pandas DataFrame, applies the exact HW1 preprocessing pipeline (IQR, imputation, error filtering, colors, scaling), and outputs human-readable classification probabilities.


# 2026-04-09
 
 Implemented functionality to drop specific class groups from the dataset, ensuring the pipeline is robust to dimension changes by automatically re-training models when mismatches occur. Expanded unit tests to cover the new filtering logic and configuration attributes.
 
 ## dev-04:06
 ### Implementing Class Group Filtering and Model Robustness
 - Added `drop_group` attribute to `DataConfig` in `src/config.py` to allow targeted data subsetting.
 - Implemented `drop_specific_group` in `src/preprocessing.py` and integrated it into the `main.py` ingestion pipeline.
 - Enhanced `main.py` with robustness checks in `run_classical_training` and `run_neural_training` to detect dimension mismatches and re-train models automatically.
 - Added comprehensive unit tests in `tests/test_preprocessing.py` and `tests/test_config.py` covering the new group filtering logic and configuration defaults.
 - Verified the end-to-end pipeline execution and confirmed automatic re-training upon configuration changes.
 
 # 2026-04-09 (Previous)

Addressed significant class imbalance by implementing SMOTE (Synthetic Minority Over-sampling Technique) into the preprocessing pipeline. Successfully integrated the `imbalanced-learn` library, updated the configuration to handle sampling strategies, and verified the balancing effect where the training set now perfectly distributes samples across all target classes.

## dev-00:35
### Handling Class Imbalance with SMOTE
- Integrated `imbalanced-learn` into the dependency manifest via Poetry.
- Added `SamplingConfig` to `src/config.py` to allow configurable oversampling strategies and SMOTE parameters.
- Implemented `apply_smote` in `src/preprocessing.py`, ensuring synthetic samples are only generated for the training split.
- Updated `hw1/main.py` to automate training set balancing and print the new class ratios for verification.

## dev-00:42
### Test Suite Expansion and CI/CD Stabilization
- Developed `hw1/tests/test_api.py` to provide 100% test coverage for the FastAPI deployment layer.
- Stabilized `hw1/tests/test_main.py` by refining mock datasets and configuration to support SMOTE logic.
- Implemented `test_apply_smote` in the preprocessing unit tests grid resampling.
- Enhanced `src/train_neural.py` with runtime thread-configuration protection.

## dev-00:44
### Visual Layout Optimization for Model Comparison
- Refactored `src/evaluation.py` to support a dynamic grid layout (defaulting to 2 columns) for confusion matrices and model metrics, replacing the monolithic row format.
- Verified the new 2x2 grid layout through pipeline execution and unit testing.
- Fixed a `RuntimeError` in neural network training related to torch thread configuration during parallel testing.
- Stabilized the main execution pipeline to ensure consistency when SMOTE is applied to small datasets by refining test mocks.

# 2026-04-08

The day focused on transitioning the machine learning pipeline from a batch-processing research tool to a production-ready system. Initial efforts centralized evaluation and visualization logic, followed by the implementation of a FastAPI deployment layer. Successfully exposed the best-performing models via a REST API, ensuring full parity between training and inference preprocessing.

## dev-00:14
### Centralizing Evaluation Logic and Modularization
- Created `hw1/src/evaluation.py` to centralize all evaluation and visualization logic (ROC-AUC, PR Curves, Confusion Matrices, Metric Bar Plots).
- Refactored `hw1/src/train_classical.py` and `hw1/src/train_neural.py` to use the shared evaluation module, reducing code duplication.
- Updated `hw1/main.py` to leverage the centralized evaluation tools for final model comparisons.
- Expanded the test suite with `hw1/tests/test_evaluation.py` and updated existing tests to align with the new modular structure.
- Ensured 100% pass rate for the refactored evaluation pipeline.

## dev-23:30
### Model Deployment and FastAPI Integration (Task 5)
- Created `hw1/src/api.py` implementing a FastAPI REST endpoint to expose the best performing model.
- Developed a `/predict` POST endpoint using Pydantic for input validation and automatic documentation.
- Integrated the saved preprocessing pipeline (Scaler, LabelEncoder, Imputer, IQR bounds) into the inference logic for feature parity.
- Enhanced `hw1/main.py` to persist all necessary preprocessing artifacts (`imputation_pipeline.pkl`, `iqr_bounds.pkl`).
- Added a `hw1/requirements.txt` manifest documenting deployment dependencies.
- Verified the production-ready API through the Swagger UI with successful real-time inference tests.

# 2026-04-07

The focus was on modularizing and centralizing the machine learning pipeline's configuration, implementing a fully functional Neural Network pipeline, and refactoring the main orchestration logic. Introduced Poetry for dependency management, established a project-wide README, and stabilized the configuration test suite. Achieved a 100% test pass rate (67 unit tests) in the new orchestrated environment.

## dev-02:12
### Centralizing Pipeline Configuration and Testing
- Extracted hyperparameter and path definitions into a new `PipelineConfig` dataclass in `src/config.py`.
- Refactored `src/preprocessing.py` and `src/train_classical.py` to utilize `PipelineConfig` as default parameter values.
- Implemented a `src/` package structure with `src/__init__.py`.
- Added `tests/test_config.py` to validate configuration constraints and derivation logic.
- Standardized function docstrings and resolved syntax issues in the training module.
- Successfully executed the complete test suite with 55 passed unit tests.

## dev-02:44
### Configuration Refinement and Test Data Integration
- Implemented robust relative path resolution in `src/config.py` using `__file__`.
- Created `hw1/data/HECATE_test.csv` as a lightweight dataset for rapid pipeline validation.
- Transitioned default pipeline configuration to utilize the test dataset.
- Simplified `main.py` by removing redundant path overrides.
- Updated `tests/test_config.py` and modularized configuration components into specialized sub-classes.
- Verified end-to-end pipeline execution with the new development configuration.

## dev-23:15
### Neural Network Pipeline Implementation and Integration
- Implemented `src/train_neural.py` with `SimpleNN` architecture and `EarlyStopping` logic.
- Integrated `NNConfig` into the centralized configuration to manage all neural network hyperparameters.
- Updated `main.py` to orchestrate a fair comparison between classical models (Random Forest, XGBoost, etc.) and the Neural Network.
- Developed 10 new unit tests in `tests/test_train_neural.py` covering model architecture, training loops, and evaluation.
- Resolved NumPy 2.0 compatibility issue by updating deprecated `np.Inf` to `np.inf`.
- Verified the complete project with 62 passing unit tests.

## dev-23:22
### Modular Refactoring of Main Pipeline and Test Suite Expansion
- Refactored `main.py` into distinct, single-responsibility phases: Ingestion/Preprocessing, Classical Training, Neural Training, and Final Evaluation.
- Introduced `run_data_ingestion_and_preprocessing`, `run_classical_training`, `run_neural_training`, and `evaluate_and_save_best_model` orchestrators.
- Implemented extensive unit tests in `tests/test_main.py` to validate each pipeline phase independently.
- Stabilized the main orchestration logic with mock-based testing, ensuring robust component interaction.
- Verified the integrity of the refactored pipeline with 16 dedicated orchestration tests, all passing.

## dev-23:52
### Dependency Management and Project Initialization
- Initialized `pyproject.toml` and `poetry.lock` for automated dependency management.
- Set up root-level `README.md` to provide project context and structure.
- Configured Poetry to utilize `package-mode = false`, facilitating loose assignment orchestration.
- Re-stabilized `hw1/tests/test_config.py` by resolving attribute failures related to dataclass field defaults.
- Verified that the entire project (67 unit tests) passes successfully within the managed Poetry environment.
