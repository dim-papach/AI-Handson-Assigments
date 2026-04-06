# 2026-04-06

Initial commit of the homework assignment, including project documentation, poetry dependency management, and refactored pipeline structure.

## dev-04:18
### Initial project structure and pipeline refactoring
- Added `pyproject.toml` and `poetry.lock` for dependency management.
- Included homework assignment outline in `assignment.txt`, `README.md`, and `1st Assignment.pdf`.
- Refactored `main.py` to break down the monolithic pipeline into modular functions.
- Extracted preprocessing logic into an independent pipeline inside `src/preprocessing.py`.
- Created tests in `tests/test_main.py` and `tests/test_preprocessing.py`.
