# STORM FastHTML User Interface

This is a user interface for STORM that allows you to interact with the STORM investor model.

To start with it, we have added the necessary dependencies to the `pyproject.toml` file (in the main directory).

Ensure that you have a file `.env` in the main directory containing the following:
```
OPENAI_API_KEY=<your-openai-api-key>
YDC_API_KEY=<your-ydc-api-key>
```
These are needed to run the application.

To install the dependencies, run `poetry install` in the main directory.

To run the application, activate the shell: `poetry shell`

Then, either:
- navigate to the `frontend/fasthtml` directory and run `python storm_fasthtml.py`.
- run `python frontend/fasthtml/storm_fasthtml.py` from the main directory.
