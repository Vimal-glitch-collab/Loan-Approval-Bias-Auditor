# Step 1 setup (Windows PowerShell)

Open a terminal in this project folder. Use an installed Python 3.10+ interpreter.

```powershell
# Create an isolated environment.
python -m venv .venv

# Activate it for this terminal.
.\.venv\Scripts\Activate.ps1

# Install the full project stack.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Initialize local version control; this does not publish anything.
git init

# Download once, inspect, clean, and save the dataset.
python src/load_data.py

# Record the versions that successfully installed.
python -m pip freeze > requirements-lock.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in the pip and loader commands. There is no need to change execution policy.

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`; subsequent commands are the same.

If downloading fails, download the ZIP from https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data and put its `german.data` file inside `data/`. Rerun the loader. Do not use `german.data-numeric`, which has a different schema.

Time budget: 30 minutes including installation and reviewing the data limitations. Stop after this step; no model training is included.
