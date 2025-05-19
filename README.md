# blue-light-aware-color-constancy
This project uses [Poetry](https://python-poetry.org/) to manage the Python environment and dependencies.

##  Getting Started

### 1. Install Poetry

If you don't have Poetry installed, run:

```bash
pip install poetry
````

---

###  2. Install Dependencies

From the project root directory, install all required packages without installing the current project as a package:

```bash
poetry install --no-root
```

---

###  3. Add New Dependencies

To add a new package (e.g., `pandas`), use:

```bash
poetry add pandas
```

---

###  4. Activate the Virtual Environment

To find the path of the virtual environment, run:

```bash
poetry env info
```

You should see output like this:

```
Virtualenv
Python:         3.10.12
Implementation: CPython
Path:           /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10
Executable:     /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10/bin/python
Valid:          True
```

To activate the environment:

```bash
source /home/username/.cache/pypoetry/virtualenvs/blue-light-aware-color-constancy-xxxxxxxx-py3.10/bin/activate
```

> Replace the path with the one shown on your machine.

---

##  Running Scripts

Once activated, you can run your Python scripts as usual:

```bash
python your_script.py
```

Or run them directly via Poetry without activating:

```bash
poetry run python your_script.py
```

## Exit the environment

```bash
deactivate
```
