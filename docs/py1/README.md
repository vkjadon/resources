
---
This article explains how to install **Python on macOS**, set up **VS Code**, and create a **virtual environment** for your projects.

---

## Step 1: Check if Python Is Already Installed

Open **Terminal** (`Cmd + Space → type terminal → Enter`) and run:

```bash
python3 --version
```

If you see something like:

```
Python 3.11.6
```

Python is already installed.  
If not, continue to the next step.
---

## Step 2: Install Python

### Option 1 — Install via Homebrew (Recommended)

If you don’t have **Homebrew**, install it first:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python:

```bash
brew install python
```

Check version:

```bash
python3 --version
```

### Option 2 — Install from Python.org

Go to [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/)  
Download the latest macOS installer and follow the on-screen steps.

Verify installation:

```bash
python3 --version
```
---

## Step 3: Install VS Code

Download and install from:  
[https://code.visualstudio.com/](https://code.visualstudio.com/)

---

## Step 4: Install the Python Extension

In VS Code:

1. Press `Cmd + Shift + X`
2. Search for **"Python"**
3. Install the one published by **Microsoft**
4. (Optional) Install **Pylance** for enhanced IntelliSense.

---

## Step 5: Run Python Code in VS Code

### Option 1 — Run in Terminal

1. Create a new file:  
   `File → New File → Save as hello.py`
2. Add code:
   ```python
   print("Hello, Python on Mac!")
   ```
3. Open a terminal (`Ctrl + ~` or `Cmd + ~`) and run:
   ```bash
   python3 hello.py
   ```

### Option 2 — Use the “Run” Button

After installing the Python extension, click the **▶ Run** button in the editor’s top-right corner.

---

## Step 6: Select Python Interpreter

If VS Code doesn’t detect Python:

1. Press `Cmd + Shift + P`
2. Search **“Python: Select Interpreter”**
3. Choose the correct version (e.g., Python 3.11).
---

# Setting Up a Virtual Environment in VS Code (Mac)
---

## Step 1: Create a Project Folder

```bash
mkdir python_project
cd python_project
```

Open in VS Code:

```bash
code .
```
---

## Step 2: Create a Virtual Environment

```bash
python3 -m venv env
```


This creates a folder named `env` containing an isolated Python environment.

---

## Step 3: Activate the Environment

```bash
source env/bin/activate
```

If successful, your prompt will look like:

```
(env) vijay@MacBook python_project %
```
---

## Step 4: Install Packages

```bash
pip install numpy pandas matplotlib
```

Check installed packages:

```bash
pip list
```
---

## Step 5: Set the Interpreter in VS Code

If VS Code doesn’t automatically detect the virtual environment:

1. Press `Cmd + Shift + P`
2. Search **Python: Select Interpreter**
3. Choose the one showing your project’s `env` path (`./env/bin/python`).

---

## Step 6: Test Script

Create `test_env.py`:

```python
import numpy as np
import pandas as pd

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

data = pd.DataFrame({
    "A": np.random.rand(3),
    "B": np.random.rand(3)
})
print("\nDataFrame:\n", data)
```

Run it:

```bash
python test_env.py
```
---

## Step 7: Deactivate the Virtual Environment

```bash
deactivate
```

You’ll return to your system Python environment.

---

## Bonus Tip: Save and Reinstall Packages

Export installed packages:

```bash
pip freeze > requirements.txt
```

Reinstall later:

```bash
pip install -r requirements.txt
```
---
