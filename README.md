# DataDictionary

A Python library and command-line tool for generating data dictionaries from CSV files and outputting them in various formats like JSON or console.

---

## 📂 Folder Structure

```plaintext
DataDictionary/
├── format/
│   └── csv_formatter.py     # Contains logic to parse CSV files
├── output/
│   └── json_output.py       # Handles JSON output
├── process/
│   └── data_processor.py    # Data processing logic (spaCy, ML, etc.)
├── utils/
│   └── dd_genai.py          # Google Generative AI integration
├── main.py                  # CLI entry point
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
```

---

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/ParveenShaik92/DataDictionary.git
cd DataDictionary
```

### 2. Create a virtual environment (Python 3.12 recommended)
```bash
py -3.12 -m venv venv312
venv312\Scripts\activate   # On Windows
# source venv312/bin/activate   # On Linux/Mac
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the spaCy language model
```bash
python -m spacy download en_core_web_trf
```

---

## ▶️ Usage

### Default run
```bash
python main.py
# Reads input.csv and prints to console (default)
```

### Read CSV and output JSON
```bash
python main.py --input-format csv --input-file input.csv --output-format json --output-file output.json
```

### Read CSV and print to console
```bash
python main.py --input-format csv --input-file input.csv --output-format console
```

---

## 📦 Dependencies

Defined in `requirements.txt`:
- spacy
- pandas
- numpy
- scikit-learn
- spacy-transformers
- transformers[torch]
- accelerate
- google-generativeai

Additional:
- Python **3.12.x** (recommended)
- spaCy model `en_core_web_trf`

---

## 📝 Notes
- If you face issues with `en_core_web_trf` (large model), you can switch to the lighter model:
  ```python
  nlp = spacy.load("en_core_web_sm")
  ```
  Install it with:
  ```bash
  python -m spacy download en_core_web_sm
  ```
