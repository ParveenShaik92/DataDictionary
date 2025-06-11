# DataDictionary

A Python library and command-line tool for generating data dictionaries from CSV files and outputting them in various formats like JSON or console.

# Folder Structure

DataDictionary/
├── format/
│ └── csv_formatter.py # Contains logic to parse CSV files
├── output/
│ └── json_output.py # Handles JSON output
├── main.py # CLI entry point
├── init.py # Package init file

# Installation

```bash
git clone https://github.com/your-username/data-dictionary.git
cd data-dictionary

pip install spacy pandas numpy spacy-transformers
python -m spacy download en_core_web_trf
```


# Usage
To run with default arguments:
```bash
python main.py
(This will read input.csv and write output.csv)
```
To read a JSON file and output to CSV:
```bash
python main.py --input-format csv --input-file input.csv --output-format json --output-file output.json
```
To read a CSV file and print to console:

```bash
python main.py --input-format csv --input-file input.csv --output-format console
```
To ingest an SQL dump (which writes directly to a SQLite DB):
```bash
python main.py --input-format sql_dump --input-file input.sql --output-file my_dumped_db.db
```

# Dependencies
Python 3.12.2+
Standard Library (no external libraries currently required)