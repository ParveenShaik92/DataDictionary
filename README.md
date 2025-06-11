To run with default arguments:
```bash
python main.py
(This will read input.csv and write output.csv)
```
To read a JSON file and output to CSV:
```bash
python main.py --input-format json --input-file input.json --output-format csv --output-file my_json_output.csv
```
To read a CSV file and print to console:

```bash
python main.py --input-format csv --input-file input.csv --output-format console
```
To ingest an SQL dump (which writes directly to a SQLite DB):
```bash
python main.py --input-format sql_dump --input-file input.sql --output-file my_dumped_db.db
```