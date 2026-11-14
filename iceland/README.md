# What The Data? (Iceland)

Cleaner for a damaged IMO geothermal station export. `main.py` reads `test-input.csv` from the working directory and writes
`test-output.csv` with the columns `reading_id,station_name,temperature_c,reading_date`.

## What it fixes
Each problem is handled generically, not hard-coded for the sample:
- a preamble of `#` comment or blank lines before the header (any length), an optional BOM and CRLF line endings;
- a `;` delimiter, decimal commas, stray whitespace and mixed unit case;
- Fahrenheit readings, converted to Celsius with exact arithmetic and rounded half-up to one decimal;
- mojibake, where UTF-8 text was decoded as cp1252/latin-1 once or several times; it is undone repeatedly;
- retransmitted duplicates (the same `reading_id`, possibly in the other unit), kept as one row per `reading_id`.

It uses the standard library only.

## Run
```
cd iceland
cp train-input.csv test-input.csv   # or the real test input
python main.py                      # -> test-output.csv
```
