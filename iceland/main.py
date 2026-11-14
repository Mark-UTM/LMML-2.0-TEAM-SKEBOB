import csv
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

IN, OUT = "test-input.csv", "test-output.csv"

def to_bytes_legacy(s):
    out = bytearray()
    for ch in s:
        o = ord(ch)
        if o < 256:
            out.append(o)
        else:
            out += ch.encode("cp1252")
    return bytes(out)

def fix_mojibake(s):
    for _ in range(10):
        if all(ord(c) < 128 for c in s):
            return s
        try:
            t = to_bytes_legacy(s).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
        if t == s:
            return s
        s = t
    return s

def parse_temp(raw):
    raw = raw.strip().replace(" ", "").replace(",", ".")
    return Fraction(Decimal(raw))

def fmt1(x):
    q = Decimal(x.numerator) / Decimal(x.denominator)
    return str(q.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def main():
    with open(IN, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    body = [ln for ln in lines[i:] if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.reader(body, delimiter=";"))
    header = [h.strip().lower() for h in rows[0]]
    col = {name: header.index(name) for name in ("reading_id", "station_name", "temperature_raw", "unit", "reading_date")}

    seen = {}
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        rid = int(r[col["reading_id"]].strip())
        if rid in seen:
            continue
        t = parse_temp(r[col["temperature_raw"]])
        unit = r[col["unit"]].strip().upper().lstrip("°")
        if unit.startswith("F"):
            t = (t - 32) * Fraction(5, 9)
        seen[rid] = (rid, fix_mojibake(r[col["station_name"]].strip()), fmt1(t), r[col["reading_date"]].strip())

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("reading_id,station_name,temperature_c,reading_date\n")
        for rid in sorted(seen):
            f.write(",".join(str(v) for v in seen[rid]) + "\n")

if __name__ == "__main__":
    main()
