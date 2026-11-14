import unicodedata
from pathlib import Path
import pandas as pd

SV_FI = {
    "abo": "turku", "alavo": "alavus", "birkala": "pirkkala", "bjorneborg": "pori", "borga": "porvoo",
    "brahestad": "raahe", "esbo": "espoo", "fredrikshamn": "hamina", "grankulla": "kauniainen", "hango": "hanko",
    "helsingfors": "helsinki", "hogfors": "karkkila", "hyvinge": "hyvinkaa", "idensalmi": "iisalmi",
    "ikalis": "ikaalinen", "ilmola": "ilmajoki", "inga": "inkoo", "jakobstad": "pietarsaari", "kajana": "kajaani",
    "karleby": "kokkola", "kasko": "kaskinen", "kervo": "kerava", "keuru": "keuruu", "kimitoon": "kemionsaari",
    "korsholm": "mustasaari", "kristinestad": "kristiinankaupunki", "kronoby": "kruunupyy", "kumo": "kokemaki",
    "kyrkslatt": "kirkkonummi", "lahtis": "lahti", "lappo": "lapua", "larsmo": "luoto", "lojo": "lohja",
    "lovisa": "loviisa", "lundo": "lieto", "malax": "maalahti", "nadendal": "naantali", "narpes": "narpio",
    "nykarleby": "uusikaarlepyy", "nystad": "uusikaupunki", "pargas": "parainen", "raseborg": "raasepori",
    "raumo": "rauma", "reso": "raisio", "s:t karins": "kaarina", "s:t michel": "mikkeli", "sibbo": "sipoo",
    "sjundea": "siuntio", "tammerfors": "tampere", "tavastehus": "hameenlinna", "tornea": "tornio",
    "traskanda": "jarvenpaa", "tusby": "tuusula", "uleaborg": "oulu", "ulvsby": "ulvila", "vanda": "vantaa",
    "vasa": "vaasa", "vichtis": "vihti", "villmanstrand": "lappeenranta", "virmo": "mynamaki", "vora": "voyri",
}

def key(name):
    s = name.strip().rstrip(".").strip().casefold()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return SV_FI.get(s, s)

d = pd.read_csv(Path(__file__).resolve().parent / "routes.csv")
d["a"], d["b"] = d.from_town.map(key), d.to_town.map(key)
edges = {tuple(sorted(p)) for p in zip(d.a, d.b) if p[0] != p[1]}
towns = sorted({t for e in edges for t in e} | set(d.a) | set(d.b))
deg = pd.Series([t for e in edges for t in e]).value_counts()
print(f"towns {len(towns)}, connections {len(edges)}  (expected 116 / 448)")
print("raw group-by on from_town says:", d.from_town.value_counts().index[0])
print(deg.head(10).to_string())
assert deg.head(8).is_unique
print("FLAG: SIGMOID_" + "".join(t[0].upper() for t in deg.index[:7]))
