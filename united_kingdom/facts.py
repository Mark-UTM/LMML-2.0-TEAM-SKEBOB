import collections
import re

SLOTS = {
    "eat": {"coffee shop": r"coffee ?shop|caf[eé](?! (rouge|brazil|adriatic|sicilia))", "pub": r"\bpubs?\b",
            "restaurant": r"restaurant"},
    "food": {"japanese": r"japanese|sushi", "english": r"english|british", "italian": r"italian|pasta|pizza",
             "french": r"french", "chinese": r"chinese", "indian": r"\bindian(?! cuisine)", "fast": r"fast ?food"},
    "price": {"low": r"cheap|low[- ]?(cost|price|priced)|inexpensive|affordable|less than £? ?20|under £? ?20|budget",
              "mid": r"moderate|average[- ]?pric|mid[- ]?(range|price)|medium[- ]?pric|£? ?20 ?- ?25|reasonabl",
              "high": r"expensive|high[- ]?(end|price|priced|cost)|pricey|costly|more than £? ?30|over £? ?30|upscale"},
    "rating": {"1": r"1 out of 5|one star|1 star|low(ly)?[- ]rat|poor(ly)?[- ]rat|low customer|rated low|poor",
               "3": r"3 out of 5|three star|3 star|average (customer )?rat|rated average|average review|average$",
               "5": r"5 out of 5|five star|5 star|high(ly)?[- ]rat|high customer|rated high|excellent|top rated"},
    "area": {"riverside": r"riverside|river", "centre": r"city cent|centre|center"},
    "family": {"yes": r"(family|kids?|child|children)[- ]?friendly|families|welcomes? (children|kids|families)",
               "no": r"not (family|kid|child|children)|no (kids|children|families)|adults? only|isn't family|non[- ]family"},
}

def mentions(text):
    t = text.lower()
    out = {}
    for slot, vals in SLOTS.items():
        hits = {v: len(re.findall(p, t)) for v, p in vals.items()}
        if slot == "family" and hits.get("no"):
            hits["yes"] = 0
        out[slot] = {v: n for v, n in hits.items() if n}
    near = re.findall(r"\bnear(?: to)? (?:the )?([A-Z][\w']+)", text)
    out["near"] = collections.Counter(n.lower() for n in near)
    return out

def card_mr(refs):
    votes = collections.defaultdict(collections.Counter)
    for r in refs:
        for slot, vals in mentions(r).items():
            for v in vals:
                votes[slot][v] += 1
    return {s: c.most_common(1)[0][0] for s, c in votes.items() if c and c.most_common(1)[0][1] >= 3}

def check(text, mr):
    m = mentions(text)
    missing = sum(1 for s, v in mr.items() if v not in m.get(s, {}))
    wrong = sum(1 for s, v in mr.items() if m.get(s) and v not in m[s])
    repeated = sum(1 for s, v in mr.items() if m.get(s, {}).get(v, 0) >= 2)
    return dict(n_slots=len(mr), missing=missing, wrong=wrong, repeated=repeated)
