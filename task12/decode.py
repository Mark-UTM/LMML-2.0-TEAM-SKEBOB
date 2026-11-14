import csv, collections, math, os

ROOT = os.path.dirname(os.path.abspath(__file__))
T, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'out')
reg = list(csv.DictReader(open(os.path.join(T, 'ring_register.csv'), encoding='utf-8')))
ALPHA = sorted({c for g in reg for c in g['ring']})
cand = [r['ring_read'] for r in csv.DictReader(open(os.path.join(OUT, 'stork_candidates.csv'), encoding='utf-8'))]

def learned_confusion():
    obs = list(csv.DictReader(open(os.path.join(T, 'observations.csv'), encoding='utf-8')))
    km = lambda a1, o1, a2, o2: 111.2 * math.hypot(float(a1) - float(a2), (float(o1) - float(o2)) * math.cos(math.radians(46.5)))
    bysp = collections.defaultdict(list)
    for g in reg:
        bysp[g['species']].append(g)
    cnt = {t: collections.Counter() for t in ALPHA}
    for x in obs:
        if not x['ring_read']:
            continue
        near = [g for g in bysp[x['species']]
                if km(x['latitude'], x['longitude'], g['ringed_latitude'], g['ringed_longitude']) < 1
                and x['date'][:4] >= g['ringed_year']]
        if len(near) == 1:
            for a, t in zip(x['ring_read'], near[0]['ring']):
                if a != '?':
                    cnt[t][a] += 1
    return {t: {a: (cnt[t][a] + 0.1) / (sum(cnt[t].values()) + 0.1 * len(ALPHA)) for a in ALPHA} for t in ALPHA}

def load_confusion(path):
    P = {}
    for row in csv.DictReader(open(path, encoding='utf-8')):
        t = row.get('true') or row.get('') or next(iter(row.values()))
        probs = {a: float(row[a]) for a in ALPHA if a in row and row[a] != ''}
        s = sum(probs.values())
        P[t] = {a: max(probs.get(a, 0) / s, 1e-6) for a in ALPHA}
    return P

def decode(reads, P):
    word, detail = '', []
    for i in range(6):
        letters = [r[i] for r in reads if r[i] != '?']
        ll = {t: sum(math.log(P[t][a]) for a in letters) for t in ALPHA}
        ranked = sorted(ll, key=ll.get, reverse=True)
        word += ranked[0]
        detail.append((i + 1, ranked[0], ranked[1], ll[ranked[0]] - ll[ranked[1]], collections.Counter(letters).most_common()))
    return word, detail

cpath = os.path.join(OUT, 'confusion.csv')
models = {'learned here': learned_confusion()}
if os.path.exists(cpath):
    models['confusion.csv'] = load_confusion(cpath)

print(f'{len(cand)} candidate reads')
answers = set()
for name, P in models.items():
    word, detail = decode(cand, P)
    answers.add(word)
    print(f'\n[{name}] ML decode: {word}')
    for pos, best, second, margin, seen in detail:
        print(f'  pos {pos}: {best} (vs {second}: log-lik margin {margin:.1f})  reads: {seen}')
    loo = collections.Counter(decode(cand[:k] + cand[k + 1:], P)[0] for k in range(len(cand)))
    print(f'  leave-one-out: {dict(loo)}')

vote = ''.join(collections.Counter(r[i] for r in cand if r[i] != '?').most_common(1)[0][0] for i in range(6))
print(f'\nplain majority vote (no model): {vote}')
answers.add(vote)
ring = sorted(answers)[0]
print('in register:', any(g['ring'] == ring for g in reg))
print('all methods agree:', len(answers) == 1)
print(f'FLAG: SIGMOID_{ring}' if len(answers) == 1 else f'methods disagree: {answers}')
