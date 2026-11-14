import csv, collections, math, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)
obs = list(csv.DictReader(open(os.path.join(D, 'observations.csv'), encoding='utf-8')))
reg = list(csv.DictReader(open(os.path.join(D, 'ring_register.csv'), encoding='utf-8')))
ALPHA = sorted({c for g in reg for c in g['ring']})

def km(a1, o1, a2, o2):
    return 111.2 * math.hypot(float(a1) - float(a2), (float(o1) - float(o2)) * math.cos(math.radians(46.5)))

def mism(read, ring):
    return sum(1 for a, b in zip(read, ring) if a != '?' and a != b)

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
    if len(near) != 1:
        continue
    for a, t in zip(x['ring_read'], near[0]['ring']):
        if a != '?':
            cnt[t][a] += 1
P = {t: {a: (cnt[t][a] + 0.1) / (sum(cnt[t].values()) + 0.1 * len(ALPHA)) for a in ALPHA} for t in ALPHA}
BG = collections.Counter(c for x in obs for c in x['ring_read'] if c != '?')
nbg = sum(BG.values())
PB = {a: BG[a] / nbg for a in ALPHA}

def llr(read, ring):
    return sum(math.log(P[t][a] / PB[a]) for a, t in zip(read, ring) if a != '?')

SP = 'Ciconia ciconia'
storks = bysp[SP]
rows = []
for x in obs:
    if not x['ring_read'] or x['species'] != SP:
        continue
    sc = []
    for g in storks:
        d = km(x['latitude'], x['longitude'], g['ringed_latitude'], g['ringed_longitude'])
        sc.append((llr(x['ring_read'], g['ring']), d, g['ring'], x['date'][:4] >= g['ringed_year']))
    best = max(sc)
    nearest = min(sc, key=lambda s: s[1])
    rows.append(dict(x=x, best=best, nearest=nearest))

print('stork reads:', len(rows))
print('LLR of best register stork vs km to its site (buckets):')
for lo, hi in [(-99, 0), (0, 3), (3, 6), (6, 99)]:
    sel = [r for r in rows if lo <= r['best'][0] < hi]
    far = sum(1 for r in sel if r['nearest'][1] > 10)
    print(f'  LLR in [{lo},{hi}): {len(sel)} reads, {far} >10 km from every registered stork')

sites = collections.defaultdict(list)
for r in rows:
    sites[(r['x']['latitude'], r['x']['longitude'])].append(r)
print('\nreading sites >10 km from every registered stork site:')
for (la, lo), rs in sorted(sites.items(), key=lambda kv: -len(kv[1])):
    if rs[0]['nearest'][1] > 10:
        print(f'  {la},{lo} {rs[0]["x"]["district"]}: {len(rs)} reads, nearest reg {rs[0]["nearest"][2]} at {rs[0]["nearest"][1]:.0f} km;',
              ' '.join(r['x']['ring_read'] for r in rs))

cols = ['record_id', 'checklist_id', 'date', 'latitude', 'longitude', 'observers', 'ring_read',
        'best_register_ring', 'best_dist', 'km_to_ring_site', 'reason']
with open(os.path.join(OUT, 'stork_candidates.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        x, best, nearest = r['x'], r['best'], r['nearest']
        if best[0] < 0 and nearest[1] > 10:
            bd = min(mism(x['ring_read'], g['ring']) for g in storks)
            w.writerow(dict(record_id=x['record_id'], checklist_id=x['checklist_id'], date=x['date'],
                            latitude=x['latitude'], longitude=x['longitude'], observers=x['observers'],
                            ring_read=x['ring_read'], best_register_ring=best[2], best_dist=bd,
                            km_to_ring_site=round(nearest[1], 1),
                            reason=f'best register LLR {best[0]:.1f} < 0; nearest registered stork {nearest[2]} is {nearest[1]:.0f} km away'))
print('wrote', os.path.join(OUT, 'stork_candidates.csv'))
