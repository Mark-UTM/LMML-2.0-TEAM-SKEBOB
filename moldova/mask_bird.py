import csv, collections, math, os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
T, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'out')
obs = list(csv.DictReader(open(os.path.join(T, 'observations.csv'), encoding='utf-8')))
reg = list(csv.DictReader(open(os.path.join(T, 'ring_register.csv'), encoding='utf-8')))
GROUPS = ['AKRX', 'BGZS', 'CODQ', 'EFLP', 'HWNM', 'IYJT']
CODE = {c: i for g in GROUPS for i, c in enumerate(g)}
GRP = {c: k for k, g in enumerate(GROUPS) for c in g}
sig = lambda s: tuple(GRP.get(c, '?') for c in s)
bysig = collections.defaultdict(list)
for g in reg:
    bysig[sig(g['ring'])].append(g)

def candidates(read):
    s = sig(read)
    return [g for k, gs in bysig.items() for g in gs if all(a == '?' or a == b for a, b in zip(s, k))]

def km(a1, o1, a2, o2):
    return 111.2 * math.hypot(float(a1) - float(a2), (float(o1) - float(o2)) * math.cos(math.radians(46.5)))

rows = []
for x in obs:
    if not x['ring_read']:
        continue
    c = candidates(x['ring_read'])
    if len(c) != 1:
        continue
    g = c[0]
    ms = [CODE[a] ^ CODE[t] for a, t in zip(x['ring_read'], g['ring']) if a != '?']
    if not ms:
        continue
    cnt = collections.Counter(ms)
    m = cnt.most_common(1)[0][0]
    rows.append(dict(x=x, g=g, mask=m, pure=cnt[m] / len(ms), n=len(ms),
                     dist=km(x['latitude'], x['longitude'], g['ringed_latitude'], g['ringed_longitude'])))

if __name__ == '__main__':
    print('assigned reads', len(rows), ' read-level purity', round(sum(r['pure'] for r in rows) / len(rows), 3))
    print('global mask counts', collections.Counter(r['mask'] for r in rows))
    bird = collections.defaultdict(list)
    for r in rows:
        bird[r['g']['ring']].append(r)
    print('\nper bird: ring species year lat lon | n reads | mask counts 0/1/2/3')
    for ring, rs in sorted(bird.items(), key=lambda kv: (kv[1][0]['g']['species'], kv[0])):
        g = rs[0]['g']; c = collections.Counter(r['mask'] for r in rs)
        print(f"{ring} {g['species'][:14]:14} {g['ringed_year']} {g['ringed_latitude']} {g['ringed_longitude']} | {len(rs):3} | "
              + ' '.join(f'{c[k]:3}' for k in range(4)))
