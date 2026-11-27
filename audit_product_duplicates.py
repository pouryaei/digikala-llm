"""Run from project3. Read products only; no changes to raw data."""
import csv
import json
from hashlib import blake2b
from pathlib import Path


def fingerprint(values):
    return blake2b(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode(), digest_size=16).digest()


def audit(path):
    first = {}
    repeated, changed, identity, price, seller = (set() for _ in range(5))
    extra = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        key, cost, shop, *attrs = map(header.index, ('id', 'Price', 'Seller', 'title_fa', 'Brand', 'Category1', 'Category2', 'sub_category'))
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f'Wrong column count near line {reader.line_num}')
            pid = row[key].strip()
            sig = fingerprint(row)
            if pid not in first:
                first[pid] = (sig, fingerprint([row[i] for i in attrs]), row[cost], row[shop])
                continue
            extra += 1
            repeated.add(pid)
            prev, facts, p, s = first[pid]
            if prev != sig:
                changed.add(pid)
                if facts != fingerprint([row[i] for i in attrs]): identity.add(pid)
                if p != row[cost]: price.add(pid)
                if s != row[shop]: seller.add(pid)
    return dict(extra_rows=extra, repeated_ids=len(repeated), identical_only_ids=len(repeated)-len(changed),
                any_difference_ids=len(changed), title_brand_or_category_diff_ids=len(identity),
                price_diff_ids=len(price), seller_diff_ids=len(seller))


if __name__ == '__main__':
    print(json.dumps(audit(Path('digikala/digikala-products.csv')), indent=2))
