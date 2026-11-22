"""Run from project3: python audit_day1.py (standard library only)."""
import csv
import json
from collections import Counter
from pathlib import Path
from time import perf_counter


def records(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, strict=True)
        header = next(reader)
        ix = [header.index(k) for k in required]
        try:
            for row in reader:
                if not row:  # skip physically empty records, not empty fields
                    continue
                if len(row) != len(header):
                    raise ValueError(f'{path.name}: wrong field count near line {reader.line_num}')
                yield tuple(row[i].strip() for i in ix)
        except csv.Error as err:
            raise ValueError(f'{path.name}: CSV error near line {reader.line_num}: {err}') from err


def main():
    base = Path('digikala')
    start = perf_counter()
    ids = set()
    products = missing = 0
    for (pid,) in records(base / 'digikala-products.csv', ['id']):
        products += 1
        if pid:
            ids.add(pid)
        else:
            missing += 1
    print(f'Products scanned: {products:,}. Reading comments...', flush=True)
    comments = matched = no_pid = no_body = 0
    labels = Counter()
    for pid, label, body in records(base / 'digikala-comments.csv', ['product_id', 'recommendation_status', 'body']):
        comments += 1
        matched += pid in ids
        no_pid += not pid
        no_body += not body
        labels[label] += 1
    print(json.dumps(dict(products=products, unique_nonblank_product_ids=len(ids), missing_product_ids=missing,
        extra_rows_with_duplicate_product_id=products-missing-len(ids), comments=comments,
        comments_matched_to_product=matched, comments_without_product_id=no_pid,
        comments_with_unknown_product_id=comments-matched-no_pid,
        join_coverage_pct=round(100*matched/comments, 2) if comments else None,
        blank_bodies=no_body, status_counts=dict(labels), elapsed_s=round(perf_counter()-start,1)), indent=2))


if __name__ == '__main__':
    main()
