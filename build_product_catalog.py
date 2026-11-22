"""Run from project3: python build_product_catalog.py"""
from pathlib import Path
from time import perf_counter
import duckdb

source = Path('digikala/digikala-products.csv')
out = Path('data/processed/product_catalog.parquet')
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix('.tmp.parquet')
fields = ['title_fa', 'Brand', 'Category1', 'Category2', 'sub_category', 'Price', 'Seller', 'Rate', 'Rate_cnt', 'Is_Fake', 'min_price_last_month']
identity = fields[:5]
start = perf_counter()
con = duckdb.connect()
con.execute("SET threads=2")
con.execute("SET memory_limit='3GB'")
# Column names below are fixed in code, not user input. Preserve raw string units.
clean = ', '.join(f'nullif(trim("{k}"), \'\') AS "{k}"' for k in fields)
# MIN=MAX means all non-null observed values agree; no arbitrary first row.
values = ', '.join(f'CASE WHEN min("{k}")=max("{k}") THEN min("{k}") END AS "{k}"' for k in fields)
flags = ', '.join(f'coalesce(min("{k}")<>max("{k}"), false) AS "{k}_conflict"' for k in fields)
ambiguous = ' OR '.join(f'"{k}_conflict"' for k in identity)
sql = f'''WITH clean AS (SELECT trim(id) AS id, {clean}
    FROM read_csv(?, header=true, all_varchar=true, delim=',', quote='"', escape='"')),
 grouped AS (SELECT id, count(*) AS source_rows, {values}, {flags} FROM clean GROUP BY id)
 SELECT *, ({ambiguous}) AS identity_conflict FROM grouped'''
con.sql(sql, params=[str(source)]).write_parquet(str(tmp), compression='zstd')
stats = con.execute('''SELECT count(*), count(DISTINCT id), count(*) FILTER (WHERE id IS NULL OR id=''), sum(source_rows), count(*) FILTER (WHERE identity_conflict) FROM read_parquet(?)''', [str(tmp)]).fetchone()
n, unique, empty, raw, ambiguous_count = stats
if not (n == unique == 948352 and empty == 0 and raw == 1283496):
    raise RuntimeError(f'Catalog validation failed: {stats}')
tmp.replace(out)
print(f'catalog_rows={n} | unique_ids={unique} | source_rows={raw} | identity_conflicts={ambiguous_count}')
print(f'{out} | {out.stat().st_size / 2**20:.1f} MiB | {perf_counter()-start:.1f} s')
