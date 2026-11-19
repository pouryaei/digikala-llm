import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

EMPTY_MARKERS = {"", "nan", "null", "none", "n/a"}
DISCOVERY_COUNT = 6
REVIEW_QA_COUNT = 10
ANALYTICS_COUNT = 4
CANDIDATES_PER_CATEGORY = 32

DISCOVERY_PATTERNS = (
    "محصول «{}» را پیدا کن.",
    "دنبال «{}» می‌گردم. کدام نتیجه با این نام مطابقت دارد؟",
    "آیا «{}» در کاتالوگ هست؟",
    "عنوان «{}» را در محصولات جست‌وجو کن.",
)
REVIEW_PATTERNS = (
    "یکی از تجربه‌های ثبت‌شده دربارهٔ «{}» را همراه شناسهٔ نظر نقل کن.",
    "از نظرهای «{}» یک نکته بیاور و شناسهٔ نظر را ذکر کن.",
    "یک بازخورد مستند دربارهٔ «{}» نشان بده.",
    "یک جمله از بدنهٔ یکی از نظرات «{}» با شناسهٔ نظر بیاور.",
)


def csv_rows(path: Path, required: set[str]):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        yield from reader


def clean(value: str | None) -> str:
    value = " ".join((value or "").split())
    return "" if value.casefold() in EMPTY_MARKERS else value


def candidates(products: Path) -> dict[str, dict[str, str]]:
    by_category: dict[str, list[str]] = {}
    selected: dict[str, dict[str, str]] = {}
    for row in csv_rows(products, {"id", "title_fa", "Category1"}):
        product_id = clean(row["id"])
        title = clean(row["title_fa"])
        category = clean(row["Category1"])
        if not title or not category or not product_id or product_id in selected:
            continue
        if category not in by_category:
            if len(by_category) >= REVIEW_QA_COUNT:
                continue
            by_category[category] = []
        ids = by_category[category]
        if len(ids) < CANDIDATES_PER_CATEGORY:
            ids.append(product_id)
            selected[product_id] = {"title": title, "category": category}
        if len(selected) == REVIEW_QA_COUNT * CANDIDATES_PER_CATEGORY:
            break
    if len(by_category) < REVIEW_QA_COUNT:
        raise ValueError("Need at least ten product categories with a title and ID.")
    return selected


def review_examples(comments: Path, selected: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    matched: dict[str, dict[str, str]] = {}
    seen_review_ids: set[str] = set()
    covered_categories: set[str] = set()
    for row in csv_rows(comments, {"product_id", "id", "body"}):
        product_id = clean(row["product_id"])
        if product_id not in selected or product_id in matched:
            continue
        review_id = clean(row["id"])
        body = clean(row["body"])
        if not body or not review_id or review_id in seen_review_ids:
            continue
        matched[product_id] = {"comment_id": review_id, "quote_prefix": " ".join(body.split()[:12])}
        seen_review_ids.add(review_id)
        covered_categories.add(selected[product_id]["category"])
        if len(covered_categories) == REVIEW_QA_COUNT:
            break
    if len(matched) < REVIEW_QA_COUNT:
        raise ValueError("Need example reviews for at least ten distinct candidate products.")
    return matched


def category_counts(products: Path, categories: set[str]) -> dict[str, int]:
    keys = defaultdict(set)
    for row in csv_rows(products, {"id", "Category1"}):
        category = clean(row["Category1"])
        product_id = clean(row["id"])
        if category in categories and product_id:
            keys[category].add(product_id)
    return {category: len(keys[category]) for category in categories}


def build(products: Path, comments: Path) -> list[dict]:
    selected = candidates(products)
    matched = review_examples(comments, selected)
    rows: list[dict] = []
    discovery: dict[str, str] = {}
    for product_id, item in selected.items():
        discovery.setdefault(item["category"], product_id)

    for index, (category, product_id) in enumerate(list(discovery.items())[:DISCOVERY_COUNT]):
        title = selected[product_id]["title"]
        rows.append({
            "qid": f"D{index + 1:02}",
            "task": "product_discovery",
            "query": DISCOVERY_PATTERNS[index % len(DISCOVERY_PATTERNS)].format(title),
            "target_product_id": product_id,
            "category": category,
            "judgment": "known_item_silver",
            "live_model_needed": False,
        })

    first_by_category: dict[str, str] = {}
    for product_id in matched:
        first_by_category.setdefault(selected[product_id]["category"], product_id)
    preferred = dict.fromkeys([*first_by_category.values(), *matched])
    for index, product_id in enumerate(list(preferred)[:REVIEW_QA_COUNT]):
        title = selected[product_id]["title"]
        rows.append({
            "qid": f"R{index + 1:02}",
            "task": "review_qa",
            "query": REVIEW_PATTERNS[index % len(REVIEW_PATTERNS)].format(title)
            + f" شناسهٔ محصول {product_id}.",
            "target_product_id": product_id,
            "reference_example": matched[product_id],
            "judgment": "verify cited review joins to product and supports answer; example is illustrative",
            "live_model_needed": False,
        })

    analytics_categories = list(discovery)[:ANALYTICS_COUNT]
    counts = category_counts(products, set(analytics_categories))
    for index, category in enumerate(analytics_categories):
        rows.append({
            "qid": f"A{index + 1:02}",
            "task": "analytics",
            "query": f"در فایل خام محصولات، تعداد شناسه‌های یکتای دستهٔ «{category}» چقدر است؟",
            "filter": {"Category1": category},
            "scope": "raw_products",
            "expected_unique_products": counts[category],
            "judgment": "count distinct trimmed nonmissing product IDs in the raw category",
            "live_model_needed": False,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new path.")
    csv.field_size_limit(16 * 1024 * 1024)
    rows = build(args.data_dir / "digikala-products.csv", args.data_dir / "digikala-comments.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"PASS: {len(rows)} draft cases written to {args.output}")
    print("Discovery: 6, Review QA: 10, Analytics: 4. QBC13 gateway requests: 0.")


if __name__ == "__main__":
    main()
