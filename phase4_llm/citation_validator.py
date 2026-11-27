"""Validate review identifiers and verbatim source quotes for Task 2."""
from .response_schema import Point, ProductAnswer


def _source_map(retrieved):
    ids, texts = set(), {}
    for item in retrieved:
        if isinstance(item, dict):
            try: cid = int(item["comment_id"])
            except (KeyError, ValueError, TypeError): continue
            ids.add(cid)
            if isinstance(item.get("text"), str): texts[cid] = item["text"]
        else:
            try: ids.add(int(item))
            except (ValueError, TypeError): pass
    return ids, texts


def validate_citations(answer, retrieved_ids, require_quotes=False):
    ids, texts = _source_map(retrieved_ids)
    invalid_ids, uncited, invalid_quotes = [], [], []
    for p in answer.positive_points + answer.negative_points + answer.observations:
        if not p.point.strip() or not p.comment_ids: uncited.append(p.point)
        invalid_ids.extend(c for c in p.comment_ids if c not in ids)
        quote = p.quote.strip()
        if require_quotes and (not quote or len(p.comment_ids) != 1):
            invalid_quotes.append(p.point)
        elif quote and texts and not any(quote in texts.get(c, "") for c in p.comment_ids):
            invalid_quotes.append(p.point)
    return {"is_valid": not (invalid_ids or uncited or invalid_quotes),
            "invalid_ids": invalid_ids, "uncited_points": uncited, "invalid_quotes": invalid_quotes}


def clean_answer(answer, retrieved_ids):
    ids, texts = _source_map(retrieved_ids)
    def clean(points):
        out=[]
        for p in points:
            valid=list(dict.fromkeys(c for c in p.comment_ids if c in ids))
            if not p.point.strip() or not valid: continue
            quote=p.quote.strip()
            if quote and texts and not any(quote in texts.get(c, "") for c in valid): continue
            out.append(Point(point=p.point, comment_ids=valid, quote=quote))
        return out
    return ProductAnswer(summary=answer.summary, positive_points=clean(answer.positive_points),
        negative_points=clean(answer.negative_points), observations=clean(answer.observations),
        insufficient_evidence=answer.insufficient_evidence, inference=answer.inference)

def validate_comparison_citations(answer, reviews_by_product):

    comment_to_product = {}

    for product_id, reviews in reviews_by_product.items():

        for review in reviews:

            comment_id = int(review["comment_id"])
            comment_to_product[comment_id] = str(product_id)

    invalid_ids = []
    wrong_product_ids = []
    uncited_points = []

    for product in answer.products:

        product_id = str(product.product_id)

        for point in product.strengths + product.weaknesses:

            if not point.comment_ids or not point.point.strip():
                uncited_points.append(point.point)

            for comment_id in point.comment_ids:

                if comment_id not in comment_to_product:
                    invalid_ids.append(comment_id)

                elif comment_to_product[comment_id] != product_id:
                    wrong_product_ids.append(comment_id)

    for criterion in answer.criteria:

        for comment_id in criterion.comment_ids:

            if comment_id not in comment_to_product:
                invalid_ids.append(comment_id)

    return {
        "is_valid": (
            not invalid_ids
            and not wrong_product_ids
            and not uncited_points
        ),
        "invalid_ids": invalid_ids,
        "wrong_product_ids": wrong_product_ids,
        "uncited_points": uncited_points
    }


def clean_comparison_answer(answer, reviews_by_product):

    comment_to_product = {}

    for product_id, reviews in reviews_by_product.items():

        for review in reviews:

            comment_id = int(review["comment_id"])
            comment_to_product[comment_id] = str(product_id)

    cleaned_products = []

    for product in answer.products:

        product_id = str(product.product_id)

        def clean_points(points):

            result = []

            for point in points:

                valid_ids = [
                    comment_id
                    for comment_id in point.comment_ids
                    if (
                        comment_id in comment_to_product
                        and comment_to_product[comment_id] == product_id
                    )
                ]

                if valid_ids and len(valid_ids) == len(point.comment_ids) and point.point.strip():

                    result.append({
                        "point": point.point,
                        "comment_ids": valid_ids
                    })

            return result

        cleaned_products.append({
            "product_id": product.product_id,
            "strengths": clean_points(product.strengths),
            "weaknesses": clean_points(product.weaknesses)
        })

    valid_all_ids = set(comment_to_product)
    cleaned_criteria = []

    for criterion in answer.criteria:

        valid_ids = [
            comment_id
            for comment_id in criterion.comment_ids
            if comment_id in valid_all_ids
        ]

        if valid_ids and len(valid_ids) == len(criterion.comment_ids):
            cleaned_criteria.append({
                "criterion": criterion.criterion,
                "differences": criterion.differences,
                "comment_ids": valid_ids
            })

    return {
        "summary": answer.summary,
        "products": cleaned_products,
        "criteria": cleaned_criteria,
        "conclusion": answer.conclusion,
        "insufficient_evidence": answer.insufficient_evidence,
        "inference": answer.inference
    }