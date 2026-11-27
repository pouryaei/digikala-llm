from functools import wraps
from unittest.mock import patch
import json
from src.phase4_llm.comparison_service import ComparisonService


class FakeLLM:

    model = "fake"

    def generate(self, system, user):

        response = {
            "summary": "محصول A باتری بهتری دارد و محصول B دوربین بهتری دارد.",

            "products": [
                {
                    "product_id": 101,
                    "strengths": [
                        {
                            "point": "باتری عملکرد خوبی دارد.",
                            "comment_ids": [125]
                        }
                    ],
                    "weaknesses": [
                        {
                            "point": "دوربین متوسط است.",
                            "comment_ids": [126]
                        }
                    ]
                },
                {
                    "product_id": 202,
                    "strengths": [
                        {
                            "point": "دوربین کیفیت خوبی دارد.",
                            "comment_ids": [289]
                        }
                    ],
                    "weaknesses": [
                        {
                            "point": "قیمت بالاست.",
                            "comment_ids": [290]
                        }
                    ]
                }
            ],

            "criteria": [
                {
                    "criterion": "دوربین",
                    "differences": "محصول B بازخورد بهتری دارد.",
                    "comment_ids": [126, 289]
                }
            ],

            "conclusion": "انتخاب به نیاز کاربر بستگی دارد.",

            "insufficient_evidence": False,

            "inference": "اگر باتری مهم‌تر است A و اگر دوربین مهم‌تر است B."
        }

        return {
            "text": json.dumps(response, ensure_ascii=False),
            "latency": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }



def _without_cache(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with patch("src.phase4_llm.comparison_service.get_cached_response", return_value=None), patch("src.phase4_llm.comparison_service.save_response", return_value=None):
            return function(*args, **kwargs)
    return wrapped

@_without_cache
def test_comparison_valid_citations():

    products = [
        {"product_id": 101, "title": "گوشی A"},
        {"product_id": 202, "title": "گوشی B"}
    ]

    reviews = {
        101: [
            {"comment_id": 125, "text": "باتری خوب است."},
            {"comment_id": 126, "text": "دوربین متوسط است."}
        ],

        202: [
            {"comment_id": 289, "text": "دوربین عالی است."},
            {"comment_id": 290, "text": "قیمت بالاست."}
        ]
    }

    service = ComparisonService(llm=FakeLLM())

    result = service.compare(
        "این دو گوشی را مقایسه کن.",
        products,
        reviews
    )

    assert result["meta"]["citation_warning"] is False

    assert len(result["answer"]["products"]) == 2


@_without_cache
def test_wrong_product_citation():

    class WrongFakeLLM(FakeLLM):

        def generate(self, system, user):

            result = json.loads(
                super().generate(system, user)["text"]
            )

            result["products"][0]["strengths"][0]["comment_ids"] = [289]

            return {
                "text": json.dumps(result, ensure_ascii=False),
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    products = [
        {"product_id": 101, "title": "گوشی A"},
        {"product_id": 202, "title": "گوشی B"}
    ]

    reviews = {
        101: [
            {"comment_id": 125, "text": "باتری خوب است."},
            {"comment_id": 126, "text": "دوربین متوسط است."}
        ],

        202: [
            {"comment_id": 289, "text": "دوربین عالی است."},
            {"comment_id": 290, "text": "قیمت بالاست."}
        ]
    }

    service = ComparisonService(llm=WrongFakeLLM())

    result = service.compare(
        "این دو گوشی را مقایسه کن.",
        products,
        reviews
    )

    assert result["meta"]["citation_warning"] is True


@_without_cache
def test_contradictory_evidence():
    products = [
        {"product_id": 101, "title": "گوشی A"},
        {"product_id": 202, "title": "گوشی B"}
    ]

    reviews = {
        101: [
            {
                "comment_id": 125,
                "text": "باتری گوشی بسیار خوب است."
            },
            {
                "comment_id": 126,
                "text": "باتری گوشی خیلی ضعیف است."
            }
        ],
        202: [
            {
                "comment_id": 289,
                "text": "دوربین گوشی کیفیت خوبی دارد."
            }
        ]
    }

    class ContradictoryFakeLLM:
        model = "fake"

        def generate(self, system, user):
            import json

            response = {
                "summary": "نظرات کاربران درباره باتری گوشی A متفاوت است.",
                "products": [
                    {
                        "product_id": 101,
                        "strengths": [
                            {
                                "point": "برخی کاربران از باتری رضایت دارند.",
                                "comment_ids": [125]
                            }
                        ],
                        "weaknesses": [
                            {
                                "point": "برخی کاربران از باتری ناراضی هستند.",
                                "comment_ids": [126]
                            }
                        ]
                    },
                    {
                        "product_id": 202,
                        "strengths": [
                            {
                                "point": "کیفیت دوربین خوب گزارش شده است.",
                                "comment_ids": [289]
                            }
                        ],
                        "weaknesses": []
                    }
                ],
                "criteria": [],
                "conclusion": "درباره باتری گوشی A تجربه کاربران یکسان نیست.",
                "insufficient_evidence": False,
                "inference": ""
            }

            return {
                "text": json.dumps(response, ensure_ascii=False),
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    service = ComparisonService(llm=ContradictoryFakeLLM())
    result = service.compare("این دو محصول را مقایسه کن.", products, reviews)
    product_a = result["answer"]["products"][0]

    assert len(product_a["strengths"]) > 0
    assert len(product_a["weaknesses"]) > 0
    assert product_a["strengths"][0]["comment_ids"] == [125]
    assert product_a["weaknesses"][0]["comment_ids"] == [126]


@_without_cache
def test_insufficient_evidence():
    import json

    products = [
        {"product_id": 101, "title": "گوشی A"},
        {"product_id": 202, "title": "گوشی B"}
    ]

    reviews = {
        101: [
            {
                "comment_id": 125,
                "text": "رنگ گوشی زیباست."
            }
        ],
        202: [
            {
                "comment_id": 289,
                "text": "طراحی گوشی زیباست."
            }
        ]
    }

    class InsufficientFakeLLM:
        model = "fake"

        def generate(self, system, user):
            response = {
                "summary": "اطلاعات کافی درباره دوام باتری این دو گوشی وجود ندارد.",
                "products": [
                    {
                        "product_id": 101,
                        "strengths": [],
                        "weaknesses": []
                    },
                    {
                        "product_id": 202,
                        "strengths": [],
                        "weaknesses": []
                    }
                ],
                "criteria": [],
                "conclusion": "با نظرات موجود نمی‌توان دوام باتری را مقایسه کرد.",
                "insufficient_evidence": True,
                "inference": ""
            }

            return {
                "text": json.dumps(response, ensure_ascii=False),
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    service = ComparisonService(llm=InsufficientFakeLLM())
    result = service.compare("دوام باتری این دو گوشی را مقایسه کن.", products, reviews)

    assert result["answer"]["insufficient_evidence"] is True
    assert "باتری" in result["answer"]["conclusion"]


if __name__=="__main__":
    tests=[(name,func) for name,func in sorted(globals().items()) if name.startswith("test_") and callable(func)]
    for name,func in tests:
        func()
        print("PASS",name)
    print(f"{len(tests)} offline tests passed (no real API calls).")
