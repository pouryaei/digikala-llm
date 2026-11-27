for p in [r"D:/goal/programming/quera/boot_camp/project3/digikala/digikala-products.csv",
          r"D:/goal/programming/quera/boot_camp/project3/digikala/digikala-comments.csv"]:
    with open(p, "rb") as f:
        n = sum(1 for _ in f)
    print(f"{p.split('/')[-1]}: {n:,} lines (incl. header)")
