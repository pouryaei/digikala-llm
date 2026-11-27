"""Compare independently filled human 0/1/2 scores with frozen assistant labels."""
from pathlib import Path
import argparse,csv,json
from sklearn.metrics import cohen_kappa_score

def compare(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8-sig",newline="")))
    result={}
    for dimension in ("quality","grounding"):
        pairs=[]
        for r in rows:
            text=r.get("human_"+dimension+"_0_2","").strip()
            if not text:continue
            if text not in ("0","1","2"):raise ValueError("Only blank, 0, 1 or 2 are valid human scores")
            pairs.append((int(r["assistant_"+dimension+"_0_2"]),int(text)))
        n=len(pairs)
        kappa=float(cohen_kappa_score(*zip(*pairs),labels=[0,1,2],weights="quadratic")) if n>1 else None
        if kappa is not None and not __import__("math").isfinite(kappa):kappa=None
        result[dimension]={"reviewed":n,"exact_agreement":sum(a==b for a,b in pairs)/n if n else None,"quadratic_weighted_cohen_kappa":kappa}
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filled_csv",type=Path)
    args=parser.parse_args()
    print(json.dumps(compare(args.filled_csv),indent=2))
