"""Actual offline head adaptation of the existing character TF-IDF model.

The original vectorizer / logistic weights / recorded test are preserved.
Only a new supervised head is trained; this is NOT transformer LoRA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import unicodedata
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import zstandard

import ml_service
from scripts.build_review_store import _HashedStream

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "bonus_artifacts"
REPORT = ROOT / "reports" / "bonus" / "adaptation.json"
LABELS = ("no_idea", "not_recommended", "recommended")
SEED = 42


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def product_bucket(pid):
    # Exact DuckDB md5_number_lower convention confirmed on the original
    # recorded test products; replicate the EDA's fixed product-group split.
    return int.from_bytes(hashlib.md5(f"42:product:{pid}".encode()).digest()[8:], "little") % 10000


def comparable(text):
    return "".join(c for c in text.lower() if unicodedata.category(c)[0] in ("L","N") or unicodedata.category(c) == "So")


def text_key(text):
    return hashlib.md5(comparable(text).encode()).hexdigest()


def sample(source_archive):
    recorded = list(csv.DictReader((ROOT/"m2_outputs/test_predictions.csv").open(encoding="utf-8-sig", newline="")))
    wanted = {r["comment_id"]: r for r in recorded}
    pools = {"train": [], "validation": [], "test": []}
    seen_test = set()
    # Deterministic subsampling from train products; oversample minority classes
    # only within train. Validation sampling is label-independent.
    gates = {"recommended": 8, "not_recommended": 40, "no_idea": 40}
    with ZipFile(source_archive) as z:
        names = [n for n in z.namelist() if n.endswith("data/analytics/reviews_payload.json")]
        if len(names) != 1:
            raise ValueError("Expected full cleaned source archive")
        manifest = json.loads(z.read(names[0]))
        with z.open(names[0].replace("reviews_payload.json","reviews_payload.csv.zst")) as raw:
            hashed = _HashedStream(raw)
            with zstandard.ZstdDecompressor().stream_reader(hashed, read_size=1024*1024) as stream:
                with io.TextIOWrapper(stream, encoding="utf8", newline="") as f:
                    reader=csv.reader(f); header=next(reader)
                    c,p,y,t,b=[header.index(x) for x in ("comment_id","product_id","recommendation_status","title_raw","body_raw")]
                    count=0
                    for row in reader:
                        count+=1
                        cid,pid,label = row[c], row[p], row[y]
                        if label not in LABELS:
                            continue
                        if cid in wanted:
                            bucket=product_bucket(pid)
                            if bucket<9000 or label!=wanted[cid]["recommendation_status"] or cid in seen_test:
                                raise ValueError("Original test provenance failed")
                            split="test";seen_test.add(cid)
                        else:
                            u=int.from_bytes(hashlib.md5(("bonus:adapt:"+cid).encode()).digest()[:8],"little")%1000
                            if u>=40:continue
                            bucket=product_bucket(pid)
                            if bucket<8000 and u<gates[label]:split="train"
                            elif 8000<=bucket<9000 and u<35:split="validation"
                            else:continue
                        text=ml_service.compose_model_text(row[b],row[t])
                        if not comparable(text):
                            if split=="test":raise ValueError("Empty original test feature")
                            continue
                        pools[split].append((cid,pid,text,label,text_key(text)))
                        if count%1000000==0:
                            print("source rows",count,flush=True)
            source_hash=hashed.digest.hexdigest()
    if source_hash != manifest["sha256"] or count != manifest["row_count"] or seen_test != set(wanted):
        raise ValueError("Full source checksum/count/test coverage failed")

    # Deduplicate each sampled split and avoid matching training/validation
    # feature keys against the original held-out test. Test labels never enter
    # training or validation decisions.
    excluded = {r[4] for r in pools["test"]}
    if len(excluded)!=len(pools["test"]):
        raise ValueError("Original test comparable texts not unique")
    removed={}
    for split in ("validation","train"):
        counts=Counter(row[4] for row in pools[split])
        filtered=[row for row in pools[split] if counts[row[4]]==1 and row[4] not in excluded]
        removed[split]=len(pools[split])-len(filtered)
        pools[split]=filtered
        excluded.update(row[4] for row in filtered)
    for split in pools:
        pools[split].sort(key=lambda r: int(r[0]))
    sets={s:{r[1] for r in rows} for s,rows in pools.items()}
    if any(sets[a]&sets[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Product overlap")
    if len({r[0] for rows in pools.values() for r in rows})!=sum(map(len,pools.values())):
        raise ValueError("Review ID overlap")
    audit={"source_payload_sha256":source_hash,"source_rows":count,"seed":SEED,
           "split_policy":"original product MD5 80/10/10; deterministic train/validation sample; unchanged recorded test IDs",
           "shared_products":0,"shared_comparable_text_keys":0,"removed_sample_duplicates":removed,
           "splits":{s:{"rows":len(rows),"products":len(sets[s]),"classes":dict(Counter(r[3] for r in rows))}
                     for s,rows in pools.items()}}
    if any(set(v["classes"])!=set(LABELS) for v in audit["splits"].values()):
        raise ValueError("Missing class")
    return pools, audit, wanted


def score(y,logits):
    pred=np.argmax(logits,axis=1)
    return {"macro_f1":float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(y,pred))}


def bias_calibrate(logits,y):
    # Fixed grid chosen before looking at test values. Select on validation only.
    best=None
    for a in np.linspace(-1.25,0.75,11):
        for b in np.linspace(-0.65,0.85,9):
            bias=np.array([a,b,0.0],dtype=np.float32)
            metric=score(y,logits+bias)["macro_f1"]
            key=(metric,-float(np.linalg.norm(bias)))
            if best is None or key>best[0]:best=(key,bias)
    return best[1]


def adapt(pools,audit,wanted,*,epochs=4):
    start=perf_counter()
    model=ml_service.load_legacy_model()
    if tuple(model.classes_)!=LABELS:
        raise ValueError("Unexpected model class order")
    vectorizer=model["tfidf"]; classifier=model["model"]
    W0=classifier.coef_.astype(np.float32,copy=True);b0=classifier.intercept_.astype(np.float32,copy=True)
    matrices={s:vectorizer.transform([r[2] for r in rows]).astype(np.float32) for s,rows in pools.items()}
    labels={s:np.array([LABELS.index(r[3]) for r in rows]) for s,rows in pools.items()}
    X,Y=matrices["train"],labels["train"]; V,Yv=matrices["validation"],labels["validation"]
    val0=V@W0.T+b0
    trials=[{"name":"saved_head","validation":score(Yv,val0)}]
    shift0=bias_calibrate(val0,Yv)
    trials.append({"name":"bias_only","validation":score(Yv,val0+shift0),"bias":shift0.tolist()})
    best_score=-1.0;best=None
    rng=np.random.default_rng(SEED)
    # Two bounded optimizer settings via a fixed validation-only comparison.
    for rate in (0.4,1.2):
        W,b=W0.copy(),b0.copy()
        for epoch in range(1,epochs+1):
            for ix in np.array_split(rng.permutation(len(Y)),max(1,int(np.ceil(len(Y)/512)))):
                batch=X[ix]; y=Y[ix]
                logits=batch@W.T+b; logits-=logits.max(axis=1,keepdims=True)
                probs=np.exp(logits);probs/=probs.sum(axis=1,keepdims=True)
                probs[np.arange(len(ix)),y]-=1
                grad=np.asarray(batch.T@probs).T/len(ix)
                # L2 tether to the original head; no fit of the frozen vectorizer.
                W-=rate*(grad+0.0005*(W-W0))
                b-=rate*(probs.mean(axis=0)+0.0005*(b-b0))
            val=V@W.T+b
            bias=bias_calibrate(val,Yv)
            calibrated=score(Yv,val+bias)
            trial={"name":"head_adapted","rate":rate,"epoch":epoch,"validation":calibrated,
                   "validation_without_bias":score(Yv,val),"bias":bias.tolist()}
            trials.append(trial)
            if calibrated["macro_f1"]>best_score:
                best_score=calibrated["macro_f1"];best=(W.copy(),b.copy()+bias,trial)
            print(json.dumps(trial,ensure_ascii=False),flush=True)
    W,b,chosen=best
    # Held-out score is calculated only AFTER selection of the adapted head.
    T,Yt=matrices["test"],labels["test"]
    original=model.predict([r[2] for r in pools["test"]]).tolist()
    if any(pred != wanted[r[0]]["predicted_status"] for pred,r in zip(original,pools["test"])):
        raise ValueError("Original model no longer reproduces the recorded test")
    base_test=score(Yt,T@W0.T+b0)
    new_logits=T@W.T+b
    new_test=score(Yt,new_logits)
    improved=new_test["macro_f1"]>base_test["macro_f1"] and chosen["validation"]["macro_f1"]>trials[0]["validation"]["macro_f1"]
    ART.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(ART/"adapted_head.npz",coef=W,intercept=b,labels=np.array(LABELS))
    report={"method":"Supervised fine-tuning of the saved logistic classifier head; frozen character TF-IDF vectorizer. Not LLM/LoRA.",
            "audit":audit,"original_model_sha256":sha(ROOT/"m2_outputs/best_model.joblib"),
            "adapted_head_sha256":sha(ART/"adapted_head.npz"),"validation_trials":trials,
            "selected_on_validation_only":chosen,"original_test_reproduced_all":len(original),
            "test":{"baseline":base_test,"adapted":new_test,"delta_macro_f1":new_test["macro_f1"]-base_test["macro_f1"],
                    "adapted_classification_report":classification_report(Yt,np.argmax(new_logits,axis=1),labels=[0,1,2],target_names=list(LABELS),output_dict=True,zero_division=0)},
            "improved_on_validation_and_test":improved,"production_model_replaced":False,
            "limitations":"Original holdout is a previously recorded test set, not a new blind benchmark; no test results used to select the head. Classical head adaptation, not transformer fine-tuning. Sampled validation is from original validation product buckets, not the exact EDA validation rows. Overlap is checked across the new sampled splits and recorded test; original full training text is unavailable for a historical-validation-overlap audit.",
            "elapsed_seconds":round(perf_counter()-start,3)}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return report


def predict_adapted(text, title=None):
    """Use exactly the preserved head, independent of the active model."""
    from recommendation_ml.model_registry import load_candidate
    clean,reason=ml_service._prepare(text,title)
    if reason:
        return {"valid":False,"reason":reason,"label":None,"probabilities":{}}
    model=load_candidate("lr_adapted")
    proba=model.predict_proba([clean])[0]
    index=int(proba.argmax())
    return {"valid":True,"label":str(model.classes_[index]),
            "confidence":float(proba[index]),"model_version":model.model_version,
            "probabilities":{str(label):float(value) for label,value in zip(model.classes_,proba)}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=4,choices=(1,2,3,4))
    args=p.parse_args()
    pools,audit,wanted=sample(args.source_archive)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    r=adapt(pools,audit,wanted,epochs=args.epochs)
    print(json.dumps({"test":r["test"],"improved":r["improved_on_validation_and_test"]},ensure_ascii=False,indent=2))
