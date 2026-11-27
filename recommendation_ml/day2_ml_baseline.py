"""Day 2: reproducible CPU-only validation baseline; run from project3."""
import csv
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

LABELS = ['not_recommended', 'no_idea', 'recommended']
CAP, SEED = 60000, 42

def clean(value):
    value = unicodedata.normalize('NFKC', value or '').translate(str.maketrans({'ي':'ی','ك':'ک','\u200c':' '}))
    value = re.sub(r'\s+', ' ', value).strip()
    return '' if value.lower() == 'nan' else value

def main():
    start = perf_counter()
    rng = random.Random(SEED)
    pool = []
    eligible = 0
    with Path('digikala/digikala-comments.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, strict=True):
            y = row['recommendation_status'].strip()
            if y not in LABELS: continue
            eligible += 1
            j = eligible-1 if eligible <= CAP else rng.randrange(eligible)
            if j >= CAP: continue
            item = (row['id'].strip(), row['product_id'].strip(), row['title'], row['body'], y)
            if eligible <= CAP: pool.append(item)
            else: pool[j] = item
    # Conservative pilot policy: drop all repeated IDs and identical normalized inputs
    # within this sample, including label conflicts. Report the resulting coverage.
    rows = [(i,p,' '.join(filter(None,(clean(t),clean(b)))),y) for i,p,t,b,y in pool]
    ids = Counter(r[0] for r in rows); texts = Counter(r[2] for r in rows)
    kept = [r for r in rows if all(r[:3]) and ids[r[0]] == texts[r[2]] == 1]
    def split(ix, fraction):
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=SEED).split(ix,groups=[kept[k][1] for k in ix]))
        return [ix[k] for k in a], [ix[k] for k in b]
    dev,test = split(list(range(len(kept))),0.15)
    train,val = split(dev,0.15/0.85)
    print(f'eligible={eligible} sampled={len(pool)} kept_unique={len(kept)} train={len(train)} validation={len(val)} reserved_test={len(test)}')
    X = [kept[k][2] for k in train]; y = [kept[k][3] for k in train]
    V = [kept[k][2] for k in val]; truth = [kept[k][3] for k in val]
    print('train_counts=',dict(Counter(y)),'validation_counts=',dict(Counter(truth)))
    if set(y)!=set(LABELS) or set(truth)!=set(LABELS): raise ValueError('Missing class in train or validation')
    for name,model in [('majority',DummyClassifier(strategy='most_frequent')),('tfidf_lr',make_pipeline(TfidfVectorizer(min_df=2,max_features=40000),LogisticRegression(max_iter=400,class_weight='balanced',random_state=SEED)))]:
        tick = perf_counter(); model.fit(X,y); pred = model.predict(V)
        print(f'{name} validation_macro_f1={f1_score(truth,pred,labels=LABELS,average="macro",zero_division=0):.4f} fit_predict_s={perf_counter()-tick:.1f}')
        if name == 'tfidf_lr':
                print(classification_report(
                truth, pred, labels=LABELS, digits=3, zero_division=0))
                print('Rows = actual, columns = predicted; order:', LABELS)
                print(confusion_matrix(truth, pred, labels=LABELS))

    print(f'total_s={perf_counter()-start:.1f}')

if __name__=='__main__': main()
