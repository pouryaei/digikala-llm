"""Offline retrieval evaluation helpers."""
def calculate_metrics(results_dict, ground_truth_dict, k=5):
    """Legacy API: return (HitRate@k, MRR@k)."""
    if not ground_truth_dict: return 0.0,0.0
    hits=0; reciprocal=[]
    for q_id,target in ground_truth_dict.items():
        retrieved=list(results_dict.get(q_id,[]))[:k]
        if target in retrieved:
            hits+=1; reciprocal.append(1.0/(retrieved.index(target)+1))
        else: reciprocal.append(0.0)
    n=len(ground_truth_dict)
    return hits/n,sum(reciprocal)/n

def calculate_extended_metrics(results_dict, ground_truth_dict):
    """Return Hit@5, Hit@10 and MRR over full rankings (missing queries count as misses)."""
    if not ground_truth_dict: return {"Hit@5":0.0,"Hit@10":0.0,"MRR":0.0}
    hit5=hit10=0; rr=0.0
    for q,target in ground_truth_dict.items():
        results=list(results_dict.get(q,[]))
        if target in results:
            rank=results.index(target)+1
            hit5 += rank<=5; hit10 += rank<=10; rr += 1.0/rank
    n=len(ground_truth_dict)
    return {"Hit@5":hit5/n,"Hit@10":hit10/n,"MRR":rr/n}
