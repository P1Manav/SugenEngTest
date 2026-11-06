import json

with open("public/data/dev_notes.jsonl", encoding="utf-8") as f:
    golds = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

with open("out/dev_predictions.jsonl", encoding="utf-8") as f:
    preds = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

print(f"{'ID':<10} {'STATUS':<8} {'GOLD':<15} {'PRED':<25} {'NOTE'}")
print("-" * 80)
for nid, g in sorted(golds.items()):
    p = preds.get(nid, {})
    codes = p.get("codes", [])
    gset = set(g["gold"])
    t1 = codes and codes[0] in gset
    t3 = any(c in gset for c in codes[:3])
    if not t1:
        status = "T3_HIT" if t3 else "MISS"
        gold_str = str(g["gold"])
        pred_str = str(codes[:3])
        note_snippet = g["text"].replace("\n", " ")[:45]
        print(f"{nid:<10} {status:<8} {gold_str:<15} {pred_str:<25} {note_snippet}")
