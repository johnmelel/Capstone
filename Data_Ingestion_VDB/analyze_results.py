import json

with open('rag_comparison.json') as f:
    results = json.load(f)

print("\n" + "="*70)
print("📊 RAG COMPARISON ANALYSIS")
print("="*70)

raw_scores = []
md_scores = []

for r in results:
    print(f"\n❓ {r['question']}")
    print(f"\n   🔵 RAW (score: {r['raw']['top_score']:.4f}):")
    print(f"      {r['raw']['answer'][:150]}...")
    print(f"\n   🟢 MD (score: {r['markdown']['top_score']:.4f}):")
    print(f"      {r['markdown']['answer'][:150]}...")
    
    raw_scores.append(r['raw']['top_score'])
    md_scores.append(r['markdown']['top_score'])
    
    # Which is better?
    if r['markdown']['top_score'] > r['raw']['top_score'] + 0.05:
        print(f"   ✅ Markdown wins (+{r['markdown']['top_score'] - r['raw']['top_score']:.3f})")
    elif r['raw']['top_score'] > r['markdown']['top_score'] + 0.05:
        print(f"   ✅ Raw wins (+{r['raw']['top_score'] - r['markdown']['top_score']:.3f})")
    else:
        print(f"   ⚖️  Tie (difference: {abs(r['markdown']['top_score'] - r['raw']['top_score']):.3f})")

print(f"\n{'='*70}")
print(f"SUMMARY:")
print(f"{'='*70}")
print(f"Average RAW score: {sum(raw_scores)/len(raw_scores):.4f}")
print(f"Average MD score:  {sum(md_scores)/len(md_scores):.4f}")
print(f"Difference:        {(sum(md_scores)-sum(raw_scores))/len(raw_scores):.4f}")

if sum(md_scores) > sum(raw_scores):
    print(f"\n🏆 MARKDOWN WINS!")
else:
    print(f"\n🏆 RAW TEXT WINS!")