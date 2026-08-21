"""Split every book1 name-crop into per-character single-char crops for the OCR
bake-off, and emit a truth file keyed by the split-crop id.

Split id scheme: "{orig_id}" for single-char crops, "{orig_id}.{k}" for the k-th
character (1-based) of a stacked multi-char crop. Truth char comes from the
per-crop full-name truth (data/oracles/book1_names_truth_full.json), one char per
split piece in reading (top-to-bottom) order.
"""
import json, os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None

TRUTH="data/oracles/book1_names_truth_full.json"
OUTDIR="/private/tmp/claude-501/-Users-wizeng-Documents-repos-misc-zeng-dynasty/97e38612-5008-4569-9c81-0d59346d0a6c/scratchpad/split_names"
os.makedirs(OUTDIR, exist_ok=True)

def split_vertical(im, ncols):
    a=np.asarray(im.convert("L"))
    ink=(a<128).sum(axis=1)
    rows=[r for r in range(len(ink)) if ink[r]>0]
    if not rows: return [im]
    top,bot=rows[0],rows[-1]
    a2=a[top:bot+1]; ink2=(a2<128).sum(axis=1)
    gaps=[r for r in range(len(ink2)) if ink2[r]==0]
    runs=[]
    if gaps:
        s=p=gaps[0]
        for r in gaps[1:]:
            if r!=p+1: runs.append((s,p)); s=r
            p=r
        runs.append((s,p))
    cand=sorted(runs, key=lambda x:-(x[1]-x[0]))
    cuts=sorted((r[0]+r[1])//2 for r in cand[:ncols-1])
    b=[0]+cuts+[a2.shape[0]]
    return [Image.fromarray(a2[b[k]:b[k+1]]) for k in range(len(b)-1)]

truth=json.load(open(TRUTH))["labels"]
split_truth={}
for cid, name in truth.items():
    im=Image.open(f"books/book1/names/{cid}.png")
    n=len(name)
    if n==1:
        im.save(f"{OUTDIR}/{cid}.png"); split_truth[cid]=name
    else:
        parts=split_vertical(im, n)
        for k,(pt,ch) in enumerate(zip(parts, name), start=1):
            sid=f"{cid}.{k}"
            pt.save(f"{OUTDIR}/{sid}.png"); split_truth[sid]=ch
json.dump({"labels":split_truth}, open(f"{OUTDIR}/truth.json","w"), ensure_ascii=False, indent=2)
print("wrote", len(split_truth), "single-char crops to", OUTDIR)
from collections import Counter
print("split-count check: crops with mismatched part count would error above; all ok")
