"""Run each OCR engine over the split single-char dataset and score vs truth."""
import json, os, sys, logging
logging.getLogger('src').setLevel(logging.ERROR)
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from src.ocr import Crop, get_engine

D="/private/tmp/claude-501/-Users-wizeng-Documents-repos-misc-zeng-dynasty/97e38612-5008-4569-9c81-0d59346d0a6c/scratchpad/split_names"
truth=json.load(open(f"{D}/truth.json"))["labels"]
ids=sorted(truth.keys(), key=lambda k:(int(k.split('.')[0]), int(k.split('.')[1]) if '.' in k else 0))
crops=[Crop(id=i, image=Image.open(f"{D}/{i}.png").convert("L")) for i in range(0)]  # placeholder

# Crop.id is int in the module; our ids have dots. Use a parallel id->str map.
class SCrop:
    def __init__(self, sid, image): self.id=sid; self.image=image
crops=[SCrop(i, Image.open(f"{D}/{i}.png").convert("L")) for i in ids]

def score(preds):
    hits=sum(1 for i in ids if preds.get(i)==truth[i])
    skip=sum(1 for i in ids if not preds.get(i))
    return hits, len(ids), skip

engine_name=sys.argv[1]
eng=get_engine(engine_name)
preds=eng.recognize(crops)
preds={str(k):v for k,v in preds.items()}
h,t,s=score(preds)
misses=[(i,truth[i],preds.get(i,'')) for i in ids if preds.get(i)!=truth[i]]
out=f"{D}/preds_{engine_name.replace(':','_')}.json"
json.dump({"engine":eng.name,"labels":preds}, open(out,"w"), ensure_ascii=False, indent=2)
print(f"\n=== {eng.name} ===")
print(f"  {h}/{t} = {100*h/t:.1f}%  (skipped {s})")
print(f"  misses ({len(misses)}):")
for i,tr,pr in misses[:40]:
    print(f"    {i}: {tr} -> {pr!r}")
