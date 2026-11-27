from classifier.model import ROOT, classify, provenance
from time import monotonic
import json
print(provenance(),flush=True)
for name in ["01-shirt.png","portrait.png","01-shirt.png"]:
    t=monotonic()
    result=classify((ROOT/"demo"/name).read_bytes())
    print(name,round(monotonic()-t,2),json.dumps(result,ensure_ascii=False),flush=True)
