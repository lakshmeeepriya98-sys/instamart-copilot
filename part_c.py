import re
import json
import random
import time
import pandas as pd
import mock_llm

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
segs = pd.read_csv("outputs/customer_segments.csv")
prem_map = segs.set_index("customer_id")["is_premium"].to_dict()
seg_map = segs.set_index("customer_id")["segment_name"].to_dict()

def redact(txt):
    txt = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', txt)
    txt = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', txt)
    txt = re.sub(r'ORD-\d{6}', '[ORDER_ID]', txt)
    return txt

FEW_SHOT = """CLASSIFY support tickets into billing, delivery, technical, or fraud.
Return strictly JSON: {"category": "...", "reason": "..."}
EXAMPLES:
User: Someone hacked my account and made fraud charges.
Assistant: {"category": "fraud", "reason": "account hack reported"}
User: Where is my order? The rider is late.
Assistant: {"category": "delivery", "reason": "late delivery"}
"""

def call_mock(text, budget, used_tokens):
    for n in range(3):
        try:
            resp = mock_llm.chat([
                {"role": "system", "content": FEW_SHOT},
                {"role": "user", "content": text}
            ])
            return resp, used_tokens + resp.usage.total_tokens
        except mock_llm.TransientLLMError:
            time.sleep(0.05 * (2**n) * random.uniform(0.9, 1.15))
    return None, used_tokens

mock_llm.reset()
results = []
used_tokens = 0
TOKEN_BUDGET = 100000

for _, r in tickets.iterrows():
    tid, cid, txt = r["ticket_id"], r["customer_id"], r["text"]
    if not txt.strip():
        results.append((tid, cid, "skipped_empty", "", 0))
        continue
    if mock_llm.est_tokens(txt) > 400:
        results.append((tid, cid, "flagged_long", "", 0))
        continue
    if used_tokens >= TOKEN_BUDGET:
        results.append((tid, cid, "budget_exhausted", "", 0))
        continue
    
    red_txt = redact(txt)
    resp, used_tokens = call_mock(red_txt, TOKEN_BUDGET, used_tokens)
    
    cat = ""
    status = "flagged_invalid_output"
    if resp:
        try:
            d = json.loads(resp.choices[0].message.content)
            if d.get("category") in ["billing", "delivery", "technical", "fraud"]:
                cat = d["category"]
                status = "classified_json"
        except:
            words = set(re.findall(r"[a-z]+", resp.choices[0].message.content.lower()))
            matches = words & {"billing", "delivery", "technical", "fraud"}
            if len(matches) == 1:
                cat = list(matches)[0]
                status = "classified_keyword"
                
    results.append((tid, cid, status, cat, resp.usage.total_tokens if resp else 0))

out_rows = []
for tid, cid, st, cat, tok in results:
    is_prem = prem_map.get(cid, False)
    sname = seg_map.get(cid, "")
    
    if st in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
        rt, prio = "human_review", "NORMAL"
    elif cat == "fraud":
        rt, prio = "security_team", "HIGH"
    elif is_prem:
        rt, prio = "premium_desk", "HIGH"
    else:
        rt, prio = "support_queue", "NORMAL"
        
    out_rows.append({
        "ticket_id": tid, "customer_id": cid, "status": st, "category": cat,
        "segment_name": sname if cat else "", "priority": prio, "route": rt, "tokens_used": tok
    })

pd.DataFrame(out_rows).to_csv("outputs/triage_results.csv", index=False)
pd.DataFrame(out_rows).head(50).to_csv("outputs/triage_budget_demo.csv", index=False)
print("Part C Done.")
