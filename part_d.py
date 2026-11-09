import os
import json
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

chunks = []
for fn in os.listdir("data/policies"):
    if fn.endswith(".txt"):
        with open(os.path.join("data/policies", fn), "r") as f:
            words = f.read().split()
            idx = 0
            while idx < len(words):
                ch = " ".join(words[idx : idx + 40])
                chunks.append({"source": fn, "text": ch})
                idx += 30

vec = TfidfVectorizer()
X_ch = vec.fit_transform([c["text"] for c in chunks])

def retrieve(query, k=3):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, X_ch)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "text": chunks[i]["text"]} for i in top_idx]

def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    df = pd.read_sql_query("SELECT * FROM orders WHERE order_id = ?", conn, params=(order_id,))
    conn.close()
    if df.empty:
        return {"error": "order not found"}
    row = df.iloc[0]
    return {
        "order_id": row["order_id"], "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": int(row["delivery_minutes"]) if pd.notna(row["delivery_minutes"]) else None
    }

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps({"action": "issue_refund", "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps({"action": "escalate_to_human", "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order, "search_policy": search_policy,
    "calculate_refund": calculate_refund, "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
with open("outputs/agent_runs.jsonl", "w") as f_out:
    for _, sc in scenarios.iterrows():
        sid, msg = sc["scenario_id"], sc["message"]
        messages = [
            {"role": "system", "content": "AGENT refund handler."},
            {"role": "user", "content": msg}
        ]
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = ""
        calls = 0
        total_tokens = 0
        steps = []
        
        for _ in range(6):
            calls += 1
            resp = mock_llm.chat(messages)
            total_tokens += resp.usage.total_tokens
            tc = resp.choices[0].message.tool_calls
            if not tc:
                final_answer = resp.choices[0].message.content
                status = final_answer.split(":")[0].strip()
                break
            call = tc[0]
            tname, targs = call.name, call.arguments
            res = TOOL_MAP[tname](**targs)
            steps.append({"tool": tname, "args": targs, "res": res})
            messages.append({"role": "assistant", "content": None, "tool_calls": [call.__dict__]})
            messages.append({"role": "tool", "name": tname, "content": json.dumps(res)})
            
        f_out.write(json.dumps({
            "scenario_id": sid, "status": status, "final_answer": final_answer,
            "llm_calls": calls, "tokens_used": total_tokens, "steps": steps
        }) + "\n")

print("Part D Done.")
