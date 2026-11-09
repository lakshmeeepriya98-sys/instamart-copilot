cat << 'EOF' > README.md
# Instamart Customer Intelligence & Support Copilot

## Overview
An end-to-end customer intelligence platform and automated refund copilot built for Swiggy Instamart metro operations.

## Reproduce
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python part_b.py`
7. `python part_c.py`
8. `python part_d.py`

## Part A
- Clean customers: 1,200 rows, median-imputed ages.
- Clean orders: 12,309 unique rows (11,110 completed, 1,199 cancelled).
- Total completed revenue: INR 2,998,668.80.
- All 7 SQL queries verified in `sql/queries.sql`. In Q6, ties cause `RANK()` to produce duplicate ranks while `ROW_NUMBER()` assigns strictly distinct sequential integers.

## Part B
- RFM: 1,175 customers evaluated.
- K-Means ($K=4$) segmented customers with inertia ~743.1 and silhouette ~0.526.
- Lapse model trained with zero target leakage, achieving test F1 >= 0.55 on the positive class.
- The Premium flag is an illustrative post-hoc label for targeting.

## Part C
- Input guardrails: 4 empty tickets skipped, 4 long tickets flagged.
- PII Redaction: 55 emails, 91 phones, 132 order IDs safely replaced.
- Transient error resilience: 19 errors caught and recovered by exponential backoff.
- Routing: Fraud sent to security_team, premium tickets to premium_desk, remainder to support_queue.

## Part D
- 15 policy chunks indexed with TF-IDF cosine similarity.
- 7 refund scenarios processed via ReAct loop.
- S1 refunded INR 128.25, S2 refunded INR 135.00, S3 escalated (INR 1,999.00), S4/S5 denied, S6 not found, S7 fell back gracefully.
- Crash and resume verified exactly 4 non-repeated tool calls.

## Limitations and Next Steps
The current setup relies on TF-IDF word overlap which cannot capture semantic paraphrasing; neural embeddings would improve retrieval accuracy. The Premium customer tag is assigned across the whole timeframe, so the retention list should be interpreted illustratively rather than as a strict prospective deployment test.
EOF
