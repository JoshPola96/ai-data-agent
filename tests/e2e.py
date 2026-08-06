"""
End-to-end check against a running stack.

Unlike everything under tests/, this makes real LLM calls and therefore costs money.
It is named so that `unittest discover` will not collect it. Run deliberately:

    docker compose up -d
    docker compose exec backend python tests/e2e.py
"""
import base64, io, struct, time, requests

API = "http://127.0.0.1:8000"
ALICE, BOB = "e2e-alice", "e2e-bob"
results = []


def plotly_values(arr):
    """Plotly encodes numeric arrays as base64 typed-arrays, not JSON lists."""
    if isinstance(arr, dict) and "bdata" in arr:
        fmt = {"i4": "i", "i8": "q", "f4": "f", "f8": "d"}[arr["dtype"]]
        raw = base64.b64decode(arr["bdata"])
        return list(struct.unpack("<" + fmt * (len(raw) // struct.calcsize(fmt)), raw))
    return list(arr)

def check(cond, msg):
    results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)

CSV = """region,quarter,revenue,units,margin_pct
North,Q1,120000,340,18.5
North,Q2,135000,390,19.2
North,Q3,128000,355,17.8
North,Q4,171000,470,21.4
South,Q1,98000,290,15.1
South,Q2,104000,310,15.9
South,Q3,99000,295,14.7
South,Q4,142000,430,19.8
East,Q1,156000,410,22.3
East,Q2,161000,425,22.9
East,Q3,149000,395,21.1
East,Q4,203000,540,25.6
West,Q1,87000,260,13.4
West,Q2,91000,275,13.9
West,Q3,88000,265,12.8
West,Q4,119000,360,17.2
"""

def chat(sid, q, rag=True):
    t = time.time()
    r = requests.post(f"{API}/chat", json={"session_id": sid, "query": q, "use_rag": rag}, timeout=300)
    r.raise_for_status()
    d = r.json()
    d["_elapsed"] = round(time.time() - t, 1)
    return d

def charts(d):
    return [v for v in d.get("metadata", {}).get("visualizations", []) if v.get("type") == "chart"]

print("\n=== 1. INGESTION ===")
r = requests.post(f"{API}/upload", files={"file": ("sales.csv", io.StringIO(CSV), "text/csv")},
                  data={"session_id": ALICE}, timeout=180)
up = r.json()
check(r.status_code == 200 and up["dataframes"] == 1, f"CSV ingested: {up['text_chunks']} chunks, {up['dataframes']} table")

dfs = requests.get(f"{API}/sessions/{ALICE}/dataframes", timeout=30).json()["dataframes"]
tbl = next(iter(dfs))
check(dfs[tbl]["rows"] == 16, f"table has {dfs[tbl]['rows']} rows, cols={len(dfs[tbl]['columns'])}")

print("\n=== 2. ANALYSIS + CHART (LLM call 1) ===")
d = chat(ALICE, "What is total revenue by region? Include a bar chart.")
ans = d["response"].replace(",", "")
tools = [s.get("tool_name") for s in d["agent_trace"] if s["type"] == "tool_call"]
cs = charts(d)
# The model may state the figures, chart them, or both. Assert the numbers are right
# wherever they surface, not which route it took to get there.
ys = plotly_values(cs[0]["chart_data"]["chart_json"]["data"][0]["y"]) if cs else []
check("669" in ans or 669000 in ys, f"East total 669000 correct  [{d['_elapsed']}s]")
check(len(cs) >= 1, f"chart attached: {len(cs)}")
check(any(t in tools for t in ("calculate_statistics", "generate_chart", "generate_dashboard")),
      f"analysis tool used: {tools}")
if ys:
    check(sorted(ys) == [385000, 443000, 554000, 669000], f"chart y-values exact: {ys}")

print("\n=== 3. DASHBOARD (LLM call 2) ===")
d = chat(ALICE, "Build a dashboard: revenue by region, revenue trend by quarter, and units vs revenue.")
tools = [s.get("tool_name") for s in d["agent_trace"] if s["type"] == "tool_call"]
n = len(charts(d))
check(n >= 2, f"multi-chart response: {n} charts  [{d['_elapsed']}s]  tools={tools}")
check("generate_dashboard" in tools or n >= 2, "dashboard tool or multiple charts produced")

print("\n=== 4. MULTILINGUAL RAG (LLM call 3) ===")
d = chat(ALICE, "According to the knowledge base, what are the eligibility conditions to register as a job seeker?")
low = d["response"].lower()
check(len(d["response"]) > 120, f"substantive answer  [{d['_elapsed']}s]")
check(any(k in low for k in ["omani", "18", "age", "citizen", "national"]),
      "answered from Arabic KB in English: " + d["response"][:90].replace("\n", " "))
check(bool(d.get("sources")), f"cited sources: {d.get('sources')}")

print("\n=== 5. SESSION ISOLATION (LLM call 4) ===")
d = chat(BOB, "What is the total revenue for the East region in the sales data?")
check("669" not in d["response"].replace(",", ""), "alice's figures not leaked to bob")
check(requests.get(f"{API}/sessions/{BOB}/files", timeout=30).json() == [], "bob has no files")
check(len(requests.get(f"{API}/sessions/{ALICE}/files", timeout=30).json()) == 1, "alice still has hers")

print("\n=== 6. HISTORY + STATS ===")
h = requests.get(f"{API}/sessions/{ALICE}/history", timeout=30).json()
check(h["stats"]["chat_turns"] >= 3, f"history: {h['stats']['chat_turns']} turns, {h['stats']['documents']} docs")
check(all(t["timestamp"] for t in h["history"]), "every turn has a real timestamp")

print("\n=== 7. MODEL SWITCHING ===")
models = requests.get(f"{API}/models", timeout=30).json()
bad = [m["name"] for m in models
       if requests.post(f"{API}/models/select", params={"model_name": m["name"]}, timeout=30).status_code != 200]
check(not bad, f"all {len(models)} advertised models selectable (rejected: {bad})")
requests.post(f"{API}/models/select", params={"model_name": "gemini:gemini-2.5-flash"}, timeout=30)

print("\n=== 8. INPUT BOUNDS ===")
c1 = requests.post(f"{API}/chat", json={"session_id": ALICE, "query": "x" * 5000}, timeout=60).status_code
c2 = requests.post(f"{API}/chat", json={"session_id": "bad:id", "query": "hi"}, timeout=60).status_code
c3 = requests.post(f"{API}/chat", json={"session_id": ALICE, "query": ""}, timeout=60).status_code
check(c1 == 422 and c2 == 422 and c3 == 422, f"oversized/malformed/empty all rejected: {c1},{c2},{c3}")

print("\n=== 9. DELETION ===")
docid = requests.get(f"{API}/sessions/{ALICE}/files", timeout=30).json()[0]["doc_id"]
before = requests.get(f"{API}/status", timeout=30).json()["indexed_vectors"]
requests.delete(f"{API}/sessions/{ALICE}/files/{docid}", timeout=60)
after = requests.get(f"{API}/status", timeout=30).json()["indexed_vectors"]
check(after < before, f"vectors actually removed: {before} -> {after}")

print(f"\n{'='*46}\n{sum(results)}/{len(results)} checks passed\n{'='*46}")
