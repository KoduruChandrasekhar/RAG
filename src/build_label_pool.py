"""Build the set of passages that need judging, and a tool to judge them.

The pool is the union of what every ablation configuration retrieves. Pooling
across configurations matters: labelling only what the current pipeline returns
would mark a rival configuration's passages as unlabelled and quietly score it
down for retrieving something nobody looked at.

    python src/build_label_pool.py

Writes two files:
    src/label_pool.json   the passages to judge
    label_passages.html   a self-contained tool to judge them in

Open the HTML file in a browser. It keeps progress in the browser as you go, so
you can stop and come back. When you are done (or part-done) use its Download
button and save the result over src/relevance_labels.json.
"""
import argparse
import collections
import json
import os

import ablation
import paths
import relevance
from query_engine import FilingScopeEngine

POOL_PATH = os.path.join(paths.SRC_DIR, "label_pool.json")
TOOL_PATH = os.path.join(paths.PROJECT_ROOT, "label_passages.html")


def build_pool(engine, questions, top_k=5):
    pool = []
    for item in questions:
        seen = {}
        appearances = collections.Counter()

        for _, runner, override in ablation.CONFIGURATIONS:
            tickers = override if override is not None \
                else engine._detect_tickers(item["question"])
            result = runner(engine, item["question"], tickers, top_k)
            for source in result["sources"]:
                cid = source.get("chunk_id") or source.get("id")
                if cid is None:
                    # No derived id would be stable: Python salts string
                    # hashing per process, so a hash-based key would orphan
                    # every label on the next run.
                    raise RuntimeError(
                        "retrieval returned a passage without a chunk_id; "
                        "labels key on it and cannot be stored without one")
                appearances[cid] += 1
                seen.setdefault(cid, source)

        n_configs = len(ablation.CONFIGURATIONS)
        for cid, source in seen.items():
            pool.append({
                "question": item["question"],
                "chunk_id": cid,
                "ticker": source["ticker"],
                "company": paths.company_name(source["ticker"]),
                "year": paths.filing_year(source["filing_id"]),
                "section": paths.section_label(source["section"], short=True),
                "text": source["text"],
                # Passages every configuration returns cannot change which one
                # wins, so judge the contested ones first.
                "contested": appearances[cid] < n_configs,
            })

    # Contested passages first, then grouped by question for context.
    order = {q["question"]: i for i, q in enumerate(questions)}
    pool.sort(key=lambda p: (not p["contested"], order[p["question"]]))
    return pool


def write_tool(pool, existing_labels, path=TOOL_PATH):
    payload = json.dumps(pool, ensure_ascii=False)
    prior = json.dumps(existing_labels, ensure_ascii=False)
    html = TOOL_TEMPLATE.replace("__POOL__", payload).replace("__PRIOR__", prior)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


TOOL_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>FilingScope - passage labelling</title>
<style>
  :root {
    --bg:#f6f7fb; --card:#fff; --ink:#1f2733; --soft:#5d6678; --line:#dfe3ec;
    --accent:#4254a0; --good:#2f7a4f; --mid:#9a7420; --bad:#9a3b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14181f; --card:#1b212b; --ink:#e8ecf5; --soft:#a6b0c4;
            --line:#2c3545; --accent:#8fa2e8; --good:#6fcf97; --mid:#e0b65c;
            --bad:#e58a8a; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:24px 16px 120px }
  h1 { font-size:20px; margin:0 0 4px }
  .sub { color:var(--soft); font-size:13px; margin-bottom:20px }
  .bar { height:6px; background:var(--line); border-radius:99px; overflow:hidden; margin:14px 0 }
  .bar > i { display:block; height:100%; background:var(--accent); width:0 }
  .q { background:var(--card); border:1px solid var(--line); border-left:3px solid var(--accent);
       border-radius:8px; padding:14px 16px; margin-bottom:14px }
  .q b { display:block; font-size:12px; letter-spacing:.08em; text-transform:uppercase;
         color:var(--soft); margin-bottom:6px; font-weight:600 }
  .meta { font-size:12px; color:var(--soft); margin-bottom:8px }
  .tag { display:inline-block; border:1px solid var(--line); border-radius:99px;
         padding:1px 9px; margin-right:6px }
  .p { background:var(--card); border:1px solid var(--line); border-radius:8px;
       padding:16px 18px; white-space:pre-wrap }
  .btns { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap }
  button { font:inherit; padding:9px 14px; border-radius:8px; cursor:pointer;
           border:1px solid var(--line); background:var(--card); color:var(--ink) }
  button:hover { border-color:var(--accent) }
  button kbd { font:inherit; font-weight:700; opacity:.75; margin-right:6px }
  .g2 { border-color:var(--good); color:var(--good) }
  .g1 { border-color:var(--mid);  color:var(--mid) }
  .g0 { border-color:var(--bad);  color:var(--bad) }
  .done { text-align:center; padding:60px 20px }
  .foot { position:fixed; left:0; right:0; bottom:0; background:var(--card);
          border-top:1px solid var(--line); padding:10px 16px; font-size:13px;
          display:flex; gap:14px; align-items:center; justify-content:center; flex-wrap:wrap }
  a.dl { color:var(--accent); font-weight:600; text-decoration:none }
</style>
<div class="wrap">
  <h1>Passage labelling</h1>
  <div class="sub">Does this passage help answer the question?
    <b>2</b> directly answers part of it &middot;
    <b>1</b> related or useful background &middot;
    <b>0</b> no help.
    Use the number keys. <b>&larr;</b> goes back.</div>
  <div class="bar"><i id="bar"></i></div>
  <div id="view"></div>
</div>
<div class="foot">
  <span id="count"></span>
  <a class="dl" href="#" id="dl">Download relevance_labels.json</a>
  <span style="color:var(--soft)">save it over <code>src/relevance_labels.json</code></span>
</div>
<script>
const POOL = __POOL__;
const PRIOR = __PRIOR__;
const KEY = "filingscope-labels-v1";

function load() {
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  // Anything already committed to the repo wins on first load.
  const merged = JSON.parse(JSON.stringify(PRIOR));
  for (const q in stored) {
    merged[q] = Object.assign({}, merged[q] || {}, stored[q]);
  }
  return merged;
}
let labels = load();
function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
}
function graded(p) {
  return labels[p.question] && labels[p.question][p.chunk_id] !== undefined;
}
let i = POOL.findIndex(p => !graded(p));
if (i < 0) i = POOL.length;

function setGrade(g) {
  const p = POOL[i];
  if (!p) return;
  (labels[p.question] = labels[p.question] || {})[p.chunk_id] = g;
  persist();
  i++;
  render();
}
function back() {
  if (i > 0) { i--; render(); }
}
function render() {
  const done = POOL.filter(graded).length;
  document.getElementById("bar").style.width = (100 * done / POOL.length) + "%";
  document.getElementById("count").textContent =
    done + " of " + POOL.length + " judged";

  const view = document.getElementById("view");
  const p = POOL[i];
  if (!p) {
    view.innerHTML = '<div class="done"><h1>All judged.</h1>' +
      '<p class="sub">Download the file below and save it over ' +
      '<code>src/relevance_labels.json</code>, then re-run ' +
      '<code>python src/ablation.py</code>.</p></div>';
    return;
  }
  const current = (labels[p.question] || {})[p.chunk_id];
  view.innerHTML =
    '<div class="q"><b>Question</b>' + esc(p.question) + '</div>' +
    '<div class="meta">' +
      '<span class="tag">' + esc(p.company) + '</span>' +
      '<span class="tag">' + (p.year ? p.year + ' 10-K' : 'filing') + '</span>' +
      '<span class="tag">' + esc(p.section) + '</span>' +
      (p.contested ? '<span class="tag">contested</span>' : '') +
      (current !== undefined ? '<span class="tag">currently ' + current + '</span>' : '') +
    '</div>' +
    '<div class="p">' + esc(p.text) + '</div>' +
    '<div class="btns">' +
      '<button class="g2" onclick="setGrade(2)"><kbd>2</kbd>answers part of it</button>' +
      '<button class="g1" onclick="setGrade(1)"><kbd>1</kbd>related</button>' +
      '<button class="g0" onclick="setGrade(0)"><kbd>0</kbd>no help</button>' +
      '<button onclick="back()"><kbd>&larr;</kbd>back</button>' +
    '</div>';
}
function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
document.addEventListener("keydown", e => {
  if (e.key === "0" || e.key === "1" || e.key === "2") setGrade(+e.key);
  else if (e.key === "ArrowLeft") back();
});
document.getElementById("dl").addEventListener("click", e => {
  e.preventDefault();
  const blob = new Blob([JSON.stringify(labels, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "relevance_labels.json";
  a.click();
});
render();
</script>
</html>
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="build_label_pool",
        description="Build the passage pool and the labelling tool.")
    parser.add_argument("-k", "--top-k", type=int, default=5, metavar="N",
                        help="passages per configuration to pool (default 5)")
    return parser


def main():
    args = build_parser().parse_args()

    with open(paths.TEST_QUESTIONS, "r", encoding="utf-8") as f:
        questions = json.load(f)

    engine = FilingScopeEngine()
    pool = build_pool(engine, questions, args.top_k)
    existing = relevance.load_labels()

    with open(POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    write_tool(pool, existing)

    already = sum(len(v) for v in existing.values())
    contested = sum(1 for p in pool if p["contested"])

    print(f"\nPool: {len(pool)} passages across {len(questions)} questions")
    print(f"  contested (configurations disagree): {contested}")
    print(f"  already labelled                   : {already}")
    print(f"\nWrote {POOL_PATH}")
    print(f"Wrote {TOOL_PATH}")
    print(f"\nOpen that file in a browser and judge with the 0/1/2 keys.")
    print(f"Progress is kept in the browser, so you can stop and resume.")
    print(f"Download when done and save over {relevance.LABELS_PATH}")


if __name__ == "__main__":
    main()
