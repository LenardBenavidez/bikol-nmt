"""
main.py — Phase 3 API service for the Catanduanes Bikol NMT capstone.

Serves:
  GET  /                      -> the web interface (single-page UI)
  POST /api/v1/translate      -> translate Bikol <-> English
  POST /api/v1/feedback       -> log an inaccurate-translation report (SQLite)
  GET  /api/v1/health         -> simple health check

The model is NLLB-200 + the Phase 2 LoRA adapter (models/nllb_bikol_lora).
Run locally:  uvicorn main:app --reload
(The first request loads the model, which takes a few seconds.)
"""

import os
import time
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_MODEL   = "facebook/nllb-200-distilled-600M"
ADAPTER_DIR  = os.environ.get("ADAPTER_DIR", "models/nllb_bikol_lora")
DB_PATH      = os.environ.get("FEEDBACK_DB", "results/feedback.db")
MAX_LEN      = 128

# The project uses Central Bikol's code (bcl_Latn) as the source tag for the
# Virac variant (NLLB has no Catanduanes code). "bcl_North" in the request maps
# to this. English is eng_Latn.
LANG_CODE = {
    "bcl_North": "bcl_Latn",   # Northern/our Virac Bikol (adapted)
    "bcl_Latn":  "bcl_Latn",
    "eng_Latn":  "eng_Latn",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Model loading (lazy: only on first translate call, so the server starts fast)
# ----------------------------------------------------------------------------
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else BASE_MODEL)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        if os.path.isdir(ADAPTER_DIR):
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        else:
            model = base   # falls back to the un-tuned base if the adapter is missing
        _model = model.to(device).eval()
        _tokenizer = tok
    return _model, _tokenizer

# ----------------------------------------------------------------------------
# Feedback database (SQLite) — criterion C3
# ----------------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_text   TEXT NOT NULL,
                output_text   TEXT NOT NULL,
                source_lang   TEXT,
                target_lang   TEXT,
                user_comment  TEXT,
                timestamp     TEXT NOT NULL
            )
        """)
        conn.commit()

# ----------------------------------------------------------------------------
# Request / response schemas (match the brief exactly)
# ----------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    source_text: str
    source_lang: str = "bcl_North"
    target_lang: str = "eng_Latn"

class FeedbackRequest(BaseModel):
    source_text: str
    output_text: str
    source_lang: str = ""
    target_lang: str = ""
    user_comment: str = ""

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="Catanduanes Bikol NMT API", version="1.0")

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "device": device, "adapter_loaded": os.path.isdir(ADAPTER_DIR)}

@app.post("/api/v1/translate")
def translate(req: TranslateRequest):
    src = LANG_CODE.get(req.source_lang, "bcl_Latn")
    tgt = LANG_CODE.get(req.target_lang, "eng_Latn")
    text = (req.source_text or "").strip()
    if not text:
        return JSONResponse(status_code=400,
                            content={"status": "error", "message": "source_text is empty"})

    model, tok = get_model()
    tok.src_lang = src

    t0 = time.perf_counter()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            max_length=MAX_LEN,
            num_beams=4,
            output_scores=True,
            return_dict_in_generate=True,
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    translated = tok.batch_decode(out.sequences, skip_special_tokens=True)[0]

    # Confidence proxy: exp(mean token log-prob) of the generated sequence.
    try:
        conf = float(torch.exp(out.sequences_scores[0]).item())
        confidence = round(min(max(conf, 0.0), 1.0), 4)
    except Exception:
        confidence = None

    return {
        "status": "success",
        "translated_text": translated,
        "confidence_score": confidence,
        "inference_latency_ms": latency_ms,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
    }

@app.post("/api/v1/feedback")
def feedback(req: FeedbackRequest):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO feedback (source_text, output_text, source_lang, target_lang, user_comment, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (req.source_text, req.output_text, req.source_lang, req.target_lang,
             req.user_comment, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    return {"status": "success", "stored_reports": n}

# ----------------------------------------------------------------------------
# Web interface (criterion C2) — served as a single self-contained page
# ----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catanduanes Bikol Translator</title>
<style>
  :root { --bg:#f4f6f8; --card:#fff; --line:#d9dee3; --accent:#1f6feb; --text:#1a1f24; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:20px 16px; text-align:center; }
  header h1 { margin:0; font-size:1.3rem; }
  header p  { margin:4px 0 0; color:#5a6672; font-size:.9rem; }
  .wrap { max-width:900px; margin:0 auto; padding:16px; }
  .panes { display:flex; gap:16px; flex-wrap:wrap; }
  .pane { flex:1 1 320px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .pane label { display:block; font-size:.8rem; color:#5a6672; margin-bottom:6px; }
  select, textarea { width:100%; font-size:1rem; border:1px solid var(--line); border-radius:8px; padding:10px; font-family:inherit; }
  textarea { height:160px; resize:vertical; }
  .out { white-space:pre-wrap; min-height:160px; background:#fafbfc; border:1px solid var(--line); border-radius:8px; padding:10px; }
  .row { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }
  button { border:0; border-radius:8px; padding:10px 16px; font-size:.95rem; cursor:pointer; }
  .primary { background:var(--accent); color:#fff; }
  .ghost { background:#eef1f4; color:#1a1f24; }
  .meta { font-size:.8rem; color:#5a6672; margin-top:8px; }
  .report { background:#fff3f0; color:#b23; }
  .hidden { display:none; }
  .ok { color:#1a7f37; }
</style>
</head>
<body>
<header>
  <h1>Catanduanes (Virac) Bikol Translator</h1>
  <p>Low-Resource NMT capstone · NLLB-200 fine-tuned with LoRA</p>
</header>
<div class="wrap">
  <div class="panes">
    <div class="pane">
      <label for="srcLang">From</label>
      <select id="srcLang">
        <option value="bcl_North">Catanduanes Bikol</option>
        <option value="eng_Latn">English</option>
      </select>
      <label for="srcText" style="margin-top:10px;">Source text</label>
      <textarea id="srcText" placeholder="Type here..."></textarea>
      <div class="row">
        <button class="primary" onclick="translate()">Translate</button>
      </div>
    </div>
    <div class="pane">
      <label for="tgtLang">To</label>
      <select id="tgtLang">
        <option value="eng_Latn">English</option>
        <option value="bcl_North">Catanduanes Bikol</option>
      </select>
      <label style="margin-top:10px;">Translation</label>
      <div id="outText" class="out"></div>
      <div class="row">
        <button class="ghost" onclick="copyOut()">Copy</button>
        <button class="report" onclick="openReport()">Report Inaccurate Translation</button>
      </div>
      <div id="meta" class="meta"></div>
      <div id="reportBox" class="hidden" style="margin-top:10px;">
        <textarea id="comment" style="height:70px;" placeholder="What was wrong? (optional)"></textarea>
        <div class="row">
          <button class="primary" onclick="sendReport()">Submit report</button>
          <span id="reportMsg" class="meta"></span>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
let last = { src:"", out:"", srcLang:"", tgtLang:"" };

async function translate() {
  const srcText = document.getElementById('srcText').value;
  const srcLang = document.getElementById('srcLang').value;
  const tgtLang = document.getElementById('tgtLang').value;
  const outEl = document.getElementById('outText');
  const metaEl = document.getElementById('meta');
  outEl.textContent = "Translating...";
  metaEl.textContent = "";
  try {
    const r = await fetch('/api/v1/translate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ source_text:srcText, source_lang:srcLang, target_lang:tgtLang })
    });
    const d = await r.json();
    if (d.status === 'success') {
      outEl.textContent = d.translated_text;
      metaEl.textContent = `confidence: ${d.confidence_score ?? 'n/a'} · latency: ${d.inference_latency_ms} ms`;
      last = { src:srcText, out:d.translated_text, srcLang, tgtLang };
    } else {
      outEl.textContent = "[" + (d.message || "error") + "]";
    }
  } catch (e) { outEl.textContent = "[request failed: " + e + "]"; }
}
function copyOut() {
  const t = document.getElementById('outText').textContent;
  navigator.clipboard.writeText(t);
}
function openReport() { document.getElementById('reportBox').classList.remove('hidden'); }
async function sendReport() {
  const comment = document.getElementById('comment').value;
  const msg = document.getElementById('reportMsg');
  try {
    const r = await fetch('/api/v1/feedback', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ source_text:last.src, output_text:last.out,
        source_lang:last.srcLang, target_lang:last.tgtLang, user_comment:comment })
    });
    const d = await r.json();
    msg.innerHTML = '<span class="ok">Report saved (#' + d.stored_reports + '). Thank you!</span>';
  } catch (e) { msg.textContent = "Could not save report."; }
}
</script>
</body>
</html>"""
