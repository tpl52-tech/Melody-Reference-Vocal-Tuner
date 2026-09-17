"""Phase 3 demo UI: upload/record -> tune -> before/after playback.

A tiny local Flask app wrapping the pipeline so a judge can go from audio to a
tuned result without the terminal. Run:

    python -m vtuner.webapp

then open http://127.0.0.1:7860 . Built on Flask (not Gradio) to avoid
dependency conflicts with the ML stack on Python 3.9."""
from __future__ import annotations

import os
import uuid

from flask import Flask, request, jsonify, send_from_directory, Response

from vtuner import pipeline

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "output")
UPLOAD_DIR = os.path.join(OUT_DIR, "uploads")
EXAMPLE_REF = os.path.join(HERE, "data", "ref.wav")
EXAMPLE_TAKE = os.path.join(HERE, "data", "take.wav")

app = Flask(__name__)


def _save_upload(fs) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(fs.filename or "")[1] or ".webm"
    name = f"{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    fs.save(path)
    return path


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/audio/<path:name>")
def audio(name):
    return send_from_directory(OUT_DIR, name, mimetype="audio/wav")


@app.route("/tune", methods=["POST"])
def tune():
    try:
        if request.form.get("example") == "1":
            ref_path, take_path = EXAMPLE_REF, EXAMPLE_TAKE
            take_stem = "take"
        else:
            ref = request.files.get("reference")
            tk = request.files.get("take")
            if not ref or not tk:
                return jsonify(error="Please provide both a reference vocal and your take."), 400
            ref_path = _save_upload(ref)
            take_path = _save_upload(tk)
            take_stem = os.path.splitext(os.path.basename(take_path))[0]

        def f(name, default):
            try:
                return float(request.form.get(name, default))
            except (TypeError, ValueError):
                return default

        res = pipeline.run(
            ref_path, take_path, out_dir=OUT_DIR,
            backend=request.form.get("backend", "world"),
            align_mode=request.form.get("align_mode", "dtw"),
            model=request.form.get("model", "full"),
            strength=f("strength", 0.6), preserve=f("preserve", 1.0),
            smooth_ms=f("smooth_ms", 95.0),
            isolate_reference=request.form.get("isolate_reference") == "1",
            isolate_take=request.form.get("isolate_take") == "1",
            measure_output=False,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500

    m = res.metrics
    if res.align_mode == "dtw":
        sync = f"DTW matched {res.warp_matched_fraction*100:.0f}% of your frames (timing drift handled)"
    else:
        sync = f"global offset {res.alignment.lag_seconds*1000:+.0f} ms (confidence {res.alignment.confidence:.2f})"

    def base(role):
        p = res.outputs.get(role)
        return os.path.basename(p) if p else None

    return jsonify(
        raw=base("raw"), world=base("world"), rubberband=base("rubberband"),
        notes=res.n_notes, register=res.register_offset, sync=sync,
        cents_before=round(m["mean_abs_cents_before"]),
        cents_after=round(m["mean_abs_cents_after_predicted"]),
    )


def main(host="127.0.0.1", port=7860):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n  Melody-Reference Vocal Tuner — open  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Melody-Reference Vocal Tuner</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:#0f1115; color:#e7e9ee; }
  .wrap { max-width:900px; margin:0 auto; padding:32px 18px 64px; }
  h1 { font-size:26px; margin:0 0 4px; }
  .tag { color:#9aa3b2; margin:0 0 28px; }
  .card { background:#181b22; border:1px solid #262b36; border-radius:14px; padding:18px; margin-bottom:16px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .col { flex:1 1 300px; }
  label.blk { display:block; font-weight:600; margin-bottom:6px; }
  input[type=file] { width:100%; }
  button { font:inherit; border:0; border-radius:10px; padding:11px 18px; cursor:pointer; }
  .primary { background:#6c8cff; color:#0b0d12; font-weight:700; }
  .ghost { background:#242a36; color:#e7e9ee; }
  .rec { background:#e5484d; color:#fff; }
  .rec.on { background:#8b1a1d; }
  details summary { cursor:pointer; font-weight:600; color:#c7ccd6; }
  .set { margin-top:14px; }
  .set .item { margin:12px 0; }
  .set .item small { color:#9aa3b2; }
  input[type=range] { width:100%; }
  .seg { display:inline-flex; gap:6px; }
  .seg button { background:#242a36; color:#c7ccd6; padding:6px 12px; border-radius:8px; }
  .seg button.sel { background:#6c8cff; color:#0b0d12; font-weight:700; }
  #status { margin:16px 0; min-height:22px; color:#c7ccd6; }
  .metric { background:#12331f; border:1px solid #1e5233; color:#c8f0d6; padding:12px 14px; border-radius:10px; }
  .players .col { min-width:250px; }
  audio { width:100%; margin-top:6px; }
  .muted { color:#9aa3b2; font-size:13px; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #6c8cff; border-top-color:transparent;
          border-radius:50%; animation:s .8s linear infinite; vertical-align:-2px; margin-right:8px; }
  @keyframes s { to { transform:rotate(360deg); } }
</style></head>
<body><div class="wrap">
  <h1>🎤 Melody-Reference Vocal Tuner</h1>
  <p class="tag">Sing along to a song and get retuned to follow its <b>actual melody</b> — not a generic scale. No key or scale to pick.</p>

  <div class="card"><div class="row">
    <div class="col">
      <label class="blk">1 · Reference vocal <span class="muted">(isolated)</span></label>
      <input type="file" id="reference" accept="audio/*">
      <label style="display:block;margin-top:8px;font-size:13px;color:#c7ccd6">
        <input type="checkbox" id="isoRef"> It's a full song — isolate the vocal
        <span class="muted">(Demucs; slower, first run downloads a model)</span></label>
    </div>
    <div class="col">
      <label class="blk">2 · Your take</label>
      <input type="file" id="take" accept="audio/*">
      <div style="margin-top:8px">
        <button class="rec" id="recBtn" type="button">● Record</button>
        <span class="muted" id="recInfo">or upload above</span>
        <audio id="recPlay" controls hidden></audio>
      </div>
    </div>
  </div>

  <details class="set"><summary>Settings <span class="muted">(defaults tuned for a natural sound)</span></summary>
    <div class="item"><label class="blk">Alignment</label>
      <span class="seg" data-name="align_mode">
        <button data-v="dtw" class="sel">DTW (tolerates drift)</button>
        <button data-v="offset">Offset</button></span></div>
    <div class="item"><label class="blk">Pitch-shifter</label>
      <span class="seg" data-name="backend">
        <button data-v="world" class="sel">WORLD</button>
        <button data-v="rubberband">RubberBand</button>
        <button data-v="both">Both</button></span></div>
    <div class="item"><label class="blk">CREPE model</label>
      <span class="seg" data-name="model">
        <button data-v="full" class="sel">Full (best)</button>
        <button data-v="tiny">Tiny (fast)</button></span></div>
    <div class="item"><label class="blk">Correction strength: <span id="sV">0.60</span> <small>(lower = more natural)</small></label>
      <input type="range" id="strength" min="0" max="1" step="0.05" value="0.6"></div>
    <div class="item"><label class="blk">Keep vibrato: <span id="pV">1.00</span></label>
      <input type="range" id="preserve" min="0" max="1" step="0.05" value="1.0"></div>
    <div class="item"><label class="blk">Glide across notes: <span id="mV">95</span> ms <small>(higher = less robotic)</small></label>
      <input type="range" id="smooth_ms" min="0" max="200" step="5" value="95"></div>
  </details>
  </div>

  <div style="display:flex; gap:12px; flex-wrap:wrap">
    <button class="primary" id="tuneBtn">Tune 🎚️</button>
    <button class="ghost" id="exBtn">Load NSYNC example &amp; tune</button>
  </div>

  <div id="status"></div>

  <div class="card players" id="results" hidden>
    <div class="row">
      <div class="col"><label class="blk">Before <span class="muted">(your take)</span></label><audio id="aRaw" controls></audio></div>
      <div class="col" id="wCol"><label class="blk">After — WORLD</label><audio id="aWorld" controls></audio></div>
      <div class="col" id="rCol" hidden><label class="blk">After — RubberBand</label><audio id="aRb" controls></audio></div>
    </div>
  </div>

<script>
const $ = s => document.querySelector(s);
const seg = {}; // name -> value
document.querySelectorAll('.seg').forEach(g => {
  const name = g.dataset.name;
  seg[name] = g.querySelector('.sel').dataset.v;
  g.querySelectorAll('button').forEach(b => b.onclick = () => {
    g.querySelectorAll('button').forEach(x => x.classList.remove('sel'));
    b.classList.add('sel'); seg[name] = b.dataset.v;
  });
});
for (const [id,out] of [['strength','sV'],['preserve','pV'],['smooth_ms','mV']]) {
  const el=$('#'+id), o=$('#'+out);
  const fmt = id==='smooth_ms' ? v=>v : v=>Number(v).toFixed(2);
  el.oninput = () => o.textContent = fmt(el.value);
}

// --- mic recording ---
let mediaRec, chunks=[], recBlob=null;
$('#recBtn').onclick = async () => {
  const btn=$('#recBtn');
  if (mediaRec && mediaRec.state === 'recording') { mediaRec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRec = new MediaRecorder(stream); chunks=[];
    mediaRec.ondataavailable = e => chunks.push(e.data);
    mediaRec.onstop = () => {
      recBlob = new Blob(chunks, {type:'audio/webm'});
      const url = URL.createObjectURL(recBlob);
      $('#recPlay').src=url; $('#recPlay').hidden=false;
      $('#recInfo').textContent='recorded ✓ (will be used as your take)';
      stream.getTracks().forEach(t=>t.stop());
      btn.classList.remove('on'); btn.textContent='● Record';
    };
    mediaRec.start(); btn.classList.add('on'); btn.textContent='■ Stop';
    $('#recInfo').textContent='recording…';
  } catch(e){ $('#recInfo').textContent='mic blocked — upload instead'; }
};

function setStatus(html){ $('#status').innerHTML = html; }

async function runTune(useExample){
  const fd = new FormData();
  fd.append('align_mode', seg.align_mode);
  fd.append('backend', seg.backend);
  fd.append('model', seg.model);
  fd.append('strength', $('#strength').value);
  fd.append('preserve', $('#preserve').value);
  fd.append('smooth_ms', $('#smooth_ms').value);
  let isolating = false;
  if (useExample) { fd.append('example','1'); }
  else {
    const ref = $('#reference').files[0];
    const take = $('#take').files[0] || (recBlob ? new File([recBlob],'recording.webm') : null);
    if (!ref || !take) { setStatus('⚠️ Provide a reference vocal and your take (upload or record).'); return; }
    fd.append('reference', ref); fd.append('take', take);
    if ($('#isoRef').checked) { fd.append('isolate_reference','1'); isolating = true; }
  }
  const slow = seg.model==='full' && !useExample;
  setStatus('<span class="spin"></span>Tuning…'
    + (isolating ? ' (isolating the vocal with Demucs — first run downloads a model, can take a few min)'
                 : (slow ? ' (full model on a fresh clip can take 1–2 min on CPU)' : '')));
  $('#tuneBtn').disabled = $('#exBtn').disabled = true;
  try {
    const r = await fetch('/tune', {method:'POST', body:fd});
    const d = await r.json();
    if (!r.ok) { setStatus('❌ ' + (d.error||'error')); return; }
    setStatus('<div class="metric">Followed the reference melody — <b>'+d.notes+' notes</b>, '
      +'<b>no key or scale selected</b>. Register fold '+(d.register>=0?'+':'')+d.register+' st.<br>'
      +d.sync+'.<br>Pitch error to the melody: <b>'+d.cents_before+' → '+d.cents_after+' cents</b>.</div>');
    const bust = '?t=' + Date.now();
    $('#results').hidden = false;
    $('#aRaw').src = '/audio/'+d.raw+bust;
    if (d.world){ $('#wCol').hidden=false; $('#aWorld').src='/audio/'+d.world+bust; } else $('#wCol').hidden=true;
    if (d.rubberband){ $('#rCol').hidden=false; $('#aRb').src='/audio/'+d.rubberband+bust; } else $('#rCol').hidden=true;
  } catch(e){ setStatus('❌ '+e); }
  finally { $('#tuneBtn').disabled = $('#exBtn').disabled = false; }
}
$('#tuneBtn').onclick = () => runTune(false);
$('#exBtn').onclick = () => runTune(true);
</script>
</div></body></html>"""


if __name__ == "__main__":
    main()
