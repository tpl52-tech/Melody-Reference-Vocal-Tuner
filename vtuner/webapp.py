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
            smooth_ms=f("smooth_ms", 95.0), transpose=int(f("transpose", 0)),
            isolate_reference=request.form.get("isolate_reference") == "1",
            isolate_take=request.form.get("isolate_take") == "1",
            mix=request.form.get("mix") == "1",
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
        cover=base("cover"), cover_error=res.outputs.get("cover_error"),
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#e7e1d2; --card:#f1ecdd; --ink:#3b3529; --olive:#6b6151; --sub:#938a76;
    --line:#d7cfbc; --sage:#aec3a6;
    --amber:#d7a44f; --red:#c0795a; --lime:#aec3a6; --green:#7f9a6a; --yellow:#e2cd84;
    color-scheme:light;
  }
  *{box-sizing:border-box;}
  body{margin:0; background:var(--bg); color:var(--ink);
       font-family:"Space Grotesk",-apple-system,system-ui,sans-serif; font-size:15px; line-height:1.5;
       -webkit-font-smoothing:antialiased;}
  .wrap{max-width:860px; margin:0 auto; padding:40px 20px 90px;}

  .kicker{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:22px;}
  .tag-pill{font-size:12px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
            background:var(--ink); color:var(--card); padding:8px 15px; border-radius:100px;}

  .hero{display:flex; justify-content:space-between; align-items:center; gap:26px; flex-wrap:wrap; margin-bottom:34px;}
  .hero-text{flex:1 1 320px;}
  h1.display{font-size:clamp(42px,8vw,78px); font-weight:700; letter-spacing:-.03em;
             line-height:.93; margin:4px 0 16px;}
  .lede{font-size:17px; color:var(--sub); max-width:460px; margin:0; font-weight:500;}
  .lede b{color:var(--ink); font-weight:700;}

  /* spinning vinyl motif */
  .vinyl{width:132px; height:132px; flex:none; background:var(--sage); border:2px solid var(--olive);
         border-radius:20px; box-shadow:7px 7px 0 0 var(--olive); display:grid; place-items:center;}
  .vinyl span{width:94px; height:94px; border-radius:50%; animation:spin 4s linear infinite;
     background:radial-gradient(circle,#f1ecdd 0 14%,var(--olive) 15% 29%,#f1ecdd 30% 35%,
       var(--olive) 36% 61%,#f1ecdd 62% 65%,var(--olive) 66%);}
  @keyframes spin{to{transform:rotate(360deg);}}

  .card{background:var(--card); border:2px solid var(--olive); border-radius:22px;
        padding:24px; margin-bottom:22px; box-shadow:6px 6px 0 0 var(--olive);}
  .row{display:flex; gap:18px; flex-wrap:wrap;}
  .col{flex:1 1 300px; min-width:250px;}
  .badge{width:34px; height:34px; border-radius:50%; display:inline-grid; place-items:center;
         font-weight:700; margin-right:11px; border:2px solid var(--olive); color:var(--ink);}
  .lbl{font-weight:700; font-size:16px; letter-spacing:-.01em; margin:0 0 13px; display:flex; align-items:center;}
  .sub{color:var(--sub); font-weight:500; font-size:13px;}

  input[type=file]{width:100%; font:inherit; font-size:13px; color:var(--sub);}
  input[type=file]::file-selector-button{font:inherit; font-weight:700; font-size:13px; border:2px solid var(--olive);
     cursor:pointer; background:var(--sage); color:var(--ink); padding:8px 14px; border-radius:100px; margin-right:12px;}

  .toggle{display:flex; align-items:center; gap:11px; margin-top:13px; font-size:13.5px; cursor:pointer; font-weight:600;}
  .toggle input{appearance:none; -webkit-appearance:none; width:44px; height:25px; background:var(--card);
     border:2px solid var(--olive); border-radius:100px; position:relative; cursor:pointer; transition:.15s; flex:none;}
  .toggle input:checked{background:var(--sage);}
  .toggle input::after{content:""; position:absolute; top:2px; left:2px; width:17px; height:17px;
     background:var(--olive); border-radius:50%; transition:.15s;}
  .toggle input:checked::after{left:21px;}

  .rec{width:56px; height:56px; border-radius:50%; border:2px solid var(--olive); cursor:pointer;
       background:var(--red); color:var(--card); font-size:16px; display:inline-grid; place-items:center;
       box-shadow:4px 4px 0 0 var(--olive); transition:.1s; flex:none;}
  .rec.on{background:var(--ink);}
  .rec:active{transform:translate(2px,2px); box-shadow:2px 2px 0 0 var(--olive);}

  details.set > summary{cursor:pointer; font-weight:700; list-style:none; padding:2px 0; font-size:15px;}
  details.set > summary::-webkit-details-marker{display:none;}
  details.set > summary::before{content:"＋ "; color:var(--sub);}
  details.set[open] > summary::before{content:"－ ";}
  .item{margin:18px 0;}
  .item .cap{font-weight:600; font-size:13.5px; margin-bottom:9px; display:flex; justify-content:space-between;
     align-items:baseline; gap:8px;}
  .item .cap small{color:var(--sub); font-weight:500; margin-left:auto; margin-right:10px;}
  .val{font-variant-numeric:tabular-nums; background:var(--ink); color:var(--card); padding:3px 11px;
     border-radius:100px; font-size:12px; font-weight:600;}

  .seg{display:inline-flex; gap:5px; flex-wrap:wrap; background:var(--line); padding:4px; border-radius:100px;
     border:2px solid var(--olive);}
  .seg button{font:inherit; font-weight:600; font-size:12.5px; border:0; cursor:pointer; background:transparent;
     color:var(--sub); padding:8px 15px; border-radius:100px; transition:.1s;}
  .seg button.sel{background:var(--sage); color:var(--ink);}

  input[type=range]{width:100%; accent-color:var(--olive); height:4px;}

  .actions{display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin:26px 0 8px;}
  button.primary{font:inherit; font-weight:700; font-size:17px; border:2px solid var(--olive); cursor:pointer;
     background:var(--ink); color:var(--card); padding:15px 32px; border-radius:100px; box-shadow:5px 5px 0 0 var(--olive); transition:.1s;}
  button.primary:active{transform:translate(3px,3px); box-shadow:2px 2px 0 0 var(--olive);}
  button.primary:disabled{opacity:.5; cursor:default; transform:none; box-shadow:5px 5px 0 0 var(--olive);}
  button.ghost{font:inherit; font-weight:700; font-size:15px; border:2px solid var(--olive); cursor:pointer;
     background:var(--card); color:var(--ink); padding:14px 24px; border-radius:100px; box-shadow:4px 4px 0 0 var(--olive); transition:.1s;}
  button.ghost:active{transform:translate(2px,2px); box-shadow:2px 2px 0 0 var(--olive);}
  button.ghost:disabled{opacity:.5;}

  #status{margin:22px 0; min-height:20px; color:var(--sub); font-weight:500;}
  .metric{background:#e4ecda; border:2px solid var(--green); color:#3f5533; padding:17px 19px;
     border-radius:18px; font-weight:500; line-height:1.7; box-shadow:5px 5px 0 0 var(--green);}
  .metric b{color:var(--ink); font-weight:700;}
  .muted{color:var(--sub);}

  .note{width:38px; height:38px; border-radius:50%; display:inline-grid; place-items:center;
     font-weight:700; margin-right:11px; font-size:15px; border:2px solid var(--olive); color:var(--ink);}
  audio{width:100%; margin-top:10px;}

  /* mini spinning vinyl as the loader */
  .spin{display:inline-block; width:18px; height:18px; border-radius:50%; vertical-align:-3px; margin-right:9px;
     background:radial-gradient(circle,#f1ecdd 0 20%,var(--ink) 21% 42%,#f1ecdd 43% 50%,var(--ink) 51%);
     animation:spin .9s linear infinite;}
  @media(max-width:560px){ h1.display{font-size:50px;} .vinyl{width:108px;height:108px;} .vinyl span{width:76px;height:76px;} }
</style></head>
<body><div class="wrap">
  <div class="kicker">
    <span class="tag-pill">Melody-Reference Vocal Tuner</span>
  </div>
  <div class="hero">
    <div class="hero-text">
      <h1 class="display">Follow the<br>melody.</h1>
      <p class="lede">Sing along to a song and get retuned to its <b>actual melody</b> — not a generic scale. No key to pick, no notes to edit.</p>
    </div>
    <div class="vinyl" aria-hidden="true"><span></span></div>
  </div>

  <div class="card">
    <div class="row">
      <div class="col">
        <div class="lbl"><span class="badge" style="background:var(--amber)">1</span> Reference vocal</div>
        <input type="file" id="reference" accept="audio/*">
        <label class="toggle"><input type="checkbox" id="isoRef"> <span>It's a full song — isolate the vocal <span class="sub">· Demucs</span></span></label>
        <label class="toggle"><input type="checkbox" id="mix"> <span>Produce a cover on the real instrumental <span class="sub">· full song</span></span></label>
      </div>
      <div class="col">
        <div class="lbl"><span class="badge" style="background:var(--lime)">2</span> Your take</div>
        <input type="file" id="take" accept="audio/*">
        <div style="margin-top:14px; display:flex; align-items:center; gap:14px;">
          <button class="rec" id="recBtn" type="button" title="Record">●</button>
          <span class="sub" id="recInfo">record, or upload above</span>
        </div>
        <audio id="recPlay" controls hidden></audio>
      </div>
    </div>

    <details class="set" style="margin-top:20px; border-top:1px solid var(--line); padding-top:14px;">
      <summary>Settings <span class="sub">— tuned for a natural sound</span></summary>
      <div class="item"><div class="cap">Alignment</div>
        <span class="seg" data-name="align_mode">
          <button data-v="dtw" class="sel">DTW · tolerates drift</button>
          <button data-v="offset">Offset</button></span></div>
      <div class="item"><div class="cap">Pitch-shifter</div>
        <span class="seg" data-name="backend">
          <button data-v="world" class="sel">WORLD</button>
          <button data-v="rubberband">RubberBand</button>
          <button data-v="both">Both</button></span></div>
      <div class="item"><div class="cap">CREPE model</div>
        <span class="seg" data-name="model">
          <button data-v="full" class="sel">Full · best</button>
          <button data-v="tiny">Tiny · fast</button></span></div>
      <div class="item"><div class="cap">Correction strength <small>lower = more natural</small> <span class="val" id="sV">0.60</span></div>
        <input type="range" id="strength" min="0" max="1" step="0.05" value="0.6"></div>
      <div class="item"><div class="cap">Keep vibrato <span class="val" id="pV">1.00</span></div>
        <input type="range" id="preserve" min="0" max="1" step="0.05" value="1.0"></div>
      <div class="item"><div class="cap">Glide across notes <small>higher = less robotic</small> <span class="val" id="mV">95</span></div>
        <input type="range" id="smooth_ms" min="0" max="200" step="5" value="95"></div>
      <div class="item"><div class="cap">Transpose · key shift <span class="val" id="tV">0</span></div>
        <input type="range" id="transpose" min="-12" max="12" step="1" value="0"></div>
    </details>
  </div>

  <div class="actions">
    <button class="primary" id="tuneBtn">Tune ↗</button>
    <button class="ghost" id="exBtn">Load NSYNC example</button>
  </div>

  <div id="status"></div>

  <div class="card players" id="results" hidden>
    <div class="row">
      <div class="col"><div class="lbl"><span class="note" style="background:var(--line)">R</span>Before</div><audio id="aRaw" controls></audio></div>
      <div class="col" id="wCol"><div class="lbl"><span class="note" style="background:var(--green)">W</span>After · WORLD</div><audio id="aWorld" controls></audio></div>
      <div class="col" id="rCol" hidden><div class="lbl"><span class="note" style="background:var(--amber)">R</span>After · RubberBand</div><audio id="aRb" controls></audio></div>
    </div>
    <div id="coverWrap" hidden style="margin-top:20px">
      <div class="lbl"><span class="note" style="background:var(--red)">♪</span>Produced cover <span class="sub" style="margin-left:8px">your voice on the real instrumental</span></div>
      <audio id="aCover" controls></audio>
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
for (const [id,out] of [['strength','sV'],['preserve','pV'],['smooth_ms','mV'],['transpose','tV']]) {
  const el=$('#'+id), o=$('#'+out);
  const fmt = id==='smooth_ms' ? (v=>v)
            : id==='transpose' ? (v=> (v>0?'+':'')+v)
            : (v=>Number(v).toFixed(2));
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
      btn.classList.remove('on'); btn.textContent='●';
    };
    mediaRec.start(); btn.classList.add('on'); btn.textContent='■';
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
  fd.append('transpose', $('#transpose').value);
  let isolating = false;
  if (useExample) { fd.append('example','1'); }
  else {
    const ref = $('#reference').files[0];
    const take = $('#take').files[0] || (recBlob ? new File([recBlob],'recording.webm') : null);
    if (!ref || !take) { setStatus('⚠️ Provide a reference vocal and your take (upload or record).'); return; }
    fd.append('reference', ref); fd.append('take', take);
    if ($('#isoRef').checked) { fd.append('isolate_reference','1'); isolating = true; }
    if ($('#mix').checked) { fd.append('mix','1'); isolating = true; }
  }
  const slow = seg.model==='full' && !useExample;
  setStatus('<span class="spin"></span>Tuning…'
    + (isolating ? ' (separating with Demucs — first run downloads a model; can take a few min)'
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
    if (d.cover){ $('#coverWrap').hidden=false; $('#aCover').src='/audio/'+d.cover+bust; }
    else { $('#coverWrap').hidden=true; if (d.cover_error) $('#status').innerHTML += '<br><span class="muted">cover failed: '+d.cover_error+'</span>'; }
  } catch(e){ setStatus('❌ '+e); }
  finally { $('#tuneBtn').disabled = $('#exBtn').disabled = false; }
}
$('#tuneBtn').onclick = () => runTune(false);
$('#exBtn').onclick = () => runTune(true);
</script>
</div></body></html>"""


if __name__ == "__main__":
    main()
