#!/usr/bin/env python3
"""Prepare a blinded expert annotation package without generating labels."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parents[1]
OUT = REVISION_ROOT / "06_analysis_outputs/expert_audit"
PACKAGE = OUT / "blinded_annotation_package"
SEED = 4524210
DOMAINS = ("WBC", "TBI", "PBS", "BMA", "TBF")
COUNTS = {"WBC": 25773, "TBI": 30, "PBS": 77, "BMA": 38, "TBF": 182}
ROOTS = {
    "WBC": Path("/mnt/d/New_folder/datasets/WBC_dataset1"),
    "TBI": Path("/mnt/d/New_folder/datasets/New folder/TBSI/folders"),
    "PBS": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/pbs_imgs"),
    "BMA": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/bma_imgs"),
    "TBF": Path("/mnt/d/New_folder/datasets/New folder/bma pbf tfa/tbf_imgs"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WBC_SLIDE_MAP = REVISION_ROOT / "06_analysis_outputs/statistical_inference/wbc_slide_mapping/wbc_stack_to_slide.csv"


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def folder_map(domain: str) -> list[Path]:
    if domain == "WBC":
        return [ROOTS[domain] / str(index) for index in range(COUNTS[domain])]
    return sorted((Path(entry.path) for entry in os.scandir(ROOTS[domain]) if entry.is_dir()), key=lambda path: natural_key(path.name))


def image_paths(folder: Path) -> list[str]:
    return [str(path) for path in sorted((Path(entry.path) for entry in os.scandir(folder) if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES and ":Zone.Identifier" not in entry.name), key=lambda path: natural_key(path.name))]


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True); (PACKAGE / "annotations").mkdir(exist_ok=True)
    rng = random.Random(SEED)
    selected: list[dict] = []
    disagreements = list(csv.DictReader((REVISION_ROOT / "06_analysis_outputs/reference_audit/affected_stacks.csv").open()))
    wbc_rows = list(csv.DictReader(WBC_SLIDE_MAP.open(newline="", encoding="utf-8")))
    wbc_by_slide: dict[int, list[int]] = {}
    for row in wbc_rows:
        wbc_by_slide.setdefault(int(row["slide_num"]), []).append(int(row["stack_index"]))
    wbc_slide_by_stack = {int(row["stack_index"]): int(row["slide_num"]) for row in wbc_rows}
    for domain in DOMAINS:
        folders = folder_map(domain)
        if len(folders) != COUNTS[domain]: raise RuntimeError(f"{domain} folder count {len(folders)} != {COUNTS[domain]}")
        if domain == "WBC":
            sampled_slides = sorted(rng.sample(sorted(wbc_by_slide), 30))
            primary = sorted(rng.choice(wbc_by_slide[slide]) for slide in sampled_slides)
        else:
            primary = sorted(rng.sample(range(COUNTS[domain]), 30))
        domain_disagreement = [row for row in disagreements if row["domain"] == domain and row["comparison"] == "REF-B_corrected_ten_vs_REF-C_disjoint_four"]
        domain_disagreement.sort(key=lambda row: (-int(row["absolute_shift"]), int(row["stack_index"])))
        enriched = [int(row["stack_index"]) for row in domain_disagreement[:10]]
        for sample_type, indices in (("primary_unbiased", primary), ("disagreement_enriched_qualitative", enriched)):
            for index in indices:
                paths = image_paths(folders[index])
                anonymous = hashlib.sha256(f"{SEED}:{domain}:{index}:{sample_type}".encode()).hexdigest()[:16]
                selected.append({"anonymous_stack_id": anonymous, "domain": domain, "stack_index": index, "sample_type": sample_type, "source_folder": str(folders[index]), "slice_count": len(paths), "image_paths_json": json.dumps(paths), "selection_seed": SEED, "slide_or_specimen_id": f"WBC_slide_{wbc_slide_by_stack[index]}" if domain == "WBC" else "not available", "patient_id": "not available", "algorithm_outputs_exposed": False, "consensus_label_exposed": False})
    rng.shuffle(selected)
    with (OUT / "annotation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    blind = [{"anonymous_stack_id": row["anonymous_stack_id"], "slice_count": row["slice_count"], "sample_type": row["sample_type"]} for row in selected]
    (PACKAGE / "blinded_manifest.json").write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    template_fields = ["anonymous_stack_id", "annotator_id", "role", "started_at", "submitted_at", "best_focus_slice", "acceptable_interval_start", "acceptable_interval_end", "uncertain", "ungradable", "notes"]
    for filename in ("annotation_template.csv", "adjudication_template.csv"):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=template_fields); writer.writeheader()
    (PACKAGE / "index.html").write_text('''<!doctype html><html><head><meta charset="utf-8"><title>Blinded focus annotation</title><style>body{font:16px system-ui;max-width:1050px;margin:24px auto;color:#172033}header{display:flex;justify-content:space-between}img{display:block;max-width:100%;max-height:68vh;margin:12px auto;background:#111}fieldset{display:flex;gap:18px;flex-wrap:wrap;padding:12px}input[type=range]{width:100%}button{padding:9px 16px}.muted{color:#64748b}</style></head><body><header><h1>Blinded focus annotation</h1><div id="progress"></div></header><p class="muted">Algorithm outputs, consensus indices, dataset names and source identifiers are hidden.</p><label>Annotator ID <input id="annotator" required></label> <label>Role <select id="role"><option>assessor_1</option><option>assessor_2</option><option>adjudicator</option></select></label><img id="image"><input type="range" id="slice" min="0" step="1"><div id="sliceLabel"></div><fieldset><label>Best-focus slice <input id="best" type="number" min="0"></label><label>Acceptable start <input id="start" type="number" min="0"></label><label>Acceptable end <input id="end" type="number" min="0"></label><label><input id="uncertain" type="checkbox"> Uncertain</label><label><input id="ungradable" type="checkbox"> Ungradable</label><label>Notes <input id="notes"></label></fieldset><button id="previous">Previous</button> <button id="save">Save & next</button><script>let items=[],i=0,started=new Date().toISOString();const $=x=>document.getElementById(x);fetch('/manifest').then(r=>r.json()).then(x=>{items=x;show()});function show(){const x=items[i];$('progress').textContent=`${i+1} / ${items.length}`;$('slice').max=x.slice_count-1;$('slice').value=Math.floor(x.slice_count/2);['best','start','end'].forEach(k=>{$(k).max=x.slice_count-1;$(k).value=''});['uncertain','ungradable'].forEach(k=>$(k).checked=false);$('notes').value='';started=new Date().toISOString();paint()}function paint(){const x=items[i],z=$('slice').value;$('image').src=`/image?id=${x.anonymous_stack_id}&slice=${z}`;$('sliceLabel').textContent=`Slice ${z} of ${x.slice_count-1}`}$('slice').oninput=paint;$('previous').onclick=()=>{if(i>0){i--;show()}};$('save').onclick=async()=>{if(!$('annotator').value)return alert('Annotator ID is required');const x=items[i],payload={anonymous_stack_id:x.anonymous_stack_id,annotator_id:$('annotator').value,role:$('role').value,started_at:started,submitted_at:new Date().toISOString(),best_focus_slice:$('best').value,acceptable_interval_start:$('start').value,acceptable_interval_end:$('end').value,uncertain:$('uncertain').checked,ungradable:$('ungradable').checked,notes:$('notes').value};const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok)return alert(await r.text());if(i<items.length-1){i++;show()}else alert('Annotation set complete.')}};</script></body></html>''', encoding="utf-8")
    (PACKAGE / "serve_annotations.py").write_text('''#!/usr/bin/env python3
import csv,json,sys
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import cv2 as cv, numpy as np
ROOT=Path(__file__).resolve().parent; OUT=ROOT.parent
rows=list(csv.DictReader((OUT/'annotation_manifest.csv').open())); lookup={r['anonymous_stack_id']:r for r in rows}
blind=json.loads((ROOT/'blinded_manifest.json').read_text())
class H(BaseHTTPRequestHandler):
 def send(self,data,ctype='application/json',status=200):
  self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_GET(self):
  from urllib.parse import urlparse,parse_qs
  u=urlparse(self.path)
  if u.path=='/': return self.send((ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
  if u.path=='/manifest': return self.send(json.dumps(blind).encode())
  if u.path=='/image':
   q=parse_qs(u.query); ident=q.get('id',[''])[0]; z=int(q.get('slice',['0'])[0]); row=lookup.get(ident)
   if not row:return self.send(b'unknown id','text/plain',404)
   paths=json.loads(row['image_paths_json']);
   if not 0<=z<len(paths):return self.send(b'invalid slice','text/plain',400)
   stack=[cv.imread(p,cv.IMREAD_UNCHANGED) for p in paths]; low=min(float(x.min()) for x in stack); high=max(float(x.max()) for x in stack); image=stack[z]; display=np.clip((image.astype(np.float32)-low)/(high-low+1e-12)*255,0,255).astype(np.uint8); ok,data=cv.imencode('.jpg',display)
   return self.send(data.tobytes(),'image/jpeg') if ok else self.send(b'encode failed','text/plain',500)
  return self.send(b'not found','text/plain',404)
 def do_POST(self):
  if self.path!='/save':return self.send(b'not found','text/plain',404)
  try:
   n=int(self.headers.get('Content-Length','0')); row=json.loads(self.rfile.read(n)); ident=row.get('anonymous_stack_id');
   if ident not in lookup: raise ValueError('unknown blinded stack')
   target=ROOT/'annotations'/f"{row.get('annotator_id','anonymous')}.csv"; target.parent.mkdir(exist_ok=True); fields=['anonymous_stack_id','annotator_id','role','started_at','submitted_at','best_focus_slice','acceptable_interval_start','acceptable_interval_end','uncertain','ungradable','notes']; exists=target.exists()
   with target.open('a',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); (w.writeheader() if not exists else None); w.writerow({k:row.get(k,'') for k in fields})
   return self.send(b'{"ok":true}')
  except Exception as e:return self.send(str(e).encode(),'text/plain',400)
if __name__=='__main__':
 port=int(sys.argv[1]) if len(sys.argv)>1 else 8765; print(f'Open http://127.0.0.1:{port}'); ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()
''', encoding="utf-8")
    config = {"seed": SEED, "primary_sample": "30 stacks per domain; WBC samples 30 distinct official slides then one cell per slide, other domains sample stacks without replacement", "primary_count": sum(row["sample_type"] == "primary_unbiased" for row in selected), "qualitative_sample": "up to 10 largest REF-B-versus-REF-C disagreements per domain", "qualitative_count": sum(row["sample_type"].startswith("disagreement") for row in selected), "WBC_slide_balanced_sampling": True, "WBC_slide_clusters_available": len(wbc_by_slide), "patient_level_WBC_sampling": "blocked: official CSV does not map 214 slides to 72 patients", "stack_order_randomized": True, "display_normalization": "single min/max across every slice in the displayed stack", "algorithm_outputs_hidden": True, "consensus_hidden": True}
    (OUT / "annotation_configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (OUT / "annotation_instructions.md").write_text("""# Blinded expert annotation instructions

Run `python blinded_annotation_package/serve_annotations.py` from the expert-audit directory in an environment containing OpenCV and NumPy, then open the printed local URL. Each assessor works independently with a unique ID. Scroll every z-slice, select one best-focus slice, optionally enter an inclusive acceptable-focus interval, and mark uncertain or ungradable when appropriate. Do not consult algorithm curves, consensus indices, or another assessor. A third adjudicator reviews disagreements only after both assessor files are locked.

The primary sample is the inferential set. The disagreement-enriched set is qualitative and must never be pooled into unbiased agreement estimates. Within-stack display normalization is fixed across all slices. Raw images are served read-only from their source locations and are not copied into this package.
""", encoding="utf-8")
    (OUT / "EXPERT_AUDIT_BLOCKER.md").write_text("""# Expert audit blocker

No expert focus annotations were supplied, so REF-E and expert-agreement figures/tables cannot be computed. The complete blinded package is ready for two independent assessors and a third adjudicator. After genuine files are returned, refresh: expert inter-rater agreement, expert-consensus reference, algorithmic-consensus comparison, operator localization against expert consensus, acceptable-interval performance, the expert figure, manuscript Results/Limitations, and reviewer-response numerical locations.

The official `slide_number.csv` was recovered and exactly reconciled to the 25,773 frozen WBC stack positions. The primary WBC annotation sample is therefore balanced across 30 distinct slides. A slide-to-patient mapping is not supplied, so patient-balanced sampling across the reported 72 patients remains unavailable.
""", encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
