#!/usr/bin/env python3
"""Apply config-aware campaign gates to a shared Complexa preflight report.

CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged, and express the
campaign's choices as arguments. Validated by CBLN1/5KC5, first complete run
2026-08-28.

What was hardcoded in the campaign this came from, and is now derived or passed:

  * the ESM model      -- read from `metric.esm_model`, which the resolved config
                          already carries, rather than repeated here
  * ESMFold2 HF repos  -- `--require-hf-repo`, repeatable. Deliberately NOT derived
                          from backend names: that mapping lives in the pipeline
                          (`folding_models.py`) and copying it here would give the
                          template its own stale copy of someone else's constant
  * the VRAM floor     -- `--min-vram-gb`, default 40
  * ESMC/ESMFold2 imports -- checked only when the config asks for them

A campaign that uses plain ESMFold and colabdesign passes no extra repos and gets
no ESMFold2 import check, which the original could not express.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import yaml

def hf_repo_present(root: Path, repo: str) -> bool:
    d=root/("models--"+repo.replace("/","--"))
    return d.is_dir() and any((d/"snapshots").glob("*"))

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("preflight",type=Path); p.add_argument("--resolved-config",type=Path,required=True); p.add_argument("--expected-designs",type=int,required=True)
    p.add_argument("--require-hf-repo",action="append",default=[],metavar="REPO",
                   help="HF repo that must have a usable snapshot; repeatable")
    p.add_argument("--min-vram-gb",type=int,default=40)
    a=p.parse_args()
    data=json.loads(a.preflight.read_text()); cfg=yaml.safe_load(a.resolved_config.read_text()); metric=cfg["metric"]; failures=[]
    gpu=data.get("gpu",{}); cm=data.get("community_models",{}); tools=data.get("tools",{})
    if not gpu.get("available"): failures.append("no CUDA GPU visible")
    elif int(gpu.get("vram_gb",0))<a.min_vram_gb: failures.append(f"visible GPU has <{a.min_vram_gb} GB VRAM")
    for ck in ("complexa.ckpt","complexa_ae.ckpt"):
        if not data.get("checkpoints",{}).get(ck,{}).get("exists"): failures.append(f"missing {ck}")
    if metric.get("binder_folding_method")=="colabdesign" and not cm.get("AF2_DIR",{}).get("exists"): failures.append("missing AF2_DIR")
    for tool in ("foldseek","mmseqs"):
        if not tools.get(tool,{}).get("exists"): failures.append(f"missing {tool}")
    community=Path(os.environ.get("COMMUNITY_MODELS_PATH", os.path.join(os.environ.get("COMPLEXA_REPO",""),"community_models")))
    ckpt=Path(os.environ.get("SOLUBLE_MPNN_CKPT", community/"LigandMPNN/model_params/solublempnn_v_48_020.pt"))
    if not ckpt.is_file(): failures.append(f"missing soluble ProteinMPNN checkpoint: {ckpt}")
    hf=Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME",Path.home()/".cache/huggingface"))/"hub")))
    repos=list(a.require_hf_repo)
    # The config names the ESM model, so it does not need naming twice.
    if metric.get("compute_esm_metrics") and metric.get("esm_model"): repos.append(metric["esm_model"])
    for repo in dict.fromkeys(repos):
        if not hf_repo_present(hf,repo): failures.append(f"HF cache lacks usable snapshot for {repo} under {hf}")
    # Only when something in the config actually routes to them. Checking
    # unconditionally fails a perfectly good plain-ESMFold campaign.
    backends={str(x) for k in ("consensus_backends","apo_folding_models","monomer_folding_models") for x in (metric.get(k) or [])}
    if "esmfold2" in backends or str(metric.get("esm_backend","")).startswith("esmc"):
        try:
            import esm  # noqa: F401
            from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: F401
            from transformers import AutoModelForMaskedLM  # noqa: F401
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: F401
        except Exception as exc: failures.append(f"ESMC/ESMFold2 imports failed: {exc}")
    need=max(5,int(a.expected_designs/100*20*2)); free=data.get("disk",{}).get("cwd_free_gb")
    if free is not None and float(free)<need: failures.append(f"campaign filesystem has {free} GB free; estimate requires {need} GB")
    if data.get("env",{}).get("missing_required"): failures.append(f"missing required env: {data['env']['missing_required']}")
    for f in failures: print("FAIL:",f)
    if failures:return 1
    print("PASS: resolved-config preflight requirements satisfied"); return 0
if __name__=="__main__": raise SystemExit(main())
