#!/usr/bin/env python3
"""Verify sharding, global filtering, configured tracks, and analyzed outputs.

CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged. Validated by
CBLN1/5KC5, first complete run 2026-08-28.

Four things were hardcoded in the campaign this came from and are now arguments,
because each is a campaign choice rather than a property of Complexa:

  * shard count      -- was the glob `rewards_pipeline_[01].csv`, i.e. exactly two
  * the trim report  -- was `shard_trim_{'smoke' if expected_retained==8 else
                        'production'}.json`, keyed off a magic 8; now `--trim-report`
  * result columns   -- were a fixed list naming esmfold2 and esmc columns, which a
                        campaign using plain ESMFold does not produce
  * redesign model   -- was pinned to soluble_mpnn

Defaults are deliberately permissive: with no `--require-column` this checks the
shape of a run (counts reconcile, tracks match the config, one combined CSV) and
not which metrics it chose to compute. Name the columns your campaign depends on.
"""
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
import yaml

def rows(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def one(root,pat):
    x=sorted(root.glob(pat))
    if len(x)!=1:raise SystemExit(f"expected one {pat} under {root}, found {len(x)}")
    return x[0]
def samples(root):
    return [p for p in root.iterdir() if p.is_dir() and p.name not in {"filtered_out_samples","timing"} and any(p.glob("*.pdb"))]
def main():
    p=argparse.ArgumentParser();p.add_argument("--inference-dir",type=Path,required=True);p.add_argument("--evaluation-dir",type=Path,required=True);p.add_argument("--expected-retained",type=int,required=True);p.add_argument("--resolved-config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shards",type=int,default=2)
    p.add_argument("--trim-report",type=Path,required=True,help="the JSON trim_shards.py wrote")
    p.add_argument("--require-column",action="append",default=[],metavar="COL",
                   help="result column the campaign depends on; repeatable")
    p.add_argument("--redesign-model",default=None,help="expected redesign_model value, if pinned")
    a=p.parse_args()
    rewards=[a.inference_dir/f"rewards_pipeline_{i}.csv" for i in range(a.shards)]
    for r in rewards:
        if not r.is_file(): raise SystemExit(f"missing {r}; --shards says {a.shards}")
    raw=sum(len(rows(x)) for x in rewards)
    trim=json.loads(a.trim_report.read_text())
    retained=sum(v["retained"] for v in trim["shards"].values())
    if retained!=a.expected_retained:raise SystemExit(f"retained {retained}, expected {a.expected_retained}")
    if any(v["retained"]!=a.expected_retained//a.shards for v in trim["shards"].values()):raise SystemExit("unequal shard retention")
    live=samples(a.inference_dir)
    timing=[x for x in a.evaluation_dir.glob("timing_*.csv") if re.match(r"^timing_\d+\.csv$",x.name)]
    timing_rows=[r for x in timing for r in rows(x)]; evaluated=sum(int(r["nsamples"]) for r in timing_rows)
    ran={t for r in timing_rows for t in r.get("evals_run","").split("+") if t}
    cfg=yaml.safe_load(a.resolved_config.read_text()); wanted={x for x,f in (("binder","compute_binder_metrics"),("monomer","compute_monomer_metrics")) if cfg["metric"].get(f)}
    if ran!=wanted:raise SystemExit(f"tracks {ran}, expected {wanted}")
    if evaluated!=len(live):raise SystemExit(f"evaluated {evaluated}, live inputs {len(live)}")
    combined=one(a.evaluation_dir,"RAW_protein_binder_results_*_combined.csv"); data=rows(combined)
    if len(data)!=evaluated:raise SystemExit("combined row count differs from evaluated count")
    cols=set(data[0]) if data else set()
    missing=[x for x in a.require_column if x not in cols]
    if missing:raise SystemExit("missing result columns: "+", ".join(missing))
    if a.redesign_model and any(r.get("redesign_model") not in (a.redesign_model,"") for r in data):
        raise SystemExit(f"unexpected redesign model (expected {a.redesign_model})")
    if not list((a.evaluation_dir/"filter_results").glob("res_filter_*_pass_*.csv")):raise SystemExit("missing success output")
    report={"status":"passed","raw_generation_rows":raw,"retained_before_global_dedup":retained,"live_after_global_dedup":len(live),"evaluated":evaluated,"combined_rows":len(data),"tracks":sorted(ran),"required_columns":list(a.require_column)}
    a.output.write_text(json.dumps(report,indent=2)+"\n");print("PASS:",report);return 0
if __name__=="__main__":raise SystemExit(main())
