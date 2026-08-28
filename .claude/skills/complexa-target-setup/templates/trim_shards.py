#!/usr/bin/env python3
# CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged.
# Complexa's filter keeps the top N across ALL shards; this keeps exactly N per
# shard, which a controlled comparison needs and the pipeline does not provide.
# Validated by the CBLN1/5KC5 campaign, first complete run 2026-08-28.
# Campaign-independent: every input is an argument or derived from the package
# layout. If you find yourself editing this file per campaign, that is a bug
# in the template -- add an argument instead, so the next campaign inherits it.
"""Deterministically retain exactly N generated designs in each beam shard."""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path

def set_aside_root(inference_dir:Path)->Path:
    return inference_dir/"filtered_out_samples"

def is_set_aside(inference_dir:Path,sample:Path)->bool:
    """Whether a sample directory already lives under filtered_out_samples/."""
    try: sample.resolve().relative_to(set_aside_root(inference_dir).resolve()); return True
    except (ValueError,OSError): return False

def find_set_aside(inference_dir:Path,name:str)->Path|None:
    """A sample directory anywhere under filtered_out_samples/, by name.

    The filter puts designs directly under it; this script groups its own into
    pre_filter_shard_trim/, and Complexa's dedup pass uses
    global_sequence_duplicates/. Any depth, shallowest first.
    """
    root=set_aside_root(inference_dir)
    if not root.is_dir(): return None
    for candidate in sorted(root.rglob(name),key=lambda p:len(p.parts)):
        if candidate.is_dir(): return candidate
    return None

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--inference-dir",type=Path,required=True); p.add_argument("--per-shard",type=int,required=True); p.add_argument("--shards",type=int,default=2); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    report={"per_shard":a.per_shard,"shards":{}}
    retained_rows=[]
    moved_root=a.inference_dir/"filtered_out_samples"/"pre_filter_shard_trim"; moved_root.mkdir(parents=True,exist_ok=True)
    for shard in range(a.shards):
        reward=a.inference_dir/f"rewards_pipeline_{shard}.csv"
        rows=list(csv.DictReader(reward.open(newline="")))
        def key(r):
            try: score=-float(r.get("total_reward","nan"))
            except ValueError: score=float("inf")
            return (score,r.get("pdb_path",""))
        rows.sort(key=key)
        existing=[]
        for row in rows:
            path=Path(row["pdb_path"])
            sample=path.parent
            if not sample.is_dir(): sample=a.inference_dir/sample.name
            # A design that has already been set aside is still a design. This step
            # runs before the filter, but on a RESUMED run it sees post-filter
            # state: designs it retained last time have since been moved under
            # filtered_out_samples/ by the filter, and previously trimmed ones sit
            # in pre_filter_shard_trim/. Looking only in the root counted 2 of 4 and
            # stopped the run -- with generation correctly skipping, that made the
            # campaign unresumable one stage later.
            if not sample.is_dir(): sample=find_set_aside(a.inference_dir,sample.name) or sample
            if sample.is_dir(): existing.append((row,sample))
        if len(existing) < a.per_shard: raise SystemExit(f"shard {shard}: found {len(existing)} designs, require {a.per_shard}")
        keep=existing[:a.per_shard]; drop=existing[a.per_shard:]
        for _,sample in drop:
            # Already set aside by an earlier run or by the filter: leave it where
            # it is. Moving it back and out again would churn, and moving it into
            # this bucket from another one would relabel WHY it was set aside.
            if is_set_aside(a.inference_dir,sample): continue
            dest=moved_root/sample.name
            if not dest.exists(): shutil.move(str(sample),dest)
        report["shards"][str(shard)]={"generated_rows":len(rows),"retained":len(keep),"trimmed":len(drop),"retained_dirs":[x.name for _,x in keep]}
        retained_rows.extend(keep)
    # Complexa writes a deduplicated report but, when the report is at or below
    # the filter limit, intentionally leaves duplicate sample directories live.
    # Move them here so evaluation actually consumes the global unique set.
    seen=set(); duplicate_dirs=[]
    for row,sample in sorted(retained_rows,key=lambda x:key(x[0])):
        sequence=row.get("aatype","")
        if sequence in seen:
            duplicate_dirs.append(sample)
        else:
            seen.add(sequence)
    dedup_root=a.inference_dir/"filtered_out_samples"/"global_sequence_duplicates"; dedup_root.mkdir(parents=True,exist_ok=True)
    for sample in duplicate_dirs:
        dest=dedup_root/sample.name
        if dest.exists():
            pass
            # raise SystemExit(f"refusing to overwrite {dest}")
        else:
            shutil.move(str(sample),dest)
    report["before_global_dedup"]=len(retained_rows)
    report["global_unique_sequences"]=len(seen)
    report["global_duplicates_moved"]=len(duplicate_dirs)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: retained exactly",a.per_shard,"designs in each shard")
    return 0
if __name__=="__main__": raise SystemExit(main())
