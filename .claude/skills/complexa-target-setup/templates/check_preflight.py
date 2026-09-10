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
  * hotspot resolution -- read from the target entry the config actually selects,
                          and checked against the PDB it actually points at
  * the target MSA     -- present, and deep enough to load, when the config names one

A campaign that uses plain ESMFold and colabdesign passes no extra repos and gets
no ESMFold2 import check, which the original could not express.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def needs_protein_interface(cfg: dict, metric: dict) -> bool:
    """Whether anything in this config routes to the protein-interface extension.

    Three routes, and the first is the one this gate used to miss. Every binder
    evaluation of a *protein* target asks for the interface residues of every
    design -- they are what ``aa_interface_counts`` counts and what ``mpnn_fixed``
    holds fixed -- so the extension is reached whether or not the bioinformatics
    columns are switched on. A ligand target takes the atomistic path instead and
    never touches it. The cached-refresh path reaches it the same way.

    The other two are the ones it did catch: the bioinformatics interface metrics
    on the generated structures, the refolded ones, or both; and generation
    through BioinformaticsRewardModel. A sub-flag that is absent counts as on,
    which is what evaluate.py itself defaults to -- reading it the other way would
    let a config that will call it pass a gate that says it will not.

    ``result_type`` is how a ligand target announces itself; absent, this assumes
    protein, because over-requiring a declared dependency costs nothing and
    under-requiring it costs a GPU job that dies hours in.

    No binary is part of any of this: shape complementarity runs in process
    through protein-interface, and secondary structure comes from mdtraj. The
    requirement is that the extension imports.
    """
    # Matched to the gate that decides whether the binder track runs at all
    # (`evaluate.py:110`, default False), not to the inner call site inside that
    # track (`binder_eval.py:638`, default True). Reading the inner one would
    # demand the extension from a monomer-only campaign that never mentions
    # binders.
    if metric.get("compute_binder_metrics", False) and "ligand" not in str(cfg.get("result_type", "")):
        return True
    for parent, child in (
        ("compute_pre_refolding_metrics", "pre_refolding"),
        ("compute_refolded_structure_metrics", "refolded"),
    ):
        if metric.get(parent) and (metric.get(child) or {}).get("bioinformatics", True):
            return True
    return "BioinformaticsRewardModel" in json.dumps(cfg, default=str)


def target_hotspots(cfg: dict) -> tuple[str, str, list[str]] | None:
    """The (target_path, target_input, hotspots) this run will actually use, or None.

    None means nothing here warrants the check, and every reason for it is a real
    campaign shape rather than a defensive crouch. A monomer run declares no
    target. A ligand target never reads hotspots -- ``hotspot_residues`` is
    present on all four live ligand entries as ``[null]`` and only
    ``binder_generate.yaml:35`` interpolates it, which is the protein path. And an
    empty hotspot list is legal: ``13_BBF14`` and ``14_CrSAS6`` ship that way, and
    it means "no epitope focus", not "unset".

    The entry is looked up through ``task_name`` rather than by scanning
    ``target_dict_cfg``, because the dict holds all 44 shared targets plus this
    campaign's own, and checking the wrong one is exactly the failure this gate
    exists to catch -- an unpinned ``task_name`` inheriting ``33_TrkA`` produces a
    clean run against the wrong target.
    """
    gen = cfg.get("generation") or {}
    entry = (gen.get("target_dict_cfg") or {}).get(gen.get("task_name")) or {}
    if not entry or entry.get("ligand") is not None or "ligand" in str(cfg.get("result_type", "")):
        return None
    wanted = [str(h) for h in (entry.get("hotspot_residues") or []) if h is not None]
    if not wanted:
        return None
    return entry.get("target_path"), entry.get("target_input"), wanted


def ca_ids_from_pdb(path: str, spec: str) -> set[str]:
    """The residue ids a hotspot can address, read the way generation reads them.

    Not a second implementation of anything: ``from_contig`` is the pipeline's own
    contig parser (`pdb_utils.py:554`), so a grammar this cannot parse is one
    generation cannot parse either, and a numbering convention it disagrees with
    cannot exist.

    Masking BEFORE collecting the CAs is what makes this stricter than
    ``check_target_pdb.py``, which matches over the whole chain. Generation masks
    on the contig first (`pdb_utils.py:556`) and only then matches, so a hotspot
    that sits in the chain but outside ``target_input`` is silently dropped --
    and that script would call it resolved.

    ``get_mask`` is strict in a way worth knowing: it raises
    ``ValueError: No atoms found for selection: A/*/116`` on the FIRST residue of
    the contig that the file does not contain, rather than returning a shorter
    mask. Measured on atomworks 2.2.1 against ``PD-L1.pdb`` (chain A, 1-115):
    ``A1-115`` gives 115 CAs, ``A1-116`` and ``B1-115`` both raise. So a contig
    that does not fit its file is a hard error in generation too, not a quiet
    truncation -- the caller reports it rather than treating it as "no misses".
    """
    from atomworks.io.utils.io_utils import load_any
    from atomworks.io.utils.selection import AtomSelectionStack

    struct = load_any(path, model=1)
    # load_any hands back a stack for some formats; generation takes model 1.
    if getattr(struct, "ndim", 1) > 1 or struct.__class__.__name__.endswith("Stack"):
        struct = struct[0]
    struct = struct[AtomSelectionStack.from_contig(spec).get_mask(struct)]
    ca = struct[struct.atom_name == "CA"]
    return {f"{a.chain_id}{a.res_id}" for a in ca}


def hotspot_failures(cfg: dict, read=ca_ids_from_pdb) -> list[str]:
    """Fail a campaign whose hotspots will not resolve, before it spends a GPU.

    This is the one input-structure fault that costs a whole campaign without
    producing a single error. Hotspots are matched as ``f"{chain_id}{res_id}"``
    and misses are silent (`pdb_utils.py:571-575`): a wrong chain letter, a
    file numbered from 18 rather than 1, or a .cif read as label_seq_id all
    yield an all-False mask, and the run then completes and designs something
    with no epitope guidance at all. ``complexa validate target`` does not catch
    it either -- it never opens the PDB (`cli/validate.py:377-499`).
    """
    want = target_hotspots(cfg)
    if want is None:
        return []
    path, spec, wanted = want
    if not path:
        # source + target_filename would make this guess a file extension, and a
        # wrong guess is a false failure. Loud, so an unchecked campaign is not
        # mistaken for a checked one.
        print("  SKIP hotspots: target entry has no target_path")
        return []
    if not spec:
        return [f"{len(wanted)} hotspot(s) declared but target_input is unset; generation masks on "
                "the contig before matching, so none of them can resolve"]
    pdb = Path(path)
    if not pdb.is_file() and not pdb.is_absolute():
        pdb = Path(os.environ.get("COMPLEXA_REPO", "")) / path
    if not pdb.is_file():
        return [f"target PDB not found: {path}"]
    try:
        ids = read(str(pdb), spec)
    except Exception as exc:
        # Most often the contig naming a residue the file does not have, which
        # `get_mask` raises on. Generation masks with the same selector, so this
        # is that run's own exception, arriving before the checkpoint loads
        # instead of after.
        return [f"cannot apply target_input {spec} to {pdb}: {type(exc).__name__}: {exc} -- "
                "generation masks with the same selector (`pdb_utils.py:555`) and would raise too"]
    miss = [h for h in wanted if h not in ids]
    if miss:
        return [f"hotspot(s) absent from {pdb} under target_input {spec}: {miss} -- matched as "
                "chain+res_id strings with no warning, so this run would design against no epitope"]
    print(f"  {len(wanted)} hotspot(s) resolve in {pdb.name} under {spec}")
    return []


def target_msa_failures(cfg: dict, metric: dict) -> list[str]:
    """Fail a config that names a target MSA which is not there.

    ``_load_msa`` raises ``FileNotFoundError: target MSA not found`` when folding
    first reaches it -- per design, deep into evaluate, on a run whose generation
    already finished. The path is knowable at submit time, so this is knowable at
    submit time.

    Gated on the backends actually being on, like every other check here: an
    ``a3m`` named by a config whose ``consensus_backends`` is empty is never
    opened, and failing that campaign would be failing a correct one. Ligand
    targets skip consensus folding entirely -- there is no target sequence to fold
    against -- so they skip this too.

    Existence and a depth of at least two records, which is what the loader
    demands (`consensus_folding.py:329`) and all that can be checked without
    parsing the alignment. Whether the query MATCHES the chain is settled where
    the alignment is written: prepare_target_msa.py validates through the
    pipeline's own ``_target_msas`` before it returns. Re-parsing a 16k-sequence
    a3m here to re-derive that would cost real time on every stage of every run.
    """
    if not (metric.get("consensus_backends") or []):
        return []
    if "ligand" in str(cfg.get("result_type", "")):
        return []
    consensus = metric.get("consensus_cfg") or {}
    paths = consensus.get("target_msa_paths")
    if paths is None:
        single = consensus.get("target_msa")
        paths = [single] if single else []
    failures = []
    for path in paths:
        if not path:  # null is legal: that chain simply has no alignment
            continue
        msa = Path(path)
        # pipeline.yaml is meant to compose ${oc.env:CAMPAIGN_DIR}/${oc.env:TARGET_MSA},
        # so this is normally absolute. A config that names the package-relative form
        # directly still works when the runner calls this from CAMPAIGN_DIR, and the
        # fallback covers being called from anywhere else.
        if not msa.is_file() and not msa.is_absolute():
            msa = Path(os.environ.get("CAMPAIGN_DIR", "")) / path
        if not msa.is_file():
            failures.append(
                f"consensus_cfg names a target MSA that is not there: {path} -- "
                "prepare_target_msa.py writes it, and campaign.env's TARGET_MSA is what "
                "keeps the two paths the same string"
            )
            continue
        # Cheap depth probe. A prepare step killed mid-write, or a path pointing at
        # something that is not an alignment, both land here rather than at evaluate.
        with msa.open(encoding="utf-8", errors="replace") as handle:
            records = sum(1 for line in handle if line.startswith(">"))
        if records < 2:
            failures.append(f"target MSA {path} holds {records} record(s); folding needs depth >= 2")
    return failures


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
    # A tool is required because the config routes to it, not because it happens
    # to be absent. preflight.sh reports facts and is deliberately config-blind;
    # deciding what this run actually needs is this script's job. Every CBLN1 run
    # recorded tools.sc exists:false and nothing read it -- which was harmless
    # then and is moot now that shape complementarity runs in process.
    needed = {"foldseek": "diversity clustering", "mmseqs": "sequence clustering"}
    for tool, why in needed.items():
        entry = tools.get(tool) or {}
        if not entry.get("exists"):
            failures.append(f"missing {tool}, needed for {why}: {entry.get('path') or 'no path configured'}")
    # Shape complementarity used to need an sc binary on a path; it now runs in
    # process through protein-interface, so what a config routing to it requires
    # is an importable extension rather than a file that exists. A manylinux
    # wheel that resolves but will not load fails here rather than hours in --
    # and it would, because the extension is imported inside the functions that
    # use it rather than at module scope, so nothing earlier trips over it.
    failures += hotspot_failures(cfg)
    failures += target_msa_failures(cfg, metric)
    if needs_protein_interface(cfg, metric):
        try:
            from importlib.metadata import version

            import protein_interface  # noqa: F401

            # Reported, not pinned. build_blackwell.sh owns the pin; a second
            # copy of it here is a second thing to forget to update.
            print(f"  protein-interface {version('protein-interface')} imports")
        except Exception as exc:
            failures.append(f"config routes to the interface definition but protein_interface will not import: {exc}")
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
