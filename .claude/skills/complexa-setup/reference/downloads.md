# `complexa download` Reference

Full flag matrix, destination layout, and NGC source for every checkpoint
Complexa can pull.

`complexa download` is a thin Python wrapper around `env/download_startup.sh`
— the CLI argparse (in `src/proteinfoundation/cli/cli_runner.py:download_main`)
forwards `sys.argv[2:]` straight to the bash script, so any flag the bash
script accepts is reachable.

---

## Flag matrix

The Python argparse explicitly defines these flags:

| Flag | What it downloads | Destination | Approx size | Source |
|------|-------------------|-------------|-------------|--------|
| `--complexa` | `complexa.ckpt` + `complexa_ae.ckpt` (protein binder) | `./ckpts/` | ~3 GB | NGC `proteina_complexa` |
| `--complexa-ligand` | `complexa_ligand.ckpt` + `complexa_ligand_ae.ckpt` (ligand binder) | `./ckpts/` | ~3 GB | NGC `proteina_complexa_ligand` |
| `--complexa-ame` | `complexa_ame.ckpt` + `complexa_ame_ae.ckpt` (motif scaffolding) | `./ckpts/` | ~3 GB | NGC `proteina_complexa_ame` |
| `--complexa-all` | All three Complexa pairs | `./ckpts/` | ~9 GB | NGC (3 models) |
| `--all` | ProteinMPNN + LigandMPNN + AF2 + ESM2 + RF3 (exactly these 5 — no ESMFold) | `./community_models/...` | ≈10.7 GB | Mixed (GitHub + AWS + HF + NGC) |
| `--everything` | The 3 Complexa pairs + the same 5 community models. No Boltz2, no Protenix. | both | ≈20 GB | Mixed |
| `--status` | Show install state; downloads nothing | (none) | n/a | n/a |

The underlying bash script also accepts per-model flags (`--pmpnn`,
`--ligmpnn`, `--af2`, `--esm2`, `--rf3`) that pass through unchanged. All five
are listed in `env/download_startup.sh:show_help` (`:776-805`) but not in the
Python argparse — they still work because of the `sys.argv[2:]` passthrough.

That list is exhaustive. Anything else — including `--esmfold` and `--boltz2` —
falls through to the `*)` branch, which prints `Unknown option: <arg>`, dumps the
help, and exits 1. ESMFold has no downloader in this script at all; use
`python script_utils/download/download_esmfold_model.py`. Boltz2 and Protenix
appear nowhere in the repo's download tooling.

| Passthrough flag | Destination | Approx size |
|------------------|-------------|-------------|
| `--pmpnn` | `./community_models/ProteinMPNN/{ca,vanilla}_model_weights/` | ~50 MB |
| `--ligmpnn` | `./community_models/LigandMPNN/model_params/` | ~500 MB |
| `--af2` | `./community_models/ckpts/AF2/` (tar extracted in place — no `params/` subdir) | ~5 GB |
| `--esm2` | `./community_models/ckpts/ESM2/` | ~2.5 GB |
| `--rf3` | `./community_models/ckpts/RF3/` | ~2.5 GB |

---

## Default destinations

All paths below are relative to `PROJECT_ROOT` (derived inside
`download_startup.sh`), **not** to your shell's CWD — `main()` does
`cd "$PROJECT_ROOT"` (`:817`) before parsing any flag, so where you invoke
`complexa download` from makes no difference.

```
$PROJECT_ROOT/
├── ckpts/                                   ← all 6 Complexa ckpts here, flat
│   ├── complexa.ckpt                        ← protein binder model
│   ├── complexa_ae.ckpt                     ← protein binder autoencoder
│   ├── complexa_ligand.ckpt                 ← ligand binder model
│   ├── complexa_ligand_ae.ckpt              ← ligand binder autoencoder
│   ├── complexa_ame.ckpt                    ← AME motif scaffolding model
│   └── complexa_ame_ae.ckpt                 ← AME autoencoder
└── community_models/
    ├── ProteinMPNN/
    │   ├── ca_model_weights/                ← Cα-only weights
    │   └── vanilla_model_weights/           ← full-backbone weights
    ├── LigandMPNN/model_params/             ← LigandMPNN weights
    └── ckpts/
        ├── AF2/                             ← AlphaFold2 params_model_*.npz, flat
        ├── ESM2/                            ← ESM2-650M weights
        └── RF3/                             ← RoseTTAFold3 ckpt
```

**Mismatch to fix by hand:** `LOCAL_CHECKPOINT_PATH` defaults to
`${LOCAL_CODE_PATH}/checkpoints` (`.env_example:28`) — a *different* directory
from where `complexa download` writes, which is always `$PROJECT_ROOT/ckpts`
(`download_startup.sh:237-239`) regardless of the `.env` setting. They do **not**
line up out of the box, so `${oc.env:CKPT_PATH}` resolves to an empty
`checkpoints/` even after a successful download. Either set
`LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts` after downloading, or
move/symlink `./ckpts/*` into `checkpoints/`.

---

## The three Complexa model variants

| Variant | Pipeline config | Required ckpt pair | Use case |
|---------|----------------|---------------------|----------|
| Protein binder | `configs/search_binder_local_pipeline.yaml` | `complexa.ckpt` + `complexa_ae.ckpt` | De novo binders for protein targets (PDL1, EGFR, etc.) |
| Ligand binder | `configs/search_ligand_binder_local_pipeline.yaml` | `complexa_ligand.ckpt` + `complexa_ligand_ae.ckpt` | Binders to small-molecule pockets (FAD, OQO, etc.) |
| AME | `configs/search_ame_local_pipeline.yaml` | `complexa_ame.ckpt` + `complexa_ame_ae.ckpt` | Scaffolding catalytic / functional motifs with ligand context |

Each pipeline YAML has three checkpoint fields at the top level that you
**must** point at your local ckpts. After `complexa download --complexa-all`
they will be at `./ckpts/complexa{,_ae,_ligand,_ligand_ae,_ame,_ame_ae}.ckpt`:

```yaml
ckpt_path: ./ckpts
ckpt_name: complexa.ckpt
autoencoder_ckpt_path: ./ckpts/complexa_ae.ckpt
```

Or override on the CLI without editing the YAML:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++ckpt_path=./ckpts \
    ++ckpt_name=complexa.ckpt \
    ++autoencoder_ckpt_path=./ckpts/complexa_ae.ckpt
```

> Note: `complexa download` always uses a *flat* `./ckpts/` layout. If you
> downloaded ckpts manually into per-variant subdirectories
> (`ckpts/complexa_protein/`, etc.), update the pipeline YAML `ckpt_path` to
> match — or move/symlink into the flat layout.

---

## `complexa download --status` output

Example output (post `--complexa-all` + `--af2`, no MPNNs, ESM2, or RF3):

```
  Current Installation Status
  Complexa Models (Required):
    Complexa (Protein): ✓ Installed (ckpts/)
    Complexa (Ligand):  ✓ Installed (ckpts/)
    Complexa (AME):     ✓ Installed (ckpts/)

  Core Models:
    ProteinMPNN:     ○ Missing (community_models/ProteinMPNN/):
      ✗ ca_model_weights/v_48_002.pt
      ✗ vanilla_model_weights/v_48_002.pt
    LigandMPNN:      ○ Missing (community_models/LigandMPNN/model_params/):
      ✗ proteinmpnn_v_48_002.pt
    AlphaFold2:      ✓ Installed (community_models/ckpts/AF2/)
    ESM2:            ○ Not installed (community_models/ckpts/ESM2/)
    RF3:             ○ Missing (community_models/ckpts/RF3/):
      ✗ rf3_foundry_01_24_latest_remapped.ckpt
```

There are exactly two groups (`Complexa Models (Required):` and `Core Models:`)
and exactly eight rows — no `ESMFold:` or `Boltz2:` row exists. An installed row
is a single `✓ Installed (<dir>)`; a missing row is a `○ Missing (<dir>):` header
followed by one indented `✗ <filename>` line per absent file (`:607-614`), so the
missing lists above are truncated for brevity. ESM2 is the exception: its check is
a directory-size heuristic and prints `○ Not installed (…)` with no file list.
Re-run `complexa download --<flag>` for anything not installed.

---

## Tips

- Run `complexa download --status` **before** any download — it shows what is already on disk and saves re-downloading.
- Re-running `complexa download --<flag>` is idempotent: existing non-empty ckpts are skipped (`download_complexa_weights` checks `[ -f "$fm_ckpt" ] && [ -s "$fm_ckpt" ]`).
- **Failed downloads are *not* cleaned up.** The only `rm` in the script (`:228`) removes the AF2 tar *after* a successful extract. A truncated `.ckpt` is left in place, and because the skip check is `[ -f "$f" ] && [ -s "$f" ]` (`:245`) — file exists and is non-empty — a retry silently treats the partial file as installed. Delete the offending file yourself before re-running.
- For ESM2: set `HF_TOKEN` in `.env` before downloading to avoid HF Hub rate limits. (ESMFold is not downloadable through this script — see the flag matrix above.)
- `complexa download --everything` fetches every model the script supports (≈20 GB: 3 Complexa pairs + 5 community models). Prefer the targeted flags unless you actually need all variants.
