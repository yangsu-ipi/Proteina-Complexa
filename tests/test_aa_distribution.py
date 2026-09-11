"""Amino-acid composition of a binder, and of its interface.

Two defects motivated this. The reported distribution counted every chain, so on
CBLN1 -- a 136-residue target against a ~42-residue binder -- 76% of every count
came from a molecule identical in all 340 designs. The published proportions
matched the whole-complex composition exactly and diluted the binder's own signal
about four-fold (leucine 14.8% read as 9.0%). And the interface counts the
evaluation stage had been emitting all along reached no output at all.

The enrichment ratio is the number worth reading. Interface proportions largely
track the binder's own composition, so dividing by it is what separates "there is
a lot of leucine at the interface" from "the designer put leucine there".
"""

import math

from proteinfoundation.result_analysis.analysis_utils import aa_distribution_row

AA = list("ARNDCQEGHILKMFPSTWYV")


def counts(**by_aa):
    return [float(by_aa.get(a, 0)) for a in AA]


def test_proportions_and_enrichment_are_what_they_claim():
    row = aa_distribution_row(
        [counts(L=6, K=2, A=2)],       # binder: 60% L, 20% K, 20% A
        [counts(L=3, K=1)],            # interface: 75% L, 25% K
        AA,
    )
    assert math.isclose(row["aa_prop_L"], 0.6)
    assert math.isclose(row["aa_interface_prop_L"], 0.75)
    assert math.isclose(row["aa_interface_enrichment_L"], 1.25)
    assert math.isclose(row["aa_interface_enrichment_K"], 1.25)
    assert row["aa_interface_enrichment_A"] == 0.0, "present in the binder, absent at the interface"


def test_proportions_sum_to_one():
    row = aa_distribution_row([counts(L=3, E=5, W=2)], [counts(L=1, E=1)], AA)
    assert math.isclose(sum(row[f"aa_prop_{a}"] for a in AA), 1.0)
    assert math.isclose(sum(row[f"aa_interface_prop_{a}"] for a in AA), 1.0)


def test_an_amino_acid_absent_from_the_binder_has_no_enrichment():
    """Not zero and not infinity. SolubleMPNN omits cysteine outright, so every
    mpnn redesign has none -- measured across 340 production designs -- and a
    ratio to zero is not an enrichment of anything."""
    row = aa_distribution_row([counts(L=10)], [counts(L=4)], AA)
    assert math.isnan(row["aa_interface_enrichment_C"])
    assert row["aa_prop_C"] == 0.0, "the proportion is still honestly zero"


def test_a_vector_of_the_wrong_length_is_dropped_not_padded():
    """Padding would read as a real absence for every amino acid it fails to
    reach, which is indistinguishable from a measurement."""
    good, short = counts(L=4), [1.0, 2.0]
    both = aa_distribution_row([good, short], [good, short], AA)
    only_good = aa_distribution_row([good], [good], AA)
    assert both.keys() == only_good.keys()
    # NaN never equals itself, so compare it as the category it is.
    for key, value in both.items():
        other = only_good[key]
        assert (math.isnan(value) and math.isnan(other)) or value == other, key


def test_no_designs_is_not_a_composition_of_zeros_dividing_by_zero():
    row = aa_distribution_row([], [], AA)
    assert all(row[f"aa_prop_{a}"] == 0.0 for a in AA)
    assert all(math.isnan(row[f"aa_interface_enrichment_{a}"]) for a in AA)


def test_counts_are_summed_across_designs_not_averaged():
    """A campaign's composition is over all its residues; averaging per-design
    proportions would weight a 40-residue binder like a 70-residue one."""
    row = aa_distribution_row([counts(L=9, K=1), counts(L=1, K=1)], [counts(L=1)], AA)
    assert math.isclose(row["aa_prop_L"], 10 / 12)


def test_analyze_emits_the_interface_family_and_counts_the_binder_alone():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/proteinfoundation/analyze.py").read_text()
    assert "def compute_interface_aa_distribution(" in source
    assert "aa_distribution_row(" in source, "the analyze wrapper uses the tested core"
    assert "res_aa_interface_distribution_" in source
    # and the whole-binder view no longer counts the target alongside it
    assert "_count_residues_from_pdb(path, chains=_binder_chains(path))" in source


def test_the_composition_is_one_vector_per_redesign():
    """ProteinMPNN changes the sequence, so the composition changes with it.
    Emitting aa_stats[0] reported the first redesign's vector for the row."""
    from proteinfoundation.evaluation.binder_eval import packed_aa_counts

    first = packed_aa_counts({"L": 5, "A": 2})
    second = packed_aa_counts({"L": 1, "A": 6})
    assert first != second
    assert len(first) == len(AA), "OpenFold residue order, which is the contract"
    assert sum(first) == 7


def test_the_headline_composition_is_a_vector_not_one_count_of_twenty():
    """The shape bug hiding inside the rename. {seq}_aa_counts_all held the twenty
    counts rather than a list over redesigns, so once the headline moved to
    analyze -- where the rule is X = X_all[best_idx] -- the scalar became count
    number best_idx, an integer where every consumer unpacks a vector."""
    import pandas as pd

    from proteinfoundation.evaluation.binder_eval_utils import DEFAULT_PROTEIN_RANKING_CRITERIA
    from proteinfoundation.result_analysis.binder_analysis import pick_headline_sequence

    first, second = counts(L=5, A=2), counts(L=1, A=6)
    df = pd.DataFrame(
        [
            {
                "complex_folding_backend": "af2",
                "self_complex_af2_i_pAE_all": [9.0, 1.0],
                "self_aa_counts_all": [first, second],
            }
        ]
    )
    out = pick_headline_sequence(df, ["self"], DEFAULT_PROTEIN_RANKING_CRITERIA)

    assert out.at[0, "self_best_idx"] == 1, "the second redesign ranks better"
    assert out.at[0, "self_aa_counts"] == second, "a whole vector, and the ranked one"
