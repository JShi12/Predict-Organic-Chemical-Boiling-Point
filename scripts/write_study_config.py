"""Write the representation study's configuration to one file (Step 9).

Values are read from the code that produced the results (module constants
and function defaults), so the file cannot drift from it; the fine-tuning
seeds (0, 1, 2) repeat a literal in scripts/run_finetuning.py. The split settings
stay in results/representation_study/splits_config.json.

Writes results/representation_study/config.json.

    python scripts/write_study_config.py
"""
import inspect
import json
import sys

sys.path.insert(0, "scripts")

from boiling_point import representations as rep, study, study_stats  # noqa: E402  (xgboost before torch)
from boiling_point import finetune, splits  # noqa: E402
import run_combined  # noqa: E402
import run_finetune_readout  # noqa: E402
import run_head_control  # noqa: E402
import run_learning_curves  # noqa: E402
import run_uncertainty  # noqa: E402

OUT = "results/representation_study/config.json"


def default(func, name):
    return inspect.signature(func).parameters[name].default


def main():
    config = {
        "molformer": {"model_id": rep.MOLFORMER_ID, "revision": rep.MOLFORMER_REVISION,
                      "code_sha256": rep.MOLFORMER_CODE_SHA256, "pooling": rep.POOLING, "layers": rep.LAYER_NAMES,
                      "smiles_prep": rep.SMILES_PREP_VERSION},
        "morgan": {"radius": rep.MORGAN_RADIUS, "n_bits": rep.MORGAN_BITS, "type": "counts"},
        "frozen_comparison": {"ridge_alphas": study.ALPHAS, "xgboost_grid": study.XGB_PARAM_GRID,
                              "inner_folds": study.N_INNER, "selection_metric": "inner-CV MAE"},
        "fine_tuning": {"learning_rates": list(finetune.LEARNING_RATES), "batch_size": finetune.BATCH_SIZE,
                        "max_epochs": finetune.MAX_EPOCHS, "patience": finetune.PATIENCE,
                        "weight_decay": finetune.WEIGHT_DECAY, "gradient_clip": finetune.CLIP,
                        "n_spellings_at_inference": finetune.N_TTA, "seeds": [0, 1, 2],
                        "validation_fraction": default(splits.validation_split, "fraction"),
                        "validation_seed": default(splits.validation_split, "seed")},
        "statistics": {"bootstrap_draws": default(study_stats.bootstrap_macro_difference, "n_boot"),
                       "bootstrap_seed": default(study_stats.bootstrap_macro_difference, "seed"),
                       "alpha": default(study_stats.finding, "alpha"),
                       "minimum_meaningful_difference_K": default(study_stats.finding, "min_difference"),
                       "multiple_comparisons": "Holm (1979) across C1-C5"},
        "learning_curve_sizes": list(run_learning_curves.SIZES),
        "prediction_interval_miscoverage": run_uncertainty.ALPHA,
        "exploratory": {"stacking_variants": run_combined.STACKS,
                        "head_control_recipes": {k: {**v, "lrs": list(v["lrs"])}
                                                 for k, v in run_head_control.RECIPES.items()},
                        "early_layer_fine_tuning": run_finetune_readout.VARIANTS},
    }
    with open(OUT, "w") as f:
        json.dump(config, f, indent=1)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
