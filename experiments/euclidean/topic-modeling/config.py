# config.py
# CLI config that returns an argparse.Namespace (no classes).

import argparse
from pathlib import Path

def get_config(argv=None):
    p = argparse.ArgumentParser(
        description="Train S2WTM (OCTIS) with optional downstream (classification) evaluation."
    )

    # Data
    p.add_argument("--data_dir", type=Path, default=Path("./preprocessed_datasets"),
                   help="Root folder for custom datasets.")
    p.add_argument("--datasets", type=str, default="20NG",
                   help="Comma-separated dataset names. 20NG,BBC,M10,SearchSnippets,Pascal_Flickr,Bio,DBLP")
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated integer seeds, e.g. 0,1,2. If omitted, a single random seed is used.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs per run.")
    p.add_argument("--k", type=int, default=None,
                   help="Override number of topics; if not set, inferred from labels.")

    # S2WTM hyperparameters
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--dist", type=str, default="vmf",
                   help="Distribution name accepted by S2WTM (e.g., 'vMF').")
    p.add_argument("--beta", type=float, default=8.526)
    p.add_argument("--num_projections", type=int, default=4000)
    p.add_argument("--n_trees", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--p", type=float, default=2)
    p.add_argument("--loss_type", type=str, default='stsw', help='sph_sw, stsw, sbtsw, circular_r0, dbtsw, sw, dsw, rpsw, ebprsw, isebsw')

    # Metrics (topic quality mode)
    p.add_argument("--topk", type=int, default=10, help="Top-k words for coherence/diversity metrics.")
    p.add_argument("--npmi_measure", type=str, default="c_npmi")
    p.add_argument("--cv_measure", type=str, default="c_v")
    p.add_argument("--processes", type=int, default=4,
                   help="Parallel workers for coherence metrics; 0 -> auto(cpu_count-1).")
    p.add_argument("--no_wic", dest="wic", action="store_false", default=True,
                   help="Disable WordEmbeddingsInvertedRBOCentroid.")
    p.add_argument("--word2vec_path", type=Path, default=None,
                   help="Local path to word2vec (to avoid large download).")
    p.add_argument("--irbo_weight", type=float, default=0.9, help="RBO weight parameter for IRBO metrics.")

    # Downstream (classification) switch
    p.add_argument("--downstream", action="store_true", default=False,
                   help="If set: train with validation + partitions and evaluate NMI/Purity/Accuracy/F1.")

    # I/O & logging
    p.add_argument("--output_dir", type=Path, default=Path("./artifacts"),
                   help="Where to save per-run outputs.")
    p.add_argument("--log_dir", type=Path, default=Path("./logs"),
                   help="Where to write logs.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume if model_output exists (default: on).")
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="Force retrain even if outputs exist.")
    p.add_argument("--approx_order", type=int, default=7,
                   help="Rounding order for OCTIS save_model_output.")


    args = p.parse_args(argv)

    # Normalize comma-lists
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.seeds is not None:
        args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    return args
