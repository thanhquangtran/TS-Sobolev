# main.py
from pathlib import Path
import json, os, random
import torch

from octis.dataset.dataset import Dataset

# Model
from octis.models.S2WTM import S2WTM
from octis.models.SWTM import SWTM

# Topic-quality metrics
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import (
    TopicDiversity, InvertedRBO, WordEmbeddingsInvertedRBOCentroid
)

# Downstream metrics (import * so we can support multiple naming variants across OCTIS versions)
from octis.evaluation_metrics.classification_metrics import *

# OCTIS save/load for model OUTPUT (topics + matrices)
from octis.models.model import save_model_output, load_model_output

# Local helpers
from config import get_config
import log_utils  # << safe (does not shadow stdlib logging)


# -----------------------------
# Dataset loader (unchanged API)
# -----------------------------
def get_dataset(dataset_name: str, data_dir: Path) -> Dataset:
    data = Dataset()
    if dataset_name == 'M10':
        data.fetch_dataset("M10", data_home='.')
    else:
        raise ValueError('Missing or unknown dataset name.')
    return data


def infer_num_topics(data: Dataset, k_override: int | None) -> int:
    if k_override is not None:
        return k_override
    labels = data.get_labels()
    if not labels:
        raise RuntimeError("Dataset has no labels; please pass --k explicitly.")
    return len(set(labels))


def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_load_output(run_dir: Path, logger):
    op = run_dir / "model_output.json"
    if op.exists():
        try:
            output = load_model_output(str(op))
            logger.info(f"Loaded existing OCTIS model output from {op}")
            return output
        except Exception as e:
            logger.warning(f"Found model_output.json but failed to load: {e}")
    return None


def save_output(output: dict, run_dir: Path, approx_order: int, logger):
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "model_output.json"
    save_model_output(output, path=str(out_path), appr_order=approx_order)
    logger.info(f"Saved OCTIS model output to {out_path}")


def try_save_weights(model, run_dir: Path, logger):
    # Best-effort; portable artifact remains model_output.json via OCTIS utilities.
    try:
        if hasattr(model, "save") and callable(model.save):
            ckpt_dir = run_dir / "checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt_dir))
            logger.info(f"Saved model checkpoint to {ckpt_dir}")
            return
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), run_dir / "model.pt")
            logger.info(f"Saved torch state_dict to {run_dir / 'model.pt'}")
    except Exception as e:
        logger.warning(f"Saving weights skipped (not supported by this implementation): {e}")


# ---------- Metrics ----------
def compute_topic_quality_metrics(
    data: Dataset,
    output: dict,
    topk: int,
    npmi_m: str,
    cv_m: str,
    processes: int,
    irbo_weight: float,
    use_wic: bool,
    w2v_path,
    logger
):
    texts = data.get_corpus()
    npmi = Coherence(texts=texts, topk=topk, processes=processes, measure=npmi_m).score(output)
    cv = Coherence(texts=texts, topk=topk, processes=processes, measure=cv_m).score(output)
    td = TopicDiversity(topk=topk).score(output)
    irbo = InvertedRBO(topk=topk, weight=irbo_weight).score(output)

    results = {"NPMI": float(npmi), "CV": float(cv), "TD": float(td), "irbo": float(irbo)}

    if use_wic:
        try:
            wIC = WordEmbeddingsInvertedRBOCentroid(
                topk=topk,
                weight=irbo_weight,
                word2vec_path=str(w2v_path) if w2v_path else None,
                binary=True
            )
            results["wIC"] = float(wIC.score(output))
        except Exception as e:
            logger.warning(f"wIC metric failed (skipping): {e}")

    return results

# -----------------------------
# Main
# -----------------------------
def main():
    cfg = get_config()

    # Logger (file + console)
    logger = log_utils.create_file_logger(cfg.log_dir, name="train")

    # Seeds
    if cfg.seeds is None:
        from random import randint
        seeds = [randint(0, int(2e3))]
    else:
        seeds = cfg.seeds

    # Processes for coherence metrics
    processes = cfg.processes if cfg.processes and cfg.processes > 0 else max(1, (os.cpu_count() or 2) - 1)

    all_rows = []

    for seed in seeds:
        set_seeds(seed)
        for dname in cfg.datasets:
            logger.info("-" * 80)
            logger.info(f"Dataset: {dname} | Seed: {seed} | downstream={cfg.downstream}")

            data = get_dataset(dname, cfg.data_dir)
            k = infer_num_topics(data, cfg.k)
            logger.info(f"Number of topics (K): {k}")

            run_dir = Path(cfg.output_dir) / dname / cfg.loss_type / str(cfg.num_projections) / str(cfg.n_trees) /  f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            output = None
            if cfg.resume:
                output = maybe_load_output(run_dir, logger)

            if output is None:
                # Train
                if cfg.loss_type in ['sbstsw']:
                    model = S2WTM(
                        dropout=cfg.dropout,
                        batch_size=cfg.batch_size,
                        dist=cfg.dist,
                        beta=cfg.beta,
                        num_topics=k,
                        num_epochs=cfg.epochs,
                        num_projections=cfg.num_projections,
                        use_partitions=cfg.downstream,
                        loss_type=cfg.loss_type,
                        n_trees=cfg.n_trees,
                        delta=cfg.delta,
                        p=cfg.p,
                    )
                    logger.info("Training S2WTM...")
                else:
                    model = SWTM(
                        dropout=cfg.dropout,
                        batch_size=cfg.batch_size,
                        dist=cfg.dist,
                        beta=cfg.beta,
                        num_topics=k,
                        num_epochs=cfg.epochs,
                        num_projections=cfg.num_projections,
                        use_partitions=cfg.downstream,
                        loss_type=cfg.loss_type,
                        n_trees=cfg.n_trees,
                        delta=cfg.delta,
                        p=cfg.p,
                    )

                    logger.info("Training SWTM...")
                output = model.train_model(dataset=data)

                # Save outputs + best-effort weights
                save_output(output, run_dir, cfg.approx_order, logger)
                try_save_weights(model, run_dir, logger)

                del model
                torch.cuda.empty_cache()
            else:
                logger.info("Resumed from saved output (skipping training).")

            # Metrics
            metrics = compute_topic_quality_metrics(
                data=data,
                output=output,
                topk=cfg.topk,
                npmi_m=cfg.npmi_measure,
                cv_m=cfg.cv_measure,
                processes=processes,
                irbo_weight=cfg.irbo_weight,
                use_wic=cfg.wic,
                w2v_path=cfg.word2vec_path,
                logger=logger
            )

            row = {"Dataset": dname, "K": k, "Seed": seed, **metrics}
            all_rows.append(row)

            # Per-run metrics.json
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrics: {row}")


if __name__ == "__main__":
    main()
