"""Utilities for per-example loss aggregation and caption quality detection.

NOTE: The core per-example loss calculation depends on model specifics (scheduler, prediction type, etc.) and is expected to be implemented by a caller (e.g. SDTrainer or a dataset evaluator) that can compute per-sample unreduced losses for a batch. These utilities provide aggregation and caption-flagging helpers and a thin wrapper API that accepts a compute_fn.
"""
from typing import List, Dict, Any, Callable, Optional
import math
import statistics


def aggregate_by_dataset(per_example_list: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
    """Aggregate per-example entries into per-dataset statistics.

    per_example_list items should contain at least: 'path', 'dataset', 'caption', 'loss'
    Returns a dict with dataset_summary and global stats.
    """
    dataset_map = {}
    losses_all = []
    for item in per_example_list:
        ds = item.get('dataset', 'unknown') or 'unknown'
        dataset_map.setdefault(ds, []).append(item)
        losses_all.append(item.get('loss', 0.0))

    dataset_summary = {}
    for ds, items in dataset_map.items():
        losses = [x.get('loss', 0.0) for x in items]
        mean = float(sum(losses) / len(losses)) if len(losses) > 0 else 0.0
        median = float(statistics.median(losses)) if len(losses) > 0 else 0.0
        count = len(losses)
        sorted_items = sorted(items, key=lambda x: x.get('loss', 0.0), reverse=True)
        top_k_items = sorted_items[:top_k]
        dataset_summary[ds] = {
            'mean': mean,
            'median': median,
            'count': count,
            'top_k': top_k_items,
        }

    global_summary = {
        'mean': float(sum(losses_all) / len(losses_all)) if len(losses_all) > 0 else 0.0,
        'median': float(statistics.median(losses_all)) if len(losses_all) > 0 else 0.0,
        'count': len(losses_all),
    }

    return {
        'dataset_summary': dataset_summary,
        'global': global_summary,
    }


def _repeated_tokens_score(caption: str) -> float:
    # simple heuristic: fraction of tokens that repeat consecutively
    tokens = caption.split()
    if len(tokens) <= 1:
        return 0.0
    repeats = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeats += 1
    return repeats / len(tokens)


def _suspicious_tokens_score(caption: str, suspicious=None) -> float:
    if suspicious is None:
        suspicious = ['###', '<unk>', 'NULL', 'nan']
    cap = caption.lower()
    score = 0.0
    for s in suspicious:
        if s.lower() in cap:
            score += 1.0
    return score


def flag_bad_captions(per_example_list: List[Dict[str, Any]], heuristics: Optional[Dict[str, Any]] = None, top_n: int = 20) -> Dict[str, Any]:
    """Flag captions using heuristics correlated with bad loss.

    Returns a dict with 'flagged' (list of flagged examples with reason and score) and 'summary' counts.
    """
    if heuristics is None:
        heuristics = {
            'min_tokens': 3,
            'max_tokens': 200,
            'repeated_tokens_threshold': 0.3,
            'suspicious_tokens': ['###', '<unk>', 'NULL', 'nan'],
        }

    flagged = []

    for item in per_example_list:
        caption = (item.get('caption') or '')
        losses = float(item.get('loss', 0.0))
        tokens = caption.split()
        reasons = []
        score = 0.0

        if caption.strip() == '':
            reasons.append('empty_caption')
            score += 3.0

        if len(tokens) < heuristics['min_tokens'] and len(tokens) > 0:
            reasons.append('short_caption')
            score += 1.0

        if len(tokens) > heuristics['max_tokens']:
            reasons.append('long_caption')
            score += 1.0

        repeat_score = _repeated_tokens_score(caption)
        if repeat_score >= heuristics['repeated_tokens_threshold']:
            reasons.append('repeated_tokens')
            score += 1.0 + repeat_score

        susp_score = _suspicious_tokens_score(caption, heuristics.get('suspicious_tokens'))
        if susp_score > 0.0:
            reasons.append('suspicious_tokens')
            score += susp_score

        # optional: add loss-weighted importance; we'll add normalized loss contribution
        # Note: caller may choose to filter by loss percentiles before calling this helper
        try:
            # Guard against non-finite or negative values (e.g., ablation deltas can be negative
            # when captions improve). We only want a positive contribution for "bad" captions.
            if not math.isfinite(losses):
                loss_contrib = 0.0
            else:
                # For ablation-style deltas: negative means caption helped, positive means caption hurt.
                # Use the magnitude of the "bad" direction so we don't penalize good captions.
                loss_contrib = -losses if losses < 0 else losses
            # use log1p for numerical safety and ensure non-negative input
            score += math.log1p(max(0.0, loss_contrib))
        except Exception:
            # Do not let scoring failures stop caption flagging
            pass

        if len(reasons) > 0:
            flagged.append({
                'path': item.get('path', None),
                'dataset': item.get('dataset', None),
                'caption': caption,
                'loss': losses,
                'reasons': reasons,
                'score': score,
            })

    # sort by score (descending) and return top_n
    flagged_sorted = sorted(flagged, key=lambda x: x['score'], reverse=True)[:top_n]

    # build summary
    summary = {}
    for f in flagged_sorted:
        for r in f['reasons']:
            summary[r] = summary.get(r, 0) + 1

    return {
        'flagged': flagged_sorted,
        'summary': summary,
    }


# Convenience wrapper: evaluate_dataset

def per_sample_from_loss_tensor(loss: 'torch.Tensor') -> 'torch.Tensor':
    """Given a loss tensor with batch dimension first, reduce spatial/channel/time dims to a scalar per-sample.

    Supports 4D (B,C,H,W) and 5D (B,C,T,H,W). Returns a CPU detached tensor of shape (B,).
    """
    import torch
    if loss is None:
        return None
    if len(loss.shape) == 5:
        # video: B,C,T,H,W -> mean over C,T,H,W
        out = loss.mean(dim=[1, 2, 3, 4])
    elif len(loss.shape) == 4:
        # image: B,C,H,W -> mean over C,H,W
        out = loss.mean(dim=[1, 2, 3])
    elif len(loss.shape) == 2:
        # already per-sample
        out = loss.mean(dim=1)
    elif len(loss.shape) == 1:
        out = loss
    else:
        # fallback: mean over all dims except first
        dims = list(range(1, len(loss.shape)))
        out = loss.mean(dim=dims)
    return out.detach().cpu()


def compute_per_example_loss(loss_tensor: 'torch.Tensor', prior_loss_tensor: 'torch.Tensor' = None, normalize_batch_mean: bool = False) -> tuple:
    """Compute per-example scalar losses and optional components from unreduced loss tensors.

    Args:
      loss_tensor: unreduced loss (e.g., shape B,C,H,W or B,C,T,H,W or already reduced to B,)
      prior_loss_tensor: optional per-element prior loss (same shape semantics as loss_tensor)
      normalize_batch_mean: if True, return per-sample normalised so that batch mean equals 1 (rarely used)

    Returns:
      (per_sample_tensor_cpu, components_dict)
        - per_sample_tensor_cpu: torch.Tensor on CPU, shape (B,)
        - components_dict: dict with keys like 'main' and 'prior' mapping to per-sample CPU tensors
    """
    import torch
    components = {}

    # main per-sample
    main = per_sample_from_loss_tensor(loss_tensor)
    components['main'] = main

    if prior_loss_tensor is not None:
        prior = per_sample_from_loss_tensor(prior_loss_tensor)
        components['prior'] = prior
        combined = (main.to(torch.float32) + prior.to(torch.float32))
    else:
        combined = main.to(torch.float32)

    if normalize_batch_mean:
        m = float(combined.mean()) if combined.numel() > 0 else 1.0
        if m != 0:
            combined = combined / m

    return combined.detach().cpu(), components


def per_example_from_batch(batch: 'DataLoaderBatchDTO', per_sample_losses: 'torch.Tensor') -> List[Dict[str, Any]]:
    """Build per-example records from a batch and a tensor of per-sample losses.

    Handles cases where per_sample_losses length matches batch size or is doubled (due to doubling for refiner).
    """
    records = []
    if per_sample_losses is None:
        return records
    try:
        import torch
    except Exception:
        torch = None

    losses = per_sample_losses
    # ensure it's an iterable of floats
    if hasattr(losses, 'tolist'):
        losses = losses.tolist()
    # number of file items
    file_items = batch.file_items
    n_files = len(file_items)
    n_losses = len(losses)

    if n_losses == n_files:
        for i, li in enumerate(losses):
            fi = file_items[i]
            records.append({
                'path': fi.path,
                'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) or getattr(fi.dataset_config, 'default_caption', 'unknown'),
                'caption': getattr(fi, 'raw_caption', None) or '',
                'loss': float(li),
            })
    elif n_losses == 2 * n_files:
        # assume duplication: first block then duplicate block
        for i, li in enumerate(losses):
            fi = file_items[i % n_files]
            records.append({
                'path': fi.path,
                'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) or getattr(fi.dataset_config, 'default_caption', 'unknown'),
                'caption': getattr(fi, 'raw_caption', None) or '',
                'loss': float(li),
            })
    else:
        # fallback: pair min(n_files, n_losses)
        m = min(n_files, n_losses)
        for i in range(m):
            fi = file_items[i]
            records.append({
                'path': fi.path,
                'dataset': getattr(fi.dataset_config, 'dataset_path', None) or getattr(fi.dataset_config, 'folder_path', None) or getattr(fi.dataset_config, 'default_caption', 'unknown'),
                'caption': getattr(fi, 'raw_caption', None) or '',
                'loss': float(losses[i]),
            })
    return records


class StreamingAggregator:
    """Streaming aggregator for per-item loss stats (memory-efficient).

    Maintains running count, sum, min, and max for each dataset item and can produce
    a JSON-compatible report of average/min/max per item and dataset summaries.
    """
    def __init__(self):
        # datasets -> path -> stats dict(count, sum, min, max, caption)
        self.datasets = {}

    def add_entry(self, entry: Dict[str, Any]):
        ds = entry.get('dataset', 'unknown') or 'unknown'
        path = entry.get('path', '')
        loss = float(entry.get('loss', 0.0))
        caption = entry.get('caption', '')
        # allow entries to include per-image sample min/max (from repeated stochastic samples)
        entry_min = None
        entry_max = None
        if 'min_loss' in entry and entry.get('min_loss') is not None:
            try:
                entry_min = float(entry.get('min_loss'))
            except Exception:
                entry_min = None
        if 'max_loss' in entry and entry.get('max_loss') is not None:
            try:
                entry_max = float(entry.get('max_loss'))
            except Exception:
                entry_max = None

        if ds not in self.datasets:
            self.datasets[ds] = {}
        items = self.datasets[ds]
        if path not in items:
            items[path] = {
                'count': 0,
                'sum': 0.0,
                'min': entry_min if entry_min is not None else loss,
                'max': entry_max if entry_max is not None else loss,
                'caption': caption,
            }
        st = items[path]
        st['count'] += 1
        st['sum'] += loss
        # incorporate entry-provided min/max if present, otherwise fall back to the loss value
        if entry_min is not None:
            if entry_min < st['min']:
                st['min'] = entry_min
        else:
            if loss < st['min']:
                st['min'] = loss
        if entry_max is not None:
            if entry_max > st['max']:
                st['max'] = entry_max
        else:
            if loss > st['max']:
                st['max'] = loss

    def build_report(self, eval_config: Optional[Dict[str, Any]] = None, top_k: int = 10) -> Dict[str, Any]:
        datasets_report = {}
        for ds, items in self.datasets.items():
            item_stats = {}
            # compute average/min/max
            for path, st in items.items():
                avg = float(st['sum'] / st['count']) if st['count'] > 0 else 0.0
                item_stats[path] = {
                    'average_loss': avg,
                    'min_loss': float(st['min']),
                    'max_loss': float(st['max']),
                    'caption': st.get('caption', ''),
                }

            # compute a simple dataset summary using averages
            means = [v['average_loss'] for v in item_stats.values()] if len(item_stats) > 0 else [0.0]
            mean = float(sum(means) / len(means)) if len(means) > 0 else 0.0
            median = float(sorted(means)[len(means) // 2]) if len(means) > 0 else 0.0
            count = sum([items[p]['count'] for p in items])
            # top_k by average loss
            top_k_items = sorted(
                [{'path': p, 'average_loss': item_stats[p]['average_loss'], 'caption': item_stats[p].get('caption', '')} for p in item_stats],
                key=lambda x: x['average_loss'],
                reverse=True
            )[:top_k]

            datasets_report[ds] = {
                'item_stats': item_stats,
                'summary': {'mean': mean, 'median': median, 'count': count, 'top_k': top_k_items},
            }

        json_report = {
            'config': eval_config or {},
            'datasets': datasets_report,
        }
        return json_report


def evaluate_dataset(compute_fn: Callable[[Any], List[Dict[str, Any]]], dataloader, sample_fraction: float = 1.0, max_samples: Optional[int] = None, out_csv: Optional[str] = None, eval_config: Optional[Dict[str, Any]] = None, out_json: Optional[str] = None, normalize_loss: bool = False) -> Dict[str, Any]:
    """Run compute_fn on the dataset batches to produce per-example list and aggregated results.

    compute_fn is expected to be a callable that accepts a batch and returns a list of per-sample dicts {
      'path', 'dataset', 'caption', 'loss', ...
    }
    This function only orchestrates iteration, sampling and aggregation.

    Optional args:
      - eval_config: dict with keys like 'model' and 'resolution' to include in JSON report
      - out_json: if provided, write a JSON report containing per-item stats and config
    """
    per_example = []
    total = 0
    # We'll build the aggregator after optionally normalizing losses so summaries reflect
    # the post-normalized values. This avoids incorrect per-batch normalization artifacts.
    for batch in dataloader:
        total += 1
        # sampling at batch granularity
        if sample_fraction < 1.0:
            import random
            if random.random() > sample_fraction:
                continue
        batch_entries = compute_fn(batch)
        for e in batch_entries:
            per_example.append(e)
            if max_samples is not None and len(per_example) >= max_samples:
                break
        if max_samples is not None and len(per_example) >= max_samples:
            break

    # Build raw report (pre-normalization) so we can preserve raw numbers for consumers
    raw_report = None
    if out_json is not None and len(per_example) > 0:
        try:
            # Build a raw report whose "loss" values reflect the non-ablated/original
            # per-sample losses when available (e.g., `loss_with_caption`). This ensures
            # `average_loss_raw` reflects the true non-ablated average even when the
            # active `loss` values in `per_example` represent ablation deltas.
            raw_aggregator = StreamingAggregator()
            for e in per_example:
                e_copy = dict(e)
                e_copy['loss'] = e.get('loss_with_caption', e.get('loss', 0.0))
                raw_aggregator.add_entry(e_copy)
            raw_report = raw_aggregator.build_report(eval_config=None)
        except Exception:
            raw_report = None

    # If requested, normalize losses across the entire set so the global mean equals 1.
    if normalize_loss and len(per_example) > 0:
        mean_all = float(sum([e.get('loss', 0.0) for e in per_example]) / len(per_example))
        if mean_all != 0:
            for e in per_example:
                e['loss'] = float(e.get('loss', 0.0) / mean_all)

    results = aggregate_by_dataset(per_example)
    flagged = flag_bad_captions(per_example) if len(per_example) > 0 else {'flagged': [], 'summary': {}}

    if out_csv is not None:
        try:
            import csv
            keys = ['path', 'dataset', 'caption', 'loss']
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for e in per_example:
                    writer.writerow({k: e.get(k, '') for k in keys})
        except Exception:
            # don't fail evaluation on csv write errors; caller can check filesystem
            pass

    # build per-item stats for JSON output
    json_report = None
    if out_json is not None:
        try:
            # create aggregator from post-normalized per_example entries
            aggregator = StreamingAggregator()
            for e in per_example:
                aggregator.add_entry(e)
            json_report = aggregator.build_report(eval_config=eval_config)

            # merge raw per-item average_loss into the json report so UIs can show raw values
            if raw_report is not None and 'datasets' in raw_report and 'datasets' in json_report:
                for ds_name, ds in json_report['datasets'].items():
                    raw_ds = raw_report['datasets'].get(ds_name, {})
                    raw_item_stats = raw_ds.get('item_stats', {})
                    item_stats = ds.get('item_stats', {})
                    for path_key, stats in item_stats.items():
                        raw_stats = raw_item_stats.get(path_key)
                        if raw_stats is not None:
                            stats['average_loss_raw'] = raw_stats.get('average_loss')
                        else:
                            stats['average_loss_raw'] = None

            # Compute average ablation deltas per item if any were recorded in per_example entries.
            try:
                # build sums/counts by path
                ab_sums = {}
                ab_counts = {}
                for e in per_example:
                    path = e.get('path')
                    if path is None:
                        continue
                    if 'ablation_delta' in e and e.get('ablation_delta') is not None:
                        ab_sums[path] = ab_sums.get(path, 0.0) + float(e.get('ablation_delta'))
                        ab_counts[path] = ab_counts.get(path, 0) + 1
                # assign averages into json_report item_stats
                for ds_name, ds in json_report['datasets'].items():
                    item_stats = ds.get('item_stats', {})
                    for path_key, stats in item_stats.items():
                        path = path_key
                        if path in ab_sums and ab_counts.get(path, 0) > 0:
                            stats['average_ablation_delta'] = float(ab_sums[path] / ab_counts[path])
                        else:
                            stats['average_ablation_delta'] = None
            except Exception:
                # don't fail the entire evaluation on ablation aggregation errors
                pass

            import json
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2)
        except Exception:
            json_report = None

    return {
        'per_example': per_example,
        'aggregates': results,
        'flagged': flagged,
        'json_report': json_report,
    }


def run_dataset_evaluation(compute_fn: Callable[[Any], List[Dict[str, Any]]], dataloader, job_name: Optional[str] = None, step: Optional[int] = None, out_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper to run dataset evaluation and write a job/step-named JSON report.

    If `out_dir`, `job_name`, and `step` are provided, this function will create a JSON file named
    `{job_name}_{step_zfilled}.json` inside `out_dir` using the streaming JSON writer in `evaluate_dataset`.

    Returns the same dict structure returned by `evaluate_dataset`.
    """
    import os
    # Only set a job-based out_json when the caller did NOT provide an explicit out_json.
    if out_dir is not None and job_name is not None and step is not None and 'out_json' not in kwargs:
        filename = f"{job_name}_{str(step).zfill(9)}.json"
        kwargs['out_json'] = os.path.join(out_dir, filename)
    # evaluate_dataset will write out_json if present
    return evaluate_dataset(compute_fn, dataloader, **kwargs)
