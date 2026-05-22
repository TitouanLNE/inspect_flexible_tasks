import os
from inspect_ai import task, task_with
from inspect_evals.mbpp import mbpp
from inspect_evals.class_eval import class_eval
from inspect_evals.gsm8k import gsm8k
from .gsm8k_samples import gsm8k_samples

@task(name="inspect_flexible_tasks/mbpp_samples")
def mbpp_samples():
    """Dynamically filters mbpp samples."""
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw:
        target_ids = [int(x.strip()) for x in sample_ids_raw.split(",") if x.strip().isdigit()]
        return task_with(mbpp(), sample_id=target_ids)
    return mbpp()

@task(name="inspect_flexible_tasks/gsm8k_samples")
def gsm8k_samples():
    """Dynamically filters gsm8k samples."""
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw:
        target_ids = [int(x.strip()) for x in sample_ids_raw.split(",") if x.strip().isdigit()]
        return task_with(gsm8k(), sample_id=target_ids)
    return gsm8k()
