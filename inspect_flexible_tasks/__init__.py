import os
from inspect_ai import task
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

def _filter_dataset(task_instance):
    """Helper to inject sample filtering into a task's dataset cleanly."""
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw and hasattr(task_instance, "dataset"):
        target_ids = [
            int(x.strip()) 
            for x in sample_ids_raw.split(",") 
            if x.strip().isdigit()
        ]
        if target_ids:
            # Filter standard inspect-evals datasets by sample ID
            task_instance.dataset = [
                sample for sample in task_instance.dataset 
                if getattr(sample, "id", None) in target_ids or sample.get("id") in target_ids
            ]
    return task_instance

@task(name="mbpp_samples")
def mbpp_samples():
    """Dynamically filters mbpp samples."""
    return _filter_dataset(mbpp())

@task(name="gsm8k_samples")
def gsm8k_samples():
    """Dynamically filters gsm8k samples."""
    return _filter_dataset(gsm8k())
