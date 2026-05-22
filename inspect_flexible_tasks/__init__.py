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
            for x in [x.strip()] 
            if x.isdigit()
        ]
        if target_ids:
            # Filter the dataset records by their id property
            task_instance.dataset = [
                sample for sample in task_instance.dataset 
                if getattr(sample, "id", None) in target_ids or sample.get("id") in target_ids
            ]
    return task_instance

@task(name="inspect_flexible_tasks/mbpp_samples")
def mbpp_samples():
    """Dynamically filters mbpp samples."""
    # Generate a fresh instance of the baseline task
    base_task = mbpp()
    return _filter_dataset(base_task)

@task(name="inspect_flexible_tasks/gsm8k_samples")
def gsm8k_samples():
    """Dynamically filters gsm8k samples."""
    # Generate a fresh instance of the baseline task
    base_task = gsm8k()
    return _filter_dataset(base_task)
