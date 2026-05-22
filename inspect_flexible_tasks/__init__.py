import os
from inspect_ai import task
from inspect_ai.dataset import MemoryDataset  # Native dataset container wrapper
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

def _filter_dataset(task_instance):
    """Helper to safely filter an Inspect task dataset while maintaining framework attributes."""
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw and hasattr(task_instance, "dataset") and task_instance.dataset:
        target_ids = [
            int(x.strip()) 
            for x in sample_ids_raw.split(",") 
            if x.strip().isdigit()
        ]
        
        if target_ids:
            # 1. Filter the internal sample list
            filtered_samples = [
                sample for sample in task_instance.dataset 
                if getattr(sample, "id", None) in target_ids
            ]
            
            # 2. Retain original dataset metadata fields safely
            orig_name = getattr(task_instance.dataset, "name", "filtered_dataset")
            orig_location = getattr(task_instance.dataset, "location", None)
            
            # 3. Rewrap back into an Inspect-recognized dataset container
            task_instance.dataset = MemoryDataset(
                samples=filtered_samples,
                name=orig_name,
                location=orig_location
            )
            
    return task_instance

@task(name="mbpp_samples")
def mbpp_samples():
    """Dynamically filters mbpp samples."""
    return _filter_dataset(mbpp())

@task(name="gsm8k_samples")
def gsm8k_samples():
    """Dynamically filters gsm8k samples."""
    return _filter_dataset(gsm8k())
