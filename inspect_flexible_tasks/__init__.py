import os
import logging
from inspect_ai import task
from inspect_ai.dataset import MemoryDataset
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

logger = logging.getLogger(__name__)

def _filter_dataset(task_instance):
    """Helper to safely filter an Inspect dataset without emptying it completely."""
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw and hasattr(task_instance, "dataset") and task_instance.dataset:
        target_ids = [
            int(x.strip()) 
            for x in sample_ids_raw.split(",") 
            if x.strip().isdigit()
        ]
        
        if target_ids:
            filtered_samples = []
            
            # Enumerate to fall back on item index if sample.id is missing/None
            for index, sample in enumerate(task_instance.dataset):
                sample_id = getattr(sample, "id", None)
                if sample_id is None:
                    sample_id = index  # Fallback to absolute index position
                
                # Check match against both integer and string variants
                if sample_id in target_ids or str(sample_id) in [str(t) for t in target_ids]:
                    filtered_samples.append(sample)
            
            # Prevent ZeroDivisionError if no matches were found
            if not filtered_samples:
                logger.warning(f"HAWK_SAMPLE_IDS {target_ids} matched 0 samples. Defaulting to first 2 records.")
                filtered_samples = list(task_instance.dataset)[:2]
            
            orig_name = getattr(task_instance.dataset, "name", "filtered_dataset")
            orig_location = getattr(task_instance.dataset, "location", None)
            
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
