import os
import logging
from inspect_ai import task
from inspect_ai.dataset import MemoryDataset
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

logger = logging.getLogger(__name__)

def _filter_dataset(task_instance, sample_ids=None):
    """Safely filters an Inspect dataset by matching user-requested indices

    passed via task args or environment variables.
    """
    # Prioritize task arguments over the global environment variable
    ids_source = sample_ids or os.environ.get("HAWK_SAMPLE_IDS")
    
    if ids_source and hasattr(task_instance, "dataset") and task_instance.dataset:
        # Safely extract indices handling integers or raw strings
        target_indices = {
            int(x.strip()) 
            for x in str(ids_source).split(",") 
            if x.strip().isdigit()
        }
        
        if target_indices:
            filtered_samples = []
            
            # Map samples strictly by their absolute row position
            for position, sample in enumerate(task_instance.dataset):
                if position in target_indices:
                    sample.id = position
                    filtered_samples.append(sample)
            
            # Safety fallback check
            if not filtered_samples:
                logger.warning(
                    f"Target indices {list(target_indices)} were out of range. "
                    f"Dataset size is {len(task_instance.dataset)}. Defaulting to first 2 records."
                )
                filtered_samples = list(task_instance.dataset)[:2]
                for i, s in enumerate(filtered_samples):
                    s.id = i
            
            orig_name = getattr(task_instance.dataset, "name", "filtered_dataset")
            orig_location = getattr(task_instance.dataset, "location", None)
            
            task_instance.dataset = MemoryDataset(
                samples=filtered_samples,
                name=orig_name,
                location=orig_location
            )
            
    return task_instance

@task(name="mbpp_samples")
def mbpp_samples(sample_ids=None):  # <--- MUST HAVE sample_ids=None HERE
    """Dynamically filters mbpp samples by index location."""
    return _filter_dataset(mbpp(), sample_ids=sample_ids)

@task(name="gsm8k_samples")
def gsm8k_samples(sample_ids=None):  # <--- MUST HAVE sample_ids=None HERE
    """Dynamically filters gsm8k samples by index location."""
    return _filter_dataset(gsm8k(), sample_ids=sample_ids)
