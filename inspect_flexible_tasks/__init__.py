import os
import logging
from inspect_ai import task
from inspect_ai.dataset import MemoryDataset
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

logger = logging.getLogger(__name__)

def _filter_dataset(task_instance):
    """Safely filters an Inspect dataset by matching user-requested indices 

    against absolute row positions.
    """
    sample_ids_raw = os.environ.get("HAWK_SAMPLE_IDS")
    if sample_ids_raw and hasattr(task_instance, "dataset") and task_instance.dataset:
        # Convert string list "0,10,13" into a clean set of integers: {0, 10, 13}
        target_indices = {
            int(x.strip()) 
            for x in sample_ids_raw.split(",") 
            if x.strip().isdigit()
        }
        
        if target_indices:
            filtered_samples = []
            
            # Walk through the dataset and strictly use its absolute row position
            for position, sample in enumerate(task_instance.dataset):
                if position in target_indices:
                    # Explicitly override or stamp the sample id attribute so Inspect logs it correctly
                    sample.id = position
                    filtered_samples.append(sample)
            
            # Fallback boundary check: if the user requested indices out of range
            if not filtered_samples:
                logger.warning(
                    f"HAWK_SAMPLE_IDS {list(target_indices)} were out of range for this split. "
                    f"Dataset size is {len(task_instance.dataset)}. Defaulting to first 2 records."
                )
                filtered_samples = list(task_instance.dataset)[:2]
                for i, s in enumerate(filtered_samples):
                    s.id = i
            
            # Pack it back into the Inspect container safely
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
    """Dynamically filters mbpp samples by index location."""
    return _filter_dataset(mbpp())

@task(name="gsm8k_samples")
def gsm8k_samples():
    """Dynamically filters gsm8k samples by index location."""
    return _filter_dataset(gsm8k())
