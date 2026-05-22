import os
import logging
from inspect_ai import task
from inspect_evals.mbpp import mbpp
from inspect_evals.gsm8k import gsm8k

logger = logging.getLogger(__name__)

def _safe_apply_filter(task_instance, sample_ids=None):
    """Safely filters an evaluation dataset using Inspect's native slice logic

    to prevent losing critical metadata like sandbox rules.
    """
    # Prioritize task args over the fallback global env variable
    ids_source = sample_ids or os.environ.get("HAWK_SAMPLE_IDS")
    
    if ids_source and hasattr(task_instance, "dataset") and task_instance.dataset:
        # Extract desired indices cleanly
        target_indices = {
            int(x.strip()) 
            for x in str(ids_source).split(",") 
            if x.strip().isdigit()
        }
        
        if target_indices:
            # Native filtering: Keep the original dataset class instance
            # and just filter the internal sample list by true index positioning.
            task_instance.dataset = task_instance.dataset[list(target_indices)]
            
    return task_instance

@task(name="mbpp_samples")
def mbpp_samples(sample_ids=None):
    """Dynamically filters mbpp samples keeping k8s sandbox context alive."""
    return _safe_apply_filter(mbpp(), sample_ids=sample_ids)

@task(name="gsm8k_samples")
def gsm8k_samples(sample_ids=None):
    """Dynamically filters gsm8k samples keeping basic context alive."""
    return _safe_apply_filter(gsm8k(), sample_ids=sample_ids)
