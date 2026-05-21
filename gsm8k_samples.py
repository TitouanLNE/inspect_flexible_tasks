import os
from inspect_ai import task, task_with
from inspect_evals.gsm8k import gsm8k

@task
def gsm8k_dynamic():
    if sample_ids_raw:
        target_ids = [int(x.strip()) for x in sample_ids_raw.split(",") if x.strip().isdigit()]
        return task_with(gsm8k(), sample_id=target_ids)
    return gsm8k()

    if sample_ids_raw:
        target_ids = [int(x.strip()) for x in sample_ids_raw.split(",") if x.strip().isdigit()]
        return task_with(gsm8k(), sample_id=target_ids)
    return gsm8k()
