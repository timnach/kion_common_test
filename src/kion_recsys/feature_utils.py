"""Train-only filtering shared by the active experiments."""
import hashlib
import numpy as np
import pandas as pd


def numeric_dedup(train, baseline, bank):
    """Exclude train-constant and exactly duplicate numeric additions only."""
    hashes, kept, rejected = {}, [], {}
    for col in [*baseline, *bank]:
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        arr = train[col].to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite feature {col}")
        if np.all(arr == arr[0]):
            if col in bank:
                rejected[col] = "train_constant"
            continue
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        if digest in hashes:
            if col in bank:
                rejected[col] = f"train_duplicate_of:{hashes[digest]}"
        else:
            hashes[digest] = col
            if col in bank:
                kept.append(col)
    return kept, rejected
