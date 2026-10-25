import pandas as pd
from kion_recsys.feature_utils import numeric_dedup


def test_bank_dedup_uses_only_provided_training_rows():
    train=pd.DataFrame({"base":[1.,2.,3.],"same":[1.,2.,3.],"constant":[0.,0.,0.],"new":[3.,1.,2.]})
    kept,rejected=numeric_dedup(train,["base"],["same","constant","new"])
    assert kept==["new"]
    assert rejected=={"same":"train_duplicate_of:base","constant":"train_constant"}
