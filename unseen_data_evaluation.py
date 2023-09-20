import pandas as pd
from google.cloud import bigquery
import pickle
import os
import warnings
os.chdir(os.path.dirname(__file__))
warnings.filterwarnings('ignore')
from models import *
from preprocessing import *

def target_prediction(subject_ids, target, client):
    full_df = get_full_dataset(subject_ids, target, client)
    hadm_to_subject_dict = dict(zip(full_df["hadm_id"], full_df["subject_id"]))
    results = full_df.loc[:, ["hadm_id"]].copy()
    results["subject_id"] = results["hadm_id"].apply(lambda x: hadm_to_subject_dict[x])
    full_df = define_x_test(full_df, target)
    with open(r"Saved_models/" + target + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
    results[target + "_proba"] = pd.DataFrame(model.predict_proba(full_df)).iloc[:, 1]
    results = results.drop_duplicates(subset=["subject_id"]).drop(columns=["hadm_id"])
    return results


def run_pipeline_on_unseen_data(subject_ids, client):
    """
    Run your full pipeline, from data loading to prediction.

    :param subject_ids: A list of subject IDs of an unseen test set.
    :type subject_ids: List[int]

    :param client: A BigQuery client object for accessing the MIMIC-III dataset.
    :type client: google.cloud.bigquery.client.Client

    :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
    :rtype: pandas.DataFrame
    """
    results = pd.DataFrame(index=subject_ids).reset_index().rename(columns={"index": "subject_id"})
    for target in ["mortality", "prolonged_LOS", "readmission"]:
        results = results.merge(target_prediction(subject_ids, target, client), on="subject_id", how="left")
    return results
