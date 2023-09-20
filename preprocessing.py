from google.cloud import bigquery
import pandas as pd
import numpy as np
from diagnosis_processing import *

# project = 'virtual-stratum-384310' # Your Bigquery Project ID
# client = bigquery.Client(project=project)

USER = 'talbenyishay'
API_KEY = '756896a9-5cc2-4ac5-93bc-dc970b417379'

PRED_WINDOW = 42  # duration of prediction window (hours)
MIN_LOS = 48  # minimal length of stay (hours)
PRED_GAP = 6  # minimal gap between prediction and target (hours)
PRED_FREQ = '6H'  # frequency for time discretization


def get_age(admittime, dob):
    if admittime < dob:
      return 0
    return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))


def retrieve_admissions(subject_ids, target, client=bigquery.Client(project="virtual-stratum-384310")):
    hospquery = \
        """
        SELECT admissions.subject_id, admissions.hadm_id
        , admissions.admittime, admissions.dischtime
        , admissions.ethnicity, admissions.diagnosis
        , patients.gender, patients.dob, patients.dod
        FROM `physionet-data.mimiciii_clinical.admissions` admissions
        INNER JOIN `physionet-data.mimiciii_clinical.patients` patients
            ON admissions.subject_id = patients.subject_id
        WHERE admissions.has_chartevents_data = 1 AND admissions.subject_id in UNNEST(@subjectids)
        ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("subjectids", "INTEGER", subject_ids),
        ]
    )
    hosps = client.query(hospquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    hosps['age'] = hosps.apply(lambda row: get_age(row['admittime'], row['dob']), axis=1)
    hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime).astype('timedelta64[h]')
    hosps['mort'] = np.where(~np.isnat(hosps.dod), 1, 0)

    # Ethnicity - one hot encoding
    hosps.ethnicity = hosps.ethnicity.str.lower()
    hosps.loc[(hosps.ethnicity.str.contains('^white')), 'ethnicity'] = 'white'
    hosps.loc[(hosps.ethnicity.str.contains('^black')), 'ethnicity'] = 'black'
    hosps.loc[
        (hosps.ethnicity.str.contains('^hisp')) | (hosps.ethnicity.str.contains('^latin')), 'ethnicity'] = 'hispanic'
    hosps.loc[(hosps.ethnicity.str.contains('^asia')), 'ethnicity'] = 'asian'
    hosps.loc[~(hosps.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))), 'ethnicity'] = 'other'
    hosps = pd.concat([hosps, pd.get_dummies(hosps['ethnicity'], prefix='eth')], axis=1)

    # Gender to binary
    hosps['gender'] = np.where(hosps['gender'] == "M", 1, 0)

    if target == "readmission":
        sec = hosps.sort_values('admittime').groupby('subject_id').nth(1).reset_index().loc[:,
              ['subject_id', 'admittime']]
        sec.columns = ['subject_id', 'sec_admittime']
        hosps = hosps.merge(sec, how='left', on='subject_id')
    return hosps


def patient_exclusion_criteria(hosps, target):
    """
    :param target: one of the option: {mortality, prolonged_LOS, readmission}
    :return:
    """
    hosps = hosps.sort_values('admittime').groupby('subject_id').first().reset_index()  # include only first admissions
    # print(f"1. Include only first admissions: N={hosps.shape[0]}")
    hosps = hosps[hosps['age'] >= 18]  # exclude minors
    # print(f"2. Include only adult patients: N={hosps.shape[0]}")
    hosps = hosps[hosps["los_hosp_hr"] >= MIN_LOS]  # exclude short admissions
    # print(f"3. Include only patients who admitted for at least {MIN_LOS} hours: N={hosps.shape[0]}")

    if target == "mortality":
        min_target_onset = 30 * 24  # minimal time of target since discharge (hours)
        # Exclude by death time - Include patients who never died / died between 48hr post admission and 720-hr post discharge
        hosps = hosps[(hosps["dod"].isna()) | ((hosps['admittime'] + pd.to_timedelta(PRED_WINDOW, unit='h')
                                                + pd.to_timedelta(PRED_GAP, unit='h') <= hosps['dod'])
                                               & (hosps['dischtime'] + pd.to_timedelta(min_target_onset, unit='h') >=
                                                  hosps['dod']))]
        # print(
        #     f"4. Include patients who never died / died between {MIN_LOS}-hr post admission and {min_target_onset}-hr post discharge: N={hosps.shape[0]}")
    elif target == "prolonged_LOS":
        stay_gap = 5 * 24  # minimal time of target since prediction gap (hours)
        # Exclude by death time
        hosps = hosps[(hosps["dod"].isna()) | (hosps['admittime'] + pd.to_timedelta(PRED_WINDOW, unit='h')
                                               + pd.to_timedelta(PRED_GAP, unit='h') + pd.to_timedelta(stay_gap,
                                                                                                       unit='h') <=
                                               hosps['dod'])]
        # print(
        #     f"4. Exclude patients who died within {PRED_WINDOW + PRED_GAP + stay_gap}-hours of admission: N={hosps.shape[0]}")
    else:
        hosp_gap = 30 * 24  # maximum time after discharge (hours)
        # Exclude patients who died in the first admission
        hosps = hosps[(hosps["dod"].isna()) | (hosps['dod'] <= hosps['dischtime'])]
        # print(f"4. Exclude patients who died within the first admission: N={hosps.shape[0]}")

    valid_hadm_ids = [int(x) for x in hosps["hadm_id"].unique()]
    return hosps, valid_hadm_ids


def retrieve_microbiology(valid_hadm_ids, client=bigquery.Client(project="virtual-stratum-384310")):
    microquery = \
        """--sql
          SELECT micro.subject_id ,micro.hadm_id, micro.spec_type_desc, micro.charttime as micro_charttime,
          micro.org_name, micro.ab_name, micro.interpretation, admissions.admittime,
          FROM `physionet-data.mimiciii_clinical.microbiologyevents` micro
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
            ON micro.subject_id = admissions.subject_id
            AND micro.hadm_id = admissions.hadm_id
            AND
            (lower(micro.org_name) like '%mycobacterium tuberculosis%'
  
            OR lower(micro.org_name) like '%pneumonia%'
            OR lower(micro.org_name) like '%staphylococcus aureus%'
            OR lower(micro.org_name) like '%haemophilus influenza%'
            OR lower(micro.org_name) like '%legionella pneumophila%'
            )
            AND micro.charttime >=(admissions.admittime)
            AND micro.hadm_id in UNNEST(@hadmids)
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadmids", "INTEGER", valid_hadm_ids)
        ]
    )
    micros = client.query(microquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    # Exclusion
    micros["hours since admission"] = (micros["micro_charttime"] - micros["admittime"]).dt.total_seconds() / 3600
    micros = micros[micros["hours since admission"] <= PRED_WINDOW]
    # Pivot
    micros['pivot'] = micros['org_name'] + '_' + micros['interpretation'].fillna('na')
    micros = micros.loc[:, ['hadm_id', 'pivot']]
    micros['1'] = [1 for i in range(len(micros))]
    micros = pd.pivot_table(data=micros, index='hadm_id', columns='pivot', values='1')
    micros = micros.fillna(0).reset_index()
    # print(len(micros["hadm_id"].unique()))
    return micros


def retrieve_services(valid_hadm_ids, client=bigquery.Client(project="virtual-stratum-384310")):
    srvquery = \
        """--sql
          SELECT srv.subject_id ,srv.hadm_id,
           srv.curr_service as service, srv.transfertime as srv_time, admissions.admittime,
          FROM `physionet-data.mimiciii_clinical.services` srv
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
            ON srv.subject_id = admissions.subject_id
            AND srv.hadm_id = admissions.hadm_id
            AND srv.hadm_id in UNNEST(@hadmids)
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadmids", "INTEGER", valid_hadm_ids)
        ]
    )
    services = client.query(srvquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    # Exclusion
    services["hours since admission"] = (services["srv_time"] - services["admittime"]).dt.total_seconds() / 3600
    services = services[services["hours since admission"] <= PRED_WINDOW]
    # Pivot
    services = services.loc[:, ['hadm_id', 'service']]
    services['1'] = [1 for i in range(len(services))]
    services = pd.pivot_table(data=services, index='hadm_id', columns='service', values='1')
    services = services.fillna(0).reset_index()
    # print(len(services["hadm_id"].unique()))
    return services


def retrieve_drugs(valid_hadm_ids, client=bigquery.Client(project="virtual-stratum-384310")):
    # Cancer in ['Vinblastine Sulfate','Busulfan','Anastrozole','anastrozole','Anagrelide HCl', 'DOXOrubicin', 'Methotrexate Sodium', 'VinCRIStine', 'Mitoxantrone HCl', 'Rituximab', 'Trastuzumab', 'Thalidomide', 'Pamidronate'] or like 'Tretinoin%'
    # Heart in ['Bisoprolol Fumarate', 'Diltiazem','Nimodipine', 'Dipyridamole-Aspirin'] or lower() like '%nicardipine%'
    # Brain in ['Amantadine','pramipexole','Pramipexole','Oxcarbazepine','Gabapentin','Pregabalin'] or like 'LeVETiracetam%' or like 'Carbamazepine%'
    drgquery = \
        """--sql
          SELECT prs.subject_id ,prs.hadm_id,
          prs.startdate, admissions.admittime,
          CASE
          WHEN  prs.drug  in ('Vinblastine Sulfate','Busulfan','Anastrozole','anastrozole','Anagrelide HCl', 'DOXOrubicin', 'Methotrexate Sodium', 'VinCRIStine', 'Mitoxantrone HCl', 'Rituximab', 'Trastuzumab', 'Thalidomide', 'Pamidronate')
                  OR prs.drug  like 'Tretinoin%' THEN 'cancer_related_drugs'
          WHEN prs.drug  in ('Bisoprolol Fumarate', 'Diltiazem','Nimodipine', 'Dipyridamole-Aspirin')
                  OR  lower(prs.drug) like '%nicardipine%' THEN 'heart_related_drugs'
          WHEN prs.drug  in ('Amantadine','pramipexole','Pramipexole','Oxcarbazepine','Gabapentin','Pregabalin')
                  OR (prs.drug like 'LeVETiracetam%' OR prs.drug like 'Carbamazepine%') THEN 'brain_related_drugs'
          END as disease
          FROM `physionet-data.mimiciii_clinical.prescriptions` prs
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
            ON prs.subject_id = admissions.subject_id
            AND prs.hadm_id = admissions.hadm_id
            AND prs.hadm_id in UNNEST(@hadmids)
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadmids", "INTEGER", valid_hadm_ids)
        ]
    )
    drugs = client.query(drgquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    # Exclusion
    drugs["hours since admission"] = (drugs["startdate"] - drugs["admittime"]).dt.total_seconds() / 3600
    drugs = drugs[drugs["hours since admission"] <= PRED_WINDOW]
    # Pivot
    drugs = drugs.loc[:, ['hadm_id', 'disease']]
    drugs['1'] = [1 for i in range(len(drugs))]
    drugs = pd.pivot_table(data=drugs, index='hadm_id', columns='disease', values='1')
    drugs = drugs.fillna(0).reset_index()
    # print(len(drugs["hadm_id"].unique()))
    return drugs


def retrieve_labs(valid_hadm_ids, client=bigquery.Client(project="virtual-stratum-384310")):
    labquery = \
        """--sql
          SELECT labevents.subject_id ,labevents.hadm_id ,labevents.charttime
          , labevents.itemid, labevents.valuenum
          , admissions.admittime
          FROM `physionet-data.mimiciii_clinical.labevents` labevents
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
            ON labevents.subject_id = admissions.subject_id
            AND labevents.hadm_id = admissions.hadm_id
            AND labevents.charttime >=(admissions.admittime)
            AND itemid in UNNEST(@itemids)
            AND labevents.hadm_id in UNNEST(@hadm_id)
        """
    lavbevent_meatdata = pd.read_csv('labs_metadata.csv')
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("itemids", "INTEGER", lavbevent_meatdata['itemid'].tolist()),
            bigquery.ArrayQueryParameter("hadm_id", "INTEGER", valid_hadm_ids),
        ]
    )
    labs = client.query(labquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    # Filter Invalid Measurments
    labs = labs[labs['hadm_id'].isin(valid_hadm_ids)]
    labs = pd.merge(labs, lavbevent_meatdata, on='itemid')
    labs = labs[labs['valuenum'].between(labs['min'], labs['max'], inclusive='both')]

    labs["admittime"] = labs["admittime"].apply(lambda x: x.to_pydatetime())
    labs["charttime"] = pd.to_datetime(labs['charttime'])
    labs["hours since admission"] = (labs["charttime"] - labs["admittime"]).dt.total_seconds() / 3600
    labs = labs[labs["hours since admission"] <= PRED_WINDOW]

    # print(len(labs["hadm_id"].unique()))
    return labs


def retrieve_vitals(valid_hadm_ids, client=bigquery.Client(project="virtual-stratum-384310")):
    vitquery = \
        """--sql
        -- Vital signs include heart rate, blood pressure, respiration rate, and temperature
    
          SELECT chartevents.subject_id ,chartevents.hadm_id ,chartevents.charttime
          , chartevents.itemid, chartevents.valuenum
          , admissions.admittime
          FROM `physionet-data.mimiciii_clinical.chartevents` chartevents
          INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
          ON chartevents.subject_id = admissions.subject_id
          AND chartevents.hadm_id = admissions.hadm_id
          AND chartevents.charttime >= (admissions.admittime)
          AND itemid in UNNEST(@itemids)
          -- exclude rows marked as error
          AND chartevents.error IS DISTINCT FROM 1
          AND chartevents.hadm_id in UNNEST(@hadmids)
        """
    vital_meatdata = pd.read_csv('vital_metadata.csv')
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("itemids", "INTEGER", vital_meatdata['itemid'].tolist()),
            bigquery.ArrayQueryParameter("hadmids", "INTEGER", valid_hadm_ids),
        ]
    )
    vits = client.query(vitquery, job_config=job_config).result().to_dataframe().rename(str.lower, axis='columns')
    # Filter Invalid Measurement
    vits = vits[vits['hadm_id'].isin(valid_hadm_ids)]
    vits = pd.merge(vits, vital_meatdata, on='itemid')
    vits = vits[vits['valuenum'].between(vits['min'], vits['max'], inclusive='both')]

    vits["admittime"] = vits["admittime"].apply(lambda x: x.to_pydatetime())
    vits["charttime"] = pd.to_datetime(vits['charttime'])
    vits["hours since admission"] = (vits["charttime"] - vits["admittime"]).dt.total_seconds() / 3600
    vits = vits[vits["hours since admission"] <= PRED_WINDOW]
    # Units Conversion
    vits.loc[(vits['feature name'] == 'TempF'), 'valuenum'] = (vits[vits['feature name'] == 'TempF'][
                                                                   'valuenum'] - 32) / 1.8
    vits.loc[vits['feature name'] == 'TempF', 'feature name'] = 'TempC'
    # print(len(vits["hadm_id"].unique()))
    return vits


def time_discretization(vits, labs, hosps):
    vits_labs = pd.concat([vits, labs])
    vits_labs_pivot = pd.pivot_table(vits_labs, index=["hadm_id", "charttime"], columns=["feature name"],
                                     values="valuenum")
    vits_labs_freq = vits_labs_pivot.groupby([pd.Grouper(level="charttime", freq=PRED_FREQ), "hadm_id"]).agg(
        ['min', 'max', 'mean'])
    vits_labs_freq.columns = ['_'.join(col) for col in vits_labs_freq.columns.values]
    hosps_vits_labs_freq = vits_labs_freq.reset_index().merge(hosps, on="hadm_id")
    # hosps_vits_labs_freq.reset_index(drop=True).to_csv("hosps_vits_labs_freq.csv", index=False)
    return hosps_vits_labs_freq


def ffill_imputation(hosps_vits_labs_freq):
    hosps_vits_labs_freq = hosps_vits_labs_freq.sort_values(["hadm_id", "charttime"])
    hosps_vits_labs_freq["hadm_id_temp"] = hosps_vits_labs_freq["hadm_id"]
    hosps_vits_labs_freq_imp = hosps_vits_labs_freq.groupby('hadm_id_temp').fillna(method="ffill")
    # hosps_vits_labs_freq_imp.reset_index(drop=True).to_csv('hosps_vits_labs_freq_imp.csv', index=False)
    return hosps_vits_labs_freq_imp


def add_micros_services_drugs(hosps_vits_labs_freq_imp, micros, services, drugs):
    modals = drugs.merge(micros, how='outer', on='hadm_id').merge(services, how='outer', on='hadm_id')
    modals = pd.DataFrame(hosps_vits_labs_freq_imp.loc[:, 'hadm_id']).merge(modals, how='left', on='hadm_id')
    modals = modals.fillna(0)
    full_df = hosps_vits_labs_freq_imp.merge(modals, how='left', on='hadm_id')
    # full_df.to_csv('full_df.csv', index=False)
    return full_df


def get_full_dataset(subject_ids, target, client=bigquery.Client(project="virtual-stratum-384310")):
    hosps = retrieve_admissions(subject_ids, target, client)
    hosps, valid_hadm_ids = patient_exclusion_criteria(hosps, target)
    micros = retrieve_microbiology(valid_hadm_ids, client)
    services = retrieve_services(valid_hadm_ids, client)
    drugs = retrieve_drugs(valid_hadm_ids, client)
    labs = retrieve_labs(valid_hadm_ids, client)
    vitals = retrieve_vitals(valid_hadm_ids, client)
    hosps_vits_labs_freq = time_discretization(vitals, labs, hosps)
    hosps_vits_labs_freq = ffill_imputation(hosps_vits_labs_freq)
    full_df = add_micros_services_drugs(hosps_vits_labs_freq, micros, services, drugs)
    if target == "prolonged_LOS":
        pca_diagnosis = get_features_for_prolonged_stays(list(full_df["diagnosis"].unique()))
        full_df = full_df.merge(pca_diagnosis, on="diagnosis")
    elif target == "readmission":
        clusters_diagnosis = get_features_for_readmission(list(full_df["diagnosis"].unique()))
        full_df = full_df.merge(clusters_diagnosis, on="diagnosis")
    return full_df



