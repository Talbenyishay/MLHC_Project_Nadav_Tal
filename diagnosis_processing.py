import json
import re
from collections import defaultdict
import requests
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd

USER = 'talbenyishay'
API_KEY = '013755e6-333e-44a8-a91d-4089407e0428'


# The output of MetaMap is a complicated json, this function extracts the diseases from it
def extract_diseases(metamap_res):
    out = []
    for doc in metamap_res['AllDocuments']:
        for utterance in doc['Document']['Utterances']:
            for phras in utterance['Phrases']:
                for mapping in phras['Mappings']:
                    for mapping_candidate in mapping['MappingCandidates']:
                        if ('dsyn' in mapping_candidate['SemTypes']) or ('neop' in mapping_candidate['SemTypes']) or\
                                ('sosy' in mapping_candidate['SemTypes']) or ('fndg' in mapping_candidate['SemTypes']):
                            out.append ({
                                'PhraseText': phras['PhraseText'],
                                'CandidateCUI': mapping_candidate['CandidateCUI'],
                                'CandidateScore': mapping_candidate['CandidateScore'],
                                'CandidatePreferred': mapping_candidate['CandidatePreferred'],
                                'SemTypes': mapping_candidate['SemTypes']
                            })
    return out


# Use UTS API
def make_api_request(relative_path, api_key):
    base_url = "https://uts-ws.nlm.nih.gov/rest/"
    url = f"{base_url}{relative_path}"
    params = {"apiKey": api_key,'pageSize': 100}
    response = requests.get(url, params=params)
    response_json = json.loads(response.content.decode("utf-8"))
    if response.status_code == 404:
        return None
    return response_json["result"]


# Convert CUI to NCI code
def cui_to_ncis(cui, api_key=API_KEY):
  nci_codes = []
  cui_data = make_api_request(f"content/current/CUI/{cui}/atoms?sabs=NCI", api_key)
  if cui_data is None:
    return []
  for entry in cui_data:
      if entry['rootSource'] == 'NCI':
          code = re.search('https://uts-ws.nlm.nih.gov/rest/content/2023AA/source/NCI/(?P<CODE>.*)', entry['code']).group('CODE')
          nci_codes.append(code)
  return nci_codes


# Get NCI ancestors of a NCI code
def get_nci_ancestors(nci_code, api_key = API_KEY):
    ancestors = make_api_request(f"content/current/source/NCI/{nci_code}/ancestors", api_key)
    return [anncestor['ui'] for anncestor in ancestors if anncestor['rootSource'] == 'NCI']


def save_cui_nci_dict(json_path, file_name="cui_nci_dict"):
    with open(json_path, 'r') as f:
        results_json = json.load(f)
    df = pd.Series(results_json.keys()).to_frame().rename(columns={0: "Diagnosis"})
    df["cui_set"] = df["Diagnosis"].apply(lambda x: set([hit["CandidateCUI"] for hit in extract_diseases(results_json[x])]))
    unique_cui = sorted(list(set.union(*(df["cui_set"].to_list()))))
    for i, cui in enumerate(unique_cui):
        with open("important_files/" + file_name + ".txt", "a") as f:
            f.write(cui + " " + str(cui_to_ncis(cui)) + "\n")
            print(i, cui)


def save_nci_nci_ancestors_dict(json_path, cui_nci_dict_name="cui_nci_dict",
                                nci_nci_ancestors_dict_name="nci_nci_ancestors_dict"):
    with open(json_path, 'r') as f:
        results_json = json.load(f)
    df = pd.Series(results_json.keys()).to_frame().rename(columns={0: "Diagnosis"})
    df["cui_set"] = df["Diagnosis"].apply(lambda x: set([hit["CandidateCUI"] for hit in extract_diseases(results_json[x])]))
    cui_nci_dict = defaultdict(lambda: [])
    with open("important_files/" + cui_nci_dict_name + ".txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        l = line.split("[")
        cui, nci_list = l[0][:-1], l[1][:-2].replace("'", "").split(", ")
        if nci_list[0] == "":
            cui_nci_dict[cui] = set()
        else:
            cui_nci_dict[cui] = set(nci_list)
    df["nci_set"] = df["cui_set"].apply(lambda x: set([y for row in [cui_nci_dict[i] for i in x] for y in row]))
    # extracting ancestors for each nci term
    unique_nci = sorted(list(set.union(*df["nci_set"].to_list())))
    for nci in unique_nci:
        with open("important_files/" + nci_nci_ancestors_dict_name + ".txt", "a") as f:
            nci_ancestors = get_nci_ancestors(nci)
            f.write(nci + "\t" + str(nci_ancestors) + "\n")
            print(nci, nci_ancestors)


def save_nci_sets(json_path, cui_nci_dict_name="cui_nci_dict", nci_nci_ancestors_dict_name="nci_nci_ancestors_dict",
                  nci_sets_name="diagnosis_nci_sets_df"):
    with open(json_path, 'r') as f:
        results_json = json.load(f)
    df = pd.Series(results_json.keys()).to_frame().rename(columns={0: "Diagnosis"})
    df["cui_set"] = df["Diagnosis"].apply(lambda x: set([hit["CandidateCUI"] for hit in extract_diseases(results_json[x])]))
    cui_nci_dict = defaultdict(lambda: [])
    with open("important_files/" + cui_nci_dict_name + ".txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        l = line.split("[")
        cui, nci_list = l[0][:-1], l[1][:-2].replace("'", "").split(", ")
        if nci_list[0] == "":
            cui_nci_dict[cui] = set()
        else:
            cui_nci_dict[cui] = set(nci_list)
    df["nci_set"] = df["cui_set"].apply(lambda x: set([y for row in [cui_nci_dict[i] for i in x] for y in row]))
    nci_nci_ancestors_df = pd.read_csv("important_files/" + nci_nci_ancestors_dict_name + ".txt", sep="\t", header=None)
    nci_nci_ancestors_dict = dict(zip(nci_nci_ancestors_df[0].to_list(), nci_nci_ancestors_df[1].to_list()))
    for nci_term in nci_nci_ancestors_dict.keys():
        nci_nci_ancestors_dict[nci_term] = nci_nci_ancestors_dict[nci_term][1:-1].replace("'", "").split(", ")
        nci_nci_ancestors_dict[nci_term].append(nci_term)
    df = df.drop(columns=["cui_set"])
    df["nci_set"] = df["nci_set"].apply(lambda x: set([y for row in [nci_nci_ancestors_dict[i] for i in x] for y in row]))
    df.to_csv(r"important_files/" + nci_sets_name + ".csv", index=False)


def prolonged_stays_training(nci_sets_name="diagnosis_nci_sets_df_train"):
    df_sets = pd.read_csv(r"important_files/" + nci_sets_name + ".csv")
    df_sets["nci_set"] = df_sets["nci_set"].replace("set()", "{}")
    df_sets["nci_set"] = df_sets["nci_set"].apply(lambda x: set(x[1: -1].split(", ")))
    for nci in sorted(list(set.union(*df_sets["nci_set"].to_list()).difference({""}))):
        curr_nci_col = []
        print(nci)
        for i, diagnosis in enumerate(df_sets["Diagnosis"]):
            curr_nci_col.append(nci in df_sets.loc[i, "nci_set"])
        df_sets[nci.replace("'", "")] = curr_nci_col
    df_heat_map = df_sets.copy()
    df_heat_map = df_heat_map.drop(columns=["nci_set"])
    df_heat_map = df_heat_map.set_index("Diagnosis")
    df_heat_map.to_csv("matrix_ncis_per_diagnosis_training.csv")


def get_features_for_prolonged_stays(diagnoses):
    df_diagnosis_vectors = pd.read_csv("important_files/prolonged/matrix_ncis_per_diagnosis_training.csv", index_col=0)
    pca = PCA(n_components=100)
    pca.fit(df_diagnosis_vectors)
    mean, components = pca.mean_, pca.components_
    pcs_for_prolonged_stays = pd.DataFrame(columns=["diagnosis"] + [i for i in range(100)])
    df_sets = pd.read_csv(r"important_files/prolonged/diagnosis_nci_sets_df.csv")
    df_sets["nci_set"] = df_sets["nci_set"].replace("set()", "{}")
    df_sets["nci_set"] = df_sets["nci_set"].apply(lambda x: set(x[1: -1].split(", ")))
    df_sets = df_sets.drop_duplicates(subset=["Diagnosis"])
    df_sets = df_sets.set_index("Diagnosis")
    df_heat_map = pd.DataFrame(index=diagnoses)
    for nci in sorted(df_diagnosis_vectors.columns.to_list()):
        curr_nci_col = []
        for i, diagnosis in enumerate(diagnoses):
            curr_nci_col.append(nci in [x.replace("'", "") for x in df_sets.loc[diagnosis, "nci_set"]])
        df_heat_map[nci.replace("'", "")] = curr_nci_col
    for diagnosis in diagnoses:
        diagnosis_vector = df_heat_map.loc[diagnosis]
        pcs_for_prolonged_stays.loc[len(pcs_for_prolonged_stays)] = [diagnosis] + list((diagnosis_vector - mean) @ components.T)
    return pcs_for_prolonged_stays


def cluster_train_data(nci_nci_ancestors_dict_name="nci_nci_ancestors_dict_train"):
    nci_nci_ancestors_df = pd.read_csv("important_files/" + nci_nci_ancestors_dict_name + ".txt", sep="\t", header=None)
    nci_nci_ancestors_dict = dict(zip(nci_nci_ancestors_df[0].to_list(), nci_nci_ancestors_df[1].to_list()))
    for nci_term in nci_nci_ancestors_dict.keys():
        nci_nci_ancestors_dict[nci_term] = nci_nci_ancestors_dict[nci_term][1:-1].replace("'", "").split(", ")
        nci_nci_ancestors_dict[nci_term].append(nci_term)
    columns = set(sum(nci_nci_ancestors_dict.values(), []))
    df_ans = pd.DataFrame(index=sorted(nci_nci_ancestors_dict.keys()))
    for nci in sorted(columns):
        curr_nci_col = []
        for i, diagnosis in enumerate(sorted(nci_nci_ancestors_dict.keys())):
            curr_nci_col.append(nci in nci_nci_ancestors_dict[diagnosis])
        df_ans[nci.replace("'", "")] = curr_nci_col
    n_clusters = 70
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(df_ans)
    df_ans["Cluster"] = clustering.labels_
    dict_nci_cluster = dict(zip(df_ans.index.to_list(), df_ans["Cluster"].to_list()))
    return dict_nci_cluster


def get_cuis_from_diagnosis(diagnoses_cui_dict, diagnosis):
    return diagnoses_cui_dict[diagnosis]


def get_ncis_from_cui(cui_nci_dict, cui):
    return cui_nci_dict[cui]


def get_cluster_from_nci(dict_nci_cluster, nci):
    return dict_nci_cluster[nci]


def get_features_for_readmission(diagnoses, json_path="json_metadata_results", cui_nci_dict_name="cui_nci_dict"):
    with open(r"important_files/" + json_path + ".json", 'r') as f:
        results_json = json.load(f)
    df = pd.Series(results_json.keys()).to_frame().rename(columns={0: "Diagnosis"})
    df["cui_set"] = df["Diagnosis"].apply(
        lambda x: list(set([hit["CandidateCUI"] for hit in extract_diseases(results_json[x])])))
    diagnoses_cui_dict = dict(zip(df["Diagnosis"].to_list(), df["cui_set"].to_list()))
    cui_nci_dict = defaultdict(lambda: [])
    with open("important_files/" + cui_nci_dict_name + ".txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        l = line.split("[")
        cui, nci_list = l[0][:-1], l[1][:-2].replace("'", "").split(", ")
        if nci_list[0] == "":
            cui_nci_dict[cui] = set()
        else:
            cui_nci_dict[cui] = set(nci_list)
    dict_nci_cluster = cluster_train_data()
    results = pd.DataFrame(columns=[str(i) for i in range(70)], index=diagnoses)
    for diagnosis in diagnoses:
        cuis = get_cuis_from_diagnosis(diagnoses_cui_dict, diagnosis)
        for cui in cuis:
            ncis = get_ncis_from_cui(cui_nci_dict, cui)
            for nci in ncis:
                results.loc[diagnosis, str(get_cluster_from_nci(dict_nci_cluster, nci))] = 1
    results = results.reset_index().rename(columns={"index": "diagnosis"})
    return results.fillna(0)
