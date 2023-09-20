import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


PRED_WINDOW = 42  # duration of prediction window (hours)
MIN_LOS = 48  # minimal length of stay (hours)
PRED_GAP = 6  # minimal gap between prediction and target (hours)
PRED_FREQ = '6H'  # frequency for time discretization
MIN_TARGET_ONSET = 30*24  # minimal time of target since discharge (hours)
STAY_GAP = 5*24  # minimal time of target since prediction gap (hours)
HOSP_GAP = 30*24  # maximum time after discharge (hours)


class AgeGenderImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, seen_data, y=None):
      seen_data['age_group'] = pd.cut(seen_data['age'], [0, 18, 30, 40, 50, 60, 70, 80, 400],
                                labels = ['0_17', '18_29', '30_39', '40_49', '50_59', '60_69',  '70_79', '80p'], right = False)
      overall_mean_dict = seen_data.mean().to_dict()
      X_grouped = seen_data.groupby(["age_group", "gender"]).mean()
      X_grouped = X_grouped.fillna(overall_mean_dict)
      self.groups_values = X_grouped
      return self

    def transform(self, X, y=None):
      X = X.reset_index(drop=True)
      X['age_group'] = pd.cut(X['age'], [0, 18, 30, 40, 50, 60, 70, 80, 400],
                                labels = ['0_17', '18_29', '30_39', '40_49', '50_59', '60_69',  '70_79', '80p'], right = False)
      spare = X.loc[:, ["age_group", "gender"]].merge(self.groups_values.reset_index(), on=["age_group", "gender"], how="left")
      spare = spare.loc[:, X.columns.to_list()]
      X = X.combine_first(spare)
      X = X.drop(columns=["hadm_id", "age_group"])
      return X


def get_list_not_norm(full_df, target):
    list_not_norm = []
    for c in full_df.columns:
        vals = full_df[c].unique()
        if (len(vals) == 1 and vals.min() in [0, 1]) or (vals.min() == 0 and vals.max() == 1 and len(vals) == 2):
            list_not_norm.append(c)
    list_not_norm.append('hadm_id')
    if target == "prolonged_LOS":
        list_not_norm += [str(i) for i in range(100)]
    return list_not_norm


def run_CV(X_train, y_train, groups, model, params, scoring, n_splits, list_not_norm):
    imputr = AgeGenderImputer()
    scaler = ColumnTransformer([("standard", StandardScaler(),
                               list(set(X_train.columns.to_list()).difference(set(list_not_norm))))],
                              remainder='passthrough')

    pipe = Pipeline(steps=[
        ("imputer", imputr),
        ("scaler", scaler),
        ("model", model)
      ])

    cv = StratifiedGroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups)
    CV_model = GridSearchCV(pipe, param_grid=params, cv=cv, scoring=scoring, verbose = 4, error_score="raise")
    CV_model.fit(X_train, y_train)
    return CV_model


def evaluation(best_model, X_test, y_test):
    print(best_model.best_params_)
    y_pred = best_model.predict(X_test)
    print("Accuracy = ", accuracy_score(y_test, y_pred))
    print("F1 = ", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def roc_pr_curves(best_model, X_test, y_test):
    sns.set()
    tprs, aucs, y_real, y_proba, precision_array = [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, 2)
    for i, (train, test) in enumerate(GroupShuffleSplit(n_splits=20, train_size=0.8).split(X_test, y_test, groups=X_test["hadm_id"])):
        viz = RocCurveDisplay.from_estimator(
          best_model,
          X_test.iloc[test,:],
          y_test.reset_index(drop=True)[test.tolist()],
          name="ROC fold {}".format(i),
          alpha=0,
          lw=1,
          ax=ax[0],
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        pred_proba = best_model.predict_proba(X_test.iloc[test,:])
        precision_fold, recall_fold, thresh = precision_recall_curve(y_test.reset_index(drop=True)[test.tolist()], pred_proba[:,1])
        precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]  # reverse order of results
        thresh = np.insert(thresh, 0, 1.0)
        precision_array = np.interp(recall_array, recall_fold, precision_fold)
        pr_auc = auc(recall_array, precision_array)
        lab_fold = 'Fold %d AUC=%.4f' % (i, pr_auc)
        ax[1].plot(recall_fold, precision_fold, alpha=0, label=lab_fold)
        y_real.append(y_test.reset_index(drop=True)[test.tolist()])
        y_proba.append(pred_proba[:,1])

    ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    viz = RocCurveDisplay.from_estimator(
          best_model,
          X_test,
          y_test,
          name=r"Total ROC",
          alpha=0.8,
          lw=1,
          color='b',
          ax=ax[0],
      )
    total_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(total_tpr + std_tpr, 1)
    tprs_lower = np.maximum(total_tpr - std_tpr, 0)
    ax[0].fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
    )

    ax[0].set(
      xlim=[-0.05, 1.05],
      ylim=[-0.05, 1.05],
      title="ROC",
    )
    ax[0].legend(bbox_to_anchor=(1.1, -0.2))
    ax[0].set_ylabel("TPR")
    ax[0].set_xlabel("FPR")

    precision, recall, _ = precision_recall_curve(y_test.reset_index(drop=True).tolist(), best_model.predict_proba(X_test)[:,1])
    lab = 'Overall AUC=%.4f' % (auc(recall, precision))

    ax[1].plot(recall, precision, lw=1,color='blue', label=lab)
    ax[1].axhline(y_test.mean(), lw=2, color='red', label='Chance', linestyle="--")
    std_precision = np.std(precision_array)
    ax[1].fill_between(recall, precision + std_precision, precision - std_precision, alpha=0.3, linewidth=0, color='grey', label=r"$\pm$ 1 std. dev.")
    ax[1].legend(bbox_to_anchor=(1, -0.2))
    ax[1].set_title("PR")
    ax[1].set_ylabel("Precision")
    ax[1].set_xlabel("Recall")
    ax[1].yaxis.set_label_position("right")
    plt.show()


def calibration_curve(best_model, X_test, y_test):
    fig, ax1 = plt.subplots()
    display = CalibrationDisplay.from_estimator(
          best_model,
          X_test,
          y_test,
          n_bins=20,
          name="Calibaration best model",
          ax=ax1
      )
    ax2 = ax1.twinx()
    ax2.hist(
          display.y_prob,
          range=(0, 1),
          bins=20,
          alpha=0.3
      )
    ax1.grid(False)
    ax2.grid(False)
    plt.title("calibration curve & histogram of the predicted probabilities")
    plt.show()


def define_x_y_train(full_df, target):
    # Remove unnecessary features
    unnecessary_features = list(set(full_df.columns).intersection({"charttime", "dod", "subject_id", "admittime",
                                                             'ethnicity', 'dob', "diagnosis", "hadm_id_temp",
                                                              "total_vits_labs"}))
    full_df = full_df.drop(columns=unnecessary_features)

    if target == "mortality":
        y = full_df['mort']
        X = full_df.drop(columns=['mort', 'los_hosp_hr', "dischtime"])
    elif target == "prolonged_LOS":
        y = full_df['los_hosp_hr'] > PRED_WINDOW + PRED_GAP + STAY_GAP
        X = full_df.drop(columns=['mort', 'los_hosp_hr', "dischtime"])
    else:
        full_df['sec_admittime'] = full_df['sec_admittime'].apply(lambda x: pd.to_datetime(x))
        full_df['dischtime'] = full_df['dischtime'].apply(lambda x: pd.to_datetime(x))
        y = (full_df['sec_admittime'] - full_df['dischtime']) / pd.to_timedelta(1, 'h') <= HOSP_GAP
        X = full_df.drop(columns=['mort', 'los_hosp_hr', 'dischtime', 'sec_admittime'])

    # Split to train & test (all data of a single patient needs to be in the same group)
    np.random.seed(0)
    gss = GroupShuffleSplit(train_size=0.9)
    ind_train, ind_test = next(gss.split(X, y, groups=X["hadm_id"]))
    X_train, X_test, y_train, y_test = X.iloc[ind_train.tolist(), :], X.iloc[ind_test.tolist(), :], y[
        ind_train.tolist()], y[ind_test.tolist()]
    return X_train, X_test, y_train, y_test


def grid_search_cv(full_df, classifier, params, scoring, target):
    X_train, X_test, y_train, y_test = define_x_y_train(full_df, target)
    best_model = run_CV(X_train, y_train, X_train["hadm_id"], classifier, params, scoring, 5,
                           get_list_not_norm(X_train, target))
    evaluation(best_model, X_test, y_test)
    roc_pr_curves(best_model, X_test, y_test)
    calibration_curve(best_model, X_test, y_test)
    return best_model


def random_forest_gs_cv(full_df, maximum_depth_options, minimal_samples_to_split_options, number_of_trees_options,
                        scoring, target):
    params = {"model__max_depth": maximum_depth_options, "model__min_samples_split":
        minimal_samples_to_split_options, "model__n_estimators": number_of_trees_options}
    best_model_rf = grid_search_cv(full_df, RandomForestClassifier(), params, scoring, target)
    return best_model_rf


def logistic_regression_gs_cv(full_df, c_options, solver_options, max_iter_options, scoring, target):
    params = {"model__C": c_options, "model__solver": solver_options, "model__max_iter": max_iter_options}
    best_model_lr = grid_search_cv(full_df, LogisticRegression(), params, scoring, target)
    return best_model_lr


def mlp_nn_gs_cv(full_df, activation_options, solver_options, lr_options, scoring, target):
    params = {"model__activation": activation_options, "model__solver": solver_options,
              "model__learning_rate_init": lr_options}
    best_model_nn = grid_search_cv(full_df, MLPClassifier(), params, scoring, target)
    return best_model_nn


def define_x_test(full_df, target):
    unnecessary_features = list(set(full_df.columns).intersection({"charttime", "dod", "subject_id", "admittime",
                                                              'ethnicity', 'dob', "diagnosis", "hadm_id_temp",
                                                              "total_vits_labs",'mort', 'los_hosp_hr', 'dischtime',
                                                              'sec_admittime'}))
    full_df = full_df.drop(columns=unnecessary_features)
    list_not_norm = get_list_not_norm(full_df, target)
    imputer = AgeGenderImputer()
    full_df = imputer.fit_transform(full_df)
    with open(r"./Expected_Columns_For_Models/expected_columns_" + target + ".csv", "r") as f:
        required_columns = f.read().splitlines()[1:]
        missing_columns = list(set(required_columns).difference(set(full_df.columns)))
    if len(missing_columns) > 0:
        full_df[missing_columns] = pd.DataFrame([[0] * len(missing_columns)], index=full_df.index)
    full_df.columns = [str(x) for x in full_df.columns]
    if target == "prolonged_LOS":
        full_df = full_df.loc[:, required_columns + [str(i) for i in range(100)]]
    elif target == "readmission":
        full_df = full_df.loc[:, required_columns + [str(i) for i in range(70)]]
    scaler = ColumnTransformer([("standard", StandardScaler(),
                               list(set(full_df.columns.to_list()).difference(set(list_not_norm))))],
                              remainder='passthrough')
    full_df = scaler.fit_transform(full_df)
    return full_df
