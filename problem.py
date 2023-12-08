import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import rampwf as rw

problem_title = "Volcanic events prediction from tephras"


_event_label_names = [
    "1852 Llaima",
    "Achen",
    "Arauco",
    "Cha1",
    "Enco",
    "Grande",
    "H1",
    "HW3",
    "HW6",
    "Hua-hum",
    "Huilo",
    "La Junta",
    "Lepué",
    "Llaima Pumice",
    "MB1",
    "MC12",
    "Mil Hojas",
    "Neltume",
    "PCC2",
    "PCC2011",
    "Pirehueico",
    "Playas Blanca-Negra",
    "Pucón",
    "Puesco",
    "Quet1",
    "R1",
    "Ranco",
    "Riñihue",
    "Vcha-2008",
    "Vilcún",
]

# Correspondence between categories and int8 categories
# Mapping int to categories
int_to_cat = {
    0: "1852 Llaima",
    1: "Achen",
    2: "Arauco",
    3: "Cha1",
    4: "Enco",
    5: "Grande",
    6: "H1",
    7: "HW3",
    8: "HW6",
    9: "Hua-hum",
    10: "Huilo",
    11: "La Junta",
    12: "Lepué",
    13: "Llaima Pumice",
    14: "MB1",
    15: "MC12",
    16: "Mil Hojas",
    17: "Neltume",
    18: "PCC2",
    19: "PCC2011",
    20: "Pirehueico",
    21: "Playas Blanca-Negra",
    22: "Pucón",
    23: "Puesco",
    24: "Quet1",
    25: "R1",
    26: "Ranco",
    27: "Riñihue",
    28: "Vcha-2008",
    29: "Vilcún",
}
# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

_event_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_event_label_int)
workflow = rw.workflows.Classifier()


score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]


def _get_data(path=".", split="train"):
    # Load data from csv files into pd.DataFrame
    #
    # returns X (input) and y (output) arrays

    data = pd.read_csv(os.path.join(path, "data", split + ".csv"))

    # Retrieve the geochemical data for X.
    # FeO, Fe2O3 and FeO2O3T are dropped because FeOT
    # is a different expression of the same element (Fe).
    # P2O5 and Cl are also dropped because they are sporadically analyzed.
    majors = [
        "SiO2_normalized",
        "TiO2_normalized",
        "Al2O3_normalized",
        "FeOT_normalized",
        # 'FeO_normalized', 'Fe2O3_normalized', 'Fe2O3T_normalized',
        "MnO_normalized",
        "MgO_normalized",
        "CaO_normalized",
        "Na2O_normalized",
        "K2O_normalized",
        # 'P2O5_normalized','Cl_normalized'
    ]
    traces = [
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "Pb",
        "Th",
        "U",
    ]

    X_majors = data.loc[:, majors]
    X_traces = data.loc[:, traces]
    X_df = pd.concat([X_majors, X_traces], axis=1)

    X = X_df.to_numpy()

    # labels
    y = np.array(data["Event"].map(cat_to_int).fillna(-1).astype("int8"))

    return X, y


groups = None


# Here we will define a global variable (groups) to be used in get_cv
# for the SGKF CV strategy
def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "train.csv"))
    data_df = data.copy()
    data_df["SampleID"] = data_df["SampleID"].astype("category")
    SampleID = np.array(data_df["SampleID"].cat.codes)
    global groups
    groups = SampleID
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


# def get_groups(path="."):
#     data = pd.read_csv(os.path.join(path, "data", "train.csv"))
#     data_df = data.copy()
#     data_df["SampleID"] = data_df["SampleID"].astype("category")
#     SampleID = np.array(data_df["SampleID"].cat.codes)
#     groups = SampleID
#     return groups


def get_cv(X, y):
    # groups = get_groups()
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)
