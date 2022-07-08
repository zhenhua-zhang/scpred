#!/usr/bin/env python3
# Author:  Zhenhua Zhang
# E-mail:  zhenhua.zhang217@gmail.com
# Created: 2022 Mar 01
# Updated: 2022 Jun 10

# NOTE:
#   1. A configure rule to encode the category variables. Add a function to
#      encode category variables. Use config.json? Generate a JSON file of the
#      encoding.
#   2. Features doesn't match between dataset. Currently the missing features
#      were assigned to 0, this should be bias.
#   3. An option to keep features by force to gurentee it's included in the
#      predictor variables.

import os
import json
import logging
import argparse

from datetime import datetime as dt

import matplotlib.pyplot as plt
import joblib as jbl
import pickle as pkl
import pandas as pd
import numpy as np
import shap as sp

import torch
import torch.nn as tc_nn
import torch.nn.functional as tc_func
import torch.optim as tc_optim

from torch.utils.data.dataloader import Dataset as tc_Dataset
from torch.utils.data.dataloader import DataLoader as tc_DataLoader

import pyro
import pyro.distributions.constraints as pr_const
import pyro.distributions as pr_dist
import pyro.infer as pr_infer
import pyro.optim as pr_optim

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class LogManager(logging.Logger):
    def __init__(self, name, level=logging.INFO, logstream: bool = True,
                 logfile: str = ""):
        super(LogManager, self).__init__(name)

        fmt = logging.Formatter(
            "{levelname: >8}|{asctime}|{name: >8}| {message}",
            style="{", datefmt="%Y-%m-%d,%H:%M:%S")
        if logstream:
            self._add_handler(logging.StreamHandler(), level, fmt)

        if logfile:
            self._add_handler(logging.FileHandler(logfile), level, fmt)

    def _add_handler(self, hdl, lvl, fmt):
        hdl.setLevel(lvl)
        hdl.setFormatter(fmt)
        self.addHandler(hdl)


class ExpDataSet(tc_Dataset):
    def __init__(self, X, y=None):
        super(ExpDataSet, self).__init__()
        self.x_mat = X
        self.y_vec = y

    def __len__(self):
        return self.x_mat.shape[0]

    def __getitem__(self, idx: int):
        x_vec = torch.Tensor(self.x_mat[idx, :]).unsqueeze(0)

        if self.y_vec is None:
            y_vec = torch.tensor(torch.nan)
        else:
            y_vec = torch.tensor(self.y_vec[idx])

        return x_vec, y_vec

    def __next__(self):
        for x in range(len(self)):
            yield self[x]


class ScCNN(tc_nn.Module):
    def __init__(self, fs: int = 32, cs: int = 2):
        super(ScCNN, self).__init__()

        assert fs % 2 == 0, "The fully-connected input size should be even."

        self.cv1 = tc_nn.Conv1d(1, 4, 3)
        self.cv2 = tc_nn.Conv1d(4, 8, 3)
        self.dp1 = tc_nn.Dropout(0.25)
        self.mp1 = tc_nn.MaxPool1d(2)
        self.fc1 = tc_nn.Linear(fs * 4 - 16, cs)

    def forward(self, x):
        x = self.cv1(x)
        x = tc_func.relu(x)

        x = self.cv2(x)
        x = tc_func.relu(x)

        x = self.mp1(x)
        x = self.dp1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = tc_func.relu(x)

        out = tc_func.softmax(x, dim=1)
        return out


class NNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate: float = 1e-5, neps: int = 100):
        super(NNClassifier, self).__init__()

        self.model = None
        self.learning_rate = learning_rate

        self.neps = neps

    @property
    def loss_func(self):
        return tc_nn.CrossEntropyLoss()

    @property
    def optimizer(self):
        return tc_optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y, batch_size: int = 32, shuffle=True):
        nr_features = X.shape[1]
        self.model = ScCNN(nr_features)

        dtld = tc_DataLoader(
            ExpDataSet(X, y), batch_size=batch_size, shuffle=shuffle
        )

        for bt_idx, (x_mat, y_true) in enumerate(dtld):
            y_pred = self.model(x_mat)
            y_true = y_true.type(torch.LongTensor)
            loss = self.loss_func(y_pred, y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X, pos_idx: int = 0, use_prob=False):
        dtld = tc_DataLoader(ExpDataSet(X), batch_size=1, shuffle=False)

        y_pprob, y_plabel = torch.Tensor(), torch.Tensor()

        if self.model is None:
            raise ValueError("Model wasn't ready yet, please use fit() first.")

        with torch.no_grad():
            for idx, (x_test, _) in enumerate(dtld):
                y_pred = self.model(x_test)

                y_pprob = torch.concat((y_pprob, y_pred[:, pos_idx]))
                y_plabel = torch.concat((y_plabel, y_pred.argmax(1)))

            if use_prob:
                return y_pprob.data.item()
            else:
                return y_plabel.data.numpy().astype("i8")

    def predict_proba(self, X, pos_idx: int = 0):
        return self.predict(X, pos_idx, True)


class PredEnsembler:
    """A class to ensemble the predicted probabilities."""
    def __init__(self, data, n_iters=5000,
                 logman: LogManager = LogManager("PredEnsembler")):
        self._logman = logman
        self._data = data
        self._n_iters = n_iters
        self._loss, self._alpha, self._beta, self._theta = [], [], [], []
        self._pp_val = 0
        
        self._guide = None

    @property
    def pp_val(self):
        return self._pp_val

    @property
    def theta(self):
        return self._theta[-1]

    @property
    def alpha(self):
        return self._alpha[-1]

    @property
    def beta(self):
        return self._beta[-1]

    # The model function
    def _model(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)

        a_init, b_init = torch.tensor([5, 5])
        alpha = pyro.param("alpha", a_init, constraint=pr_const.positive)
        beta = pyro.param("beta", b_init, constraint=pr_const.positive)

        with pyro.plate("obs", len(data)):
            return pyro.sample("theta", pr_dist.Beta(alpha, beta), obs=data)

    def infer(self, adam_param={"lr": 1e-3}):
        """Infer theta from given"""
        # The guide function
        self._guide = pr_infer.autoguide.AutoNormal(self._model)

        adam = pr_optim.Adam(adam_param) # Optimizer
        elbo = pr_infer.Trace_ELBO() # Trace by ELBO (evidence lower bound)

        # Stochastic Variational Inference
        svi = pr_infer.SVI(self._model, self._guide, adam, elbo)

        # Sampling
        for _ in range(self._n_iters):
            self._loss.append(svi.step(self._data)) # Collect loss

            alpha = pyro.param("alpha").item() # Collect alpha
            self._alpha.append(alpha)

            beta = pyro.param("beta").item() # Collect beta
            self._beta.append(beta)

        # The theta determined by the learned alpha and beta.
        self._theta = [a / (a + b) for a, b in zip(self._alpha, self._beta)]
    
    def cdf(self, x=0.5, n_points=10000): # Cumulative density/mass function
        beta = pr_dist.Beta(self.alpha, self.beta)
        x_space = torch.linspace(0, x, n_points)[1:-1]
        return torch.trapz(beta.log_prob(x_space).exp(), x_space)

    def report(self, pp=0.5, prefix="Report-", save_to="./"):
        # X-axis
        x_set = np.arange(self._n_iters)

        fig = plt.figure(figsize=(11, 12), constrained_layout=True)
        spec = fig.add_gridspec(3, 2)

        # Observation, theta, the most important parameters.
        ax_mean_lf = fig.add_subplot(spec[0, 0])
        ax_mean_lf.hist(self._data, alpha=0.25, color="green")
        ax_mean_lf.axvline(0.5, linestyle="--", color="gray", alpha=0.75)
        
        ## PP > pp, the pp is given by argument, e.g., 0.5
        self._pp_val = 100 * (1 - self.cdf().item())
        ax_mean_lf.set_title(f"$PP_{{{pp}}}$ = {self._pp_val:.2f}%")

        ## Sampling using the learned parameters
        pd_vi = pr_dist.Beta(self.alpha, self.beta)
        x_values = np.linspace(0, 1, num=1000)[1:-1]
        y_values = torch.exp(pd_vi.log_prob(torch.tensor(x_values)))

        ax_mean_lf_tx = ax_mean_lf.twinx()
        ax_mean_lf_tx.plot(x_values, y_values, color="blue")
        
        # Estimated theta per iteration
        ax_mean_rt = fig.add_subplot(spec[0, 1])
        ax_mean_rt.plot(x_set, self._theta)
        ax_mean_rt.axhline(0.5, linestyle="--", color="gray", alpha=0.75)
        ax_mean_rt.set_xlabel("Iterations")
        ax_mean_rt.set_ylabel("Estimated theta")
        ax_mean_rt.set_title("Estimated theta per iteration")

        # Parameter, alpha, beta
        ax_para_lf = fig.add_subplot(spec[1, 0])
        ax_para_lf.plot(x_set, self._alpha)
        ax_para_lf.plot(x_set, self._beta)
        ax_para_lf.set_xlabel("Iterations")
        ax_para_lf.set_ylabel("Learned alpha or beta value")
        ax_para_lf.set_title("Learned alpha and beta per step (diagnosis)")

        ax_para_rt = fig.add_subplot(spec[1, 1])
        ax_para_rt.hist(self._alpha, alpha=0.25)
        ax_para_rt.hist(self._beta, alpha=0.25)
        ax_para_rt.set_xlabel("Alpha or beta value")
        ax_para_rt.set_ylabel("Frequency")
        ax_para_rt.set_title("Learned alpha and beta per step bin (diagnosis)")
   
        # Loss
        ax_loss = fig.add_subplot(spec[2, :])
        ax_loss.plot(x_set, self._loss)
        ax_loss.set_xlabel("Number of iteration")
        ax_loss.set_ylabel("ELBO loss")
        ax_loss.set_title("ELBO loss (for diagnosis)")

        fig.savefig(f"{save_to}/{prefix}Diagnosis.png")


class CategoryEncoder(BaseEstimator):
    """A category variable transformer applied for whole DataFrame."""
    def __init__(self):
        self.categories_ = {}

    def _transformer(self, series: pd.Series, inverse=False):
        trans_dict = self.categories_.get(series.name, {})
        if not inverse:
            trans_dict = {v: k for k, v in trans_dict.items()}

        if trans_dict:
            return series.apply(lambda x: trans_dict.get(x, None))

        return series

    def fit(self, X: pd.DataFrame, y=None):
        _is_cat = X.dtypes == "object"
        self.categories_ = (
            X.loc[:, _is_cat]
            .reset_index(drop=True)
            .apply(lambda x: x.drop_duplicates().reset_index(drop=True))
            .to_dict()
        )

        return self

    def transform(self, X: pd.DataFrame, copy=None):
        if copy:
            return X.copy().apply(self._transformer)
        return X.apply(self._transformer)

    def inverse_transform(self, X, copy=None):
        if copy:
            return X.copy().apply(self._transformer, inverse=True)
        return X.apply(self._transformer, inverse=True)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def get_cli_opts():
    par = argparse.ArgumentParser()
    par.add_argument("-s", "--random-seed", default=31415, type=int,
                     help="Random seed. Default: %(default)s")
    par.add_argument("-P", "--proj-dir", default="Scpred",
                     help="The project dir containing train, explain, and predict results. Default: %(default)s")

    subpar = par.add_subparsers(dest="subcmd", required=True)

    trn_par = subpar.add_parser("train", help="Train a model from expression data.")
    trn_par.add_argument("-i", "--in-file", required=True,
                         help="Input file for training. Required")
    trn_par.add_argument("-c", "--config", required=True,
                         help="Configuration file. Required")
    trn_par.add_argument("-b", "--barcode-col", default="cellbarcodes",
                         help="The column used as the index column, i.e., cell barcodes. Default: %(default)s")
    trn_par.add_argument("-p", "--test-ratio", default=0.3, type=float,
                         help="Ratio of data used for test. Default: %(default)s")
    trn_par.add_argument("-I", "--n-iters", default=15, type=int,
                         help="Number of iterations used for the RandomizedSearchCV. Default: %(default)s")
    trn_par.add_argument("-x", "--cv-times", default=10, type=int,
                         help="Number of cross validation. Default: %(default)s")
    trn_par.add_argument("-f", "--force-features", default=None, nargs="*",
                         help="Features should be included by force. Default: %(default)s")
    trn_par.add_argument("-t", "--cell-types", nargs="*", default=None,
                         help="Cell types on which the model will be trained. Default: all")
    trn_par.add_argument("-m", "--model-arch", choices=["gbc", "nn"], default="gbc",
                         help="Which model architecture to be used. Default: %(default)s")
    trn_par.add_argument("-n", "--n-rows", default=None, type=int,
                         help="Number of rows to be read from the training in-file. Default: all")
    trn_par.add_argument("-@", "--n-jobs", default=1, type=int,
                         help="Number of jobs to be run in parallel. Default: %(default)s")

    exp_par = subpar.add_parser("explain", help="Explain a model by SHAP values.")
    exp_par.add_argument("-i", "--in-file", required=True,
                         help="The input data used for training. Required")
    exp_par.add_argument("-b", "--barcode-col", default="cellbarcodes",
                         help="The column used as the index column, i.e., cell barcodes. Default: %(default)s")
    exp_par.add_argument("-t", "--cell-types", nargs="*", default=None,
                         help="Cell types on which the model will be trained. Default: all")
    exp_par.add_argument("-n", "--n-rows", default=None, type=int,
                         help="Number of rows to be used for the explanation. Default: all")

    prd_par = subpar.add_parser("predict", help="Predict unseen expression data.")
    prd_par.add_argument("-i", "--in-file", required=True,
                         help="Input file to be predict. Required.")
    prd_par.add_argument("-p", "--pos-label", default=1, type=int,
                         help="The positive label. Default: %(default)s")
    prd_par.add_argument("-b", "--barcode-col", default="cellbarcodes",
                         help="The column used as the index column, i.e., cell barcodes. Default: %(default)s")
    prd_par.add_argument("-t", "--cell-types", nargs="*", default=None,
                         help="Cell types on which the model will be trained. Default: all")
    prd_par.add_argument("-n", "--n-rows", default=None, type=int,
                         help="Number of rows to be read from the training in-file. Default: all")
    prd_par.add_argument("-N", "--n-iters", default=5000, type=int,
                         help="Sampling times. Default: %(default)s")
    prd_par.add_argument("-o", "--out-subdir", default=None,
                         help="The file name prefix used to save prediction results.")

    return par.parse_args()


def setup(options):
    proj_dir = options.proj_dir
    seed = options.random_seed

    for sub_dir in ["Train", "Explain", "Predict"]:
        os.makedirs(f"{proj_dir}/{sub_dir}", exist_ok=True)

    torch.manual_seed(seed) # Pyro use Pytorch's random state.
    np.random.seed(seed)  # Scikit-learn uses Numpy's random state.


def load_expmat(fpath, label_order=None, as_train=True, min_pct=0.2,
                cell_types=None, test_ratio=None, features=None, **kwarg):
    """Load expression matrix."""
    ct_col, id_col, lb_col = "CellType", "SampleID", "SampleLabel"
    res_cols = [lb_col, id_col, ct_col]

    exp_tab: pd.DataFrame = pd.read_csv(fpath, header=0, **kwarg)

    # If there is a 'CellType' column, it allows the function to subset cell
    # types for the downstream analysis.
    # TODO: Give warns if items from cell_types aren't in the CellType column.
    if ct_col in exp_tab.columns and cell_types:
        kept_recs = [x in cell_types for x in exp_tab.loc[:, ct_col]]
        exp_tab = exp_tab.loc[kept_recs, :]

    # If the min_pct argument is greater than 0 but less equal than 1, genes
    # expressed less than min_pct will be discarded for downstream analyses.
    # FIXME Genes lowly expressed in samples A could be filtered out.
    # TODO Logics to filter category columns by counting missing values.
    if 0 <= min_pct <= 1:
        n_cell, _ = exp_tab.shape
        cat_cols = exp_tab.columns[exp_tab.dtypes == "object"].to_list()
        cat_cols = [x for x in cat_cols if x not in res_cols]
        kept = (exp_tab.drop(labels=res_cols+cat_cols, axis=1, errors="ignore")
                .ne(0).sum(0).div(n_cell).ge(min_pct))

        kept_cols = kept.index[kept].to_list() + cat_cols + res_cols
        exp_tab = exp_tab.loc[:, kept_cols]

    # If there is a 'SampleID' column, it's a multi-sample dataset. However,
    # the reality should also depend on the number of samples in the column.
    cts_map = None # cell to sample map.
    if id_col in exp_tab.columns:
        cts_map = exp_tab.loc[:, id_col]
        n_samples = cts_map.drop_duplicates().shape[0]
        if n_samples <= 1: # One or empty.
            cts_map = None

    # Select a subset of features. The logics also deal with missing features
    # in unseen data, i.e., data to be predicted.
    if features and isinstance(features, list):
        exist_features = exp_tab.columns.to_list()
        # Keep features existing in the exp_tab.
        tar_features = [x for x in features if x in exist_features]
        exp_tab = exp_tab.loc[:, tar_features]

        # Assign 1 to features not existing in the exp_tab.
        mis_features = [x for x in features if x not in exist_features]
        exp_tab.loc[:, mis_features] = 1 # FIXME way to handle missing features

        exp_tab = exp_tab.loc[:, features] # Ensure the order of input features

    # If there is a 'SampleLabel' column, it's a training dataset, otherwise,
    # it's a dataset to be predicted. When 'SampleLabel' is in the dataset,
    # `as_train=False` can be used to indicate the dataset is for training.
    xmat_tn, xmat_tt, yvec_tn, yvec_tt = [None] * 4
    if lb_col in exp_tab.columns:
        if label_order is None: # Encode the labels
            label_order = exp_tab.loc[:, lb_col].unique()
        lab_encoder = LabelEncoder().fit(label_order)

        yvec = lab_encoder.transform(exp_tab.loc[:, lb_col])
        xmat = exp_tab.drop(labels=res_cols, axis=1, errors="ignore")

        if test_ratio is None or test_ratio <= 0 or not as_train:
            xmat_tn, yvec_tn = xmat, yvec
        else:
            splits = train_test_split(
                xmat, yvec, test_size=test_ratio, stratify=yvec
            )
            xmat_tn, xmat_tt, yvec_tn, yvec_tt = splits
    else:
        xmat_tn = exp_tab

    return xmat_tn, xmat_tt, yvec_tn, yvec_tt, cts_map


def load_model(fpath, fmt="pickle"):
    with open(fpath, "rb") as fhand:
        ext = os.path.splitext(fpath)
        if ext in ["pkl", "pickle"] or fmt in ["pkl", "pickle"]:
            return pkl.load(fhand)
        elif ext in ["jbl", "joblib"] or fmt in ["jbl", "joblib"]:
            return jbl.load(fpath)

    raise ValueError("Unknown model format.")


def parse_config(fpath):
    conf = None
    with open(fpath, "r") as fhandle:
        conf = json.load(fhandle)

    return conf["pos_label"], conf["label_order"], conf["param_space"]


def ensemble_probs(pred_pc, save_to, **kwargs):
    pe_report = pd.DataFrame(
        columns=["SampleID", "ppval", "theta", "alpha", "beta"]
    )

    for pid in pred_pc.loc[:, "SampleID"].unique():
        cur_probs = pred_pc.query(f"SampleID == '{pid}'").loc[:, "y_prob"]
        pe = PredEnsembler(cur_probs, **kwargs)
        pe.infer()
        pe.report(prefix=f"{pid}-", save_to=save_to)

        pe_report.loc[pid] = pd.Series(
            dict(SampleID=pid, ppval=pe.pp_val, theta=pe.theta, alpha=pe.alpha,
                 beta=pe.beta)
        )

    (pe_report
     .reset_index(drop=True)
     .to_csv(f"{save_to}/Prediction-persample.csv", index=False))


def eval_model(xmat, y_true, model, pos_idx=1, cts_map=None, save_to="./"):
    # Best meta-parameters by the RandomizedSearchCV()
    bst_par = model.best_params_
    with open(f"{save_to}/Best_params.json", "w") as bphandle:
        json.dump(bst_par, bphandle, indent=2)

    # Test the model
    y_pred = model.predict(xmat)
    y_prob = model.predict_proba(xmat)[:, pos_idx]

    # Save the selected features.
    xmat.columns.to_frame().to_csv(
        f"{save_to}/Selected_features.csv", index=False, header=["Feature"]
    )

    # Save the predicted results.
    pred_pc = xmat.assign(
        SampleID=cts_map, y_true=y_true, y_pred=y_pred, y_prob=y_prob
    )
    pred_pc.to_csv(f"{save_to}/Prediction-percell.csv", index=False)

    # Infer expected probability per sample, TODO: put it into a function.
    ensemble_probs(pred_pc, save_to)

    # Evaluation matrices, plain text
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)

    with open(f"{save_to}/Evaluation_metrics.txt", mode="w") as ofhand:
        ofhand.write(f"Precision: {precision:0.3f}\n"
                     f"Accuracy: {accuracy:0.3f}\n"
                     f"ROC AUC: {roc_auc:0.3f}\n"
                     f"Recall: {recall:0.3f}\n")

    # ROC and PR curve
    fig, (roc_axe, prc_axe) = plt.subplots(ncols=2)

    ## ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_idx)
    roc_axe.plot(fpr, tpr)
    roc_axe.plot([0, 1], [0, 1], linestyle="dashed")
    roc_axe.text(0.75, 0.25, "{:.3f}".format(roc_auc))
    roc_axe.set_title("ROC curve (test)")
    roc_axe.set_xlabel("False positive rate")
    roc_axe.set_ylabel("True positive rate")
    roc_axe.set_xlim(-0.05, 1.05)
    roc_axe.set_ylim(-0.05, 1.05)

    ## PR curve
    pre, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_idx)
    prc_axe.plot(rec, pre)
    prc_axe.set_title("Precision-recall curve (test)")
    prc_axe.set_xlabel("Recall")
    prc_axe.set_ylabel("Precision")
    prc_axe.set_xlim(-0.05, 1.05)
    prc_axe.set_ylim(-0.05, 1.05)

    ## Save the figure
    fig.set_figwidth(12)
    fig.set_figheight(6)
    fig.set_tight_layout(True)
    fig.savefig(f"{save_to}/PR_and_ROC_curve.pdf")


def train(options, logman: LogManager = LogManager("Train")):
    """Train a model"""
    logman.info("Train a model on given inputs.")

    # CLI Options
    test_ratio = options.test_ratio
    cell_types = options.cell_types
    bcode_col = options.barcode_col
    cv_times = options.cv_times
    proj_dir = options.proj_dir
    in_file = options.in_file
    n_iters = options.n_iters
    n_jobs = options.n_jobs
    n_rows = options.n_rows
    model_arch = options.model_arch

    # IO
    save_to = f"{proj_dir}/Train"

    # Parameters for the training
    pos_lab, lab_order, param_space = parse_config(options.config)
    pos_idx = lab_order.index(pos_lab)

    # Load expression matrix
    x_tn, x_tt, y_tn, y_tt, cts_map = load_expmat(
        in_file, lab_order, test_ratio=test_ratio, index_col=bcode_col,
        cell_types=cell_types, nrows=n_rows
    )

    # A pipeline for scaling data, selecting features, classifying samples.
    pipe_steps = [("encode", CategoryEncoder()),
                  ("scale", StandardScaler()),
                  ("select", SelectKBest(f_classif))]

    if model_arch == "nn":
        pipe_steps.append(("classify", NNClassifier()))
    elif model_arch == "rfc":
        pipe_steps.append(("classify", RandomForestClassifier()))
    else:
        if model_arch != "gbc":
            logman.warning(f"Unsupported {model_arch}, using gbc by default.")
        pipe_steps.append(("classify", GradientBoostingClassifier()))

    pipe = Pipeline(steps=pipe_steps)

    # Searching the best hyper-parameters randomly.
    rscv = RandomizedSearchCV(
        pipe, param_space, n_iter=n_iters, n_jobs=n_jobs, cv=cv_times
    )
    rscv.fit(x_tn, y_tn)

    # Evaluate the model
    eval_model(x_tt, y_tt, rscv, pos_idx, cts_map, save_to)

    # Save the model
    with open(f"{save_to}/Model.pickle", "bw") as mf_handle:
        pkl.dump(rscv, mf_handle)

    logman.info(f"Finished! Check {save_to} for the results.")


def pipe_last_step(pipe, xmat):
    pbe = pipe.best_estimator_

    encoded = pbe["encode"].transform(xmat)
    scaled = pbe["scale"].transform(encoded)
    selected = pd.DataFrame(
        pbe["select"].transform(scaled), index=xmat.index,
        columns=xmat.columns[pbe["select"].get_support()]
    )

    return pbe["classify"], selected


def explain(options, logman: LogManager = LogManager("Explain")):
    logman.info("Explain the model by SHAP values.")

    # CLI options
    expmat_path = options.in_file
    cell_types = options.cell_types
    bcode_col = options.barcode_col
    proj_dir = options.proj_dir
    n_rows = options.n_rows

    # IO
    save_to = f"{proj_dir}/Explain"

    # Load model and expression matrix
    model = load_model(f"{proj_dir}/Train/Model.pickle")
    xmat, _, _, _, _ = load_expmat(
        expmat_path, index_col=bcode_col, cell_types=cell_types, nrows=n_rows
    )

    # Plot SHAP values. Including bar and dot (beeswarm) plot
    pbe, xmat = pipe_last_step(model, xmat)
    explainer = sp.TreeExplainer(pbe)
    shap_vals = explainer(xmat)

    for ptype in ["dot", "bar"]:
        plt.clf() # We clear the figure.

        sp.summary_plot(shap_vals, plot_type=ptype, show=False)
        fig = plt.gcf()
        if ptype == "dot":
            _, axe_cb = fig.get_axes()
            axe_cb.set_visible(False)
            axe_cb = plt.colorbar()

        fig.set_figwidth(7)
        fig.set_figheight(7)
        fig.set_tight_layout(True)
        fig.savefig(f"{save_to}/{ptype.title()}_plot.pdf")

    logman.info(f"Finished! Check {save_to} for the results.")


def predict(options, logman: LogManager = LogManager("Predict")):
    logman.info("Predict unseen samples.")

    # CLI options
    expmat_path = options.in_file
    cell_types = options.cell_types
    bcode_col = options.barcode_col
    pos_label = options.pos_label
    proj_dir = options.proj_dir
    n_iters = options.n_iters
    n_rows = options.n_rows
    out_subdir = options.out_subdir

    # Create sub-dir to store the predicted results.
    time_stamp = dt.now().strftime("%Y%m%d%H%M%S")
    if out_subdir is None:
        out_subdir = f"{proj_dir}/Predict/{time_stamp}"
    os.makedirs(out_subdir, exist_ok=True)

    # Load features used in the model
    tar_features = f"{proj_dir}/Train/Selected_features.csv"
    features = pd.read_csv(tar_features, header=0).loc[:, "Feature"].to_list()

    # Load model and expression matrix
    model_path = f"{proj_dir}/Train/Model.pickle"
    model = load_model(model_path)
    xmat, _, _, _, cts_map = load_expmat(
        expmat_path, as_train=False, cell_types=cell_types, min_pct=0,
        features=features, index_col=bcode_col, nrows=n_rows
    )

    # Prediction
    y_pred = model.predict(xmat)
    y_prob = model.predict_proba(xmat)[:, pos_label]

    # Save predicted results
    pred_pc = xmat.assign(SampleID=cts_map, y_prob=y_prob, y_pred=y_pred)
    pred_pc.to_csv(f"{out_subdir}/Prediction-percell.csv")

    # Infer expected probability per sample
    ensemble_probs(pred_pc, out_subdir, n_iters=n_iters)

    logman.info(f"Done! Check {out_subdir} for the results")


def main(logman: LogManager = LogManager("Scpred")):
    # Get the CLI options
    options = get_cli_opts()

    # Set up project dir, random state, etc.
    setup(options)

    # Sub-command
    subcmd = options.subcmd
    if subcmd == "train":
        train(options)
    elif subcmd == "explain":
        explain(options)
    elif subcmd == "predict":
        predict(options)
    else:
        logman.error("Unknown sub-command")


if __name__ == "__main__":
    main()
