from collections import defaultdict
import os
import numpy as np
import pandas as pd
from mlxtend.evaluate import mcnemar, mcnemar_table
from sklearn.base import (BaseEstimator, MetaEstimatorMixin, clone, is_classifier)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (GridSearchCV, KFold, ParameterGrid, StratifiedKFold, cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from tqdm.autonotebook import trange
from .misc import mode

class CrossValidator(BaseEstimator, MetaEstimatorMixin):

    def __init__( self, estimator, param_grid=None, scoring="accuracy", refit=True, coef_func=None, n_folds=5, n_trials=20, n_jobs=1, verbose=0, feature_names=None ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.coef_func = coef_func
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.feature_names = feature_names

    def fit(self, X, y=None):

        self.do_gs = bool(self.param_grid)
        self.multi_score = not isinstance(self.scoring, str)
        self._prep_results(X, y)
        for i in trange(self.n_trials):
            outer_cv, inner_cv = self._get_cv(random_state=i)
            dummy_results = self._get_dummy_results(X, y, outer_cv)
            if self.do_gs:
                estimator = self._do_gridsearch(X, y, inner_cv)
                self._store_cv_results(estimator, i)
                ct_jobs = 1
            else:
                estimator = self.estimator
                ct_jobs = self.n_jobs
            ct_results_tmp = cross_validate(estimator,
                                            X, y,
                                            scoring=self.scoring,
                                            cv=outer_cv,
                                            return_estimator=True,
                                            return_train_score=True,
                                            n_jobs=ct_jobs,
                                            verbose=self.verbose)
            self._store_ct_results(X, y, i,
                                   outer_cv,
                                   dummy_results,
                                   ct_results_tmp)

        
        return self

    def _prep_results(self, X, y):
        if self.do_gs:
            self.param_results_ = defaultdict(list)
            self.cv_results_ = defaultdict(list)
            for param in ParameterGrid(self.param_grid):
                for name, val in param.items():
                    self.cv_results_[f"param_{name}"].append(val)
        
        self.ct_results_ = defaultdict(list)
        self.predictions_ = defaultdict(
            lambda: np.zeros((self.n_trials, len(y))))

        if self.coef_func:
            self.coefs_ = np.zeros((self.n_trials, X.shape[1]))

    def _get_cv(self, random_state=None):

        if is_classifier(self.estimator):
            outer_cv = StratifiedKFold(
                self.n_folds, shuffle=True, random_state=random_state)
            inner_cv = StratifiedKFold(
                self.n_folds, shuffle=True, random_state=random_state)
        else:
            outer_cv = KFold(self.n_folds, shuffle=True,
                             random_state=random_state)
            inner_cv = KFold(self.n_folds, shuffle=True,
                             random_state=random_state)

        return outer_cv, inner_cv

    def _get_dummy_results(self, X, y, outer_cv):
        if is_classifier(self.estimator):
            dummy_results = cross_val_predict(
                DummyClassifier(), X, y, cv=outer_cv)
        else:
            dummy_results = cross_val_predict(
                DummyRegressor(), X, y, cv=outer_cv)

        return dummy_results

    def _do_gridsearch(self, X, y, cv):

        gridsearch = GridSearchCV(self.estimator,
                                  param_grid=self.param_grid,
                                  scoring=self.scoring,
                                  refit=self.refit,
                                  return_train_score=True,
                                  cv=cv,
                                  verbose=self.verbose,
                                  n_jobs=self.n_jobs)

        gridsearch.fit(X, y)

        return gridsearch

    def _store_cv_results(self, gridsearch, i):
        if self.multi_score:
            for score in self.scoring:
                self.cv_results_[f"train_{score}_{i}"] = gridsearch.cv_results_[
                    f"mean_train_{score}"]
                self.cv_results_[f"test_{score}_{i}"] = gridsearch.cv_results_[
                    f"mean_test_{score}"]
        else:
            self.cv_results_[f"train_score_{i}"] = gridsearch.cv_results_[
                f"mean_train_score"]
            self.cv_results_[f"test_score_{i}"] = gridsearch.cv_results_[
                f"mean_test_score"]
        for name, val in gridsearch.best_params_.items():
            self.param_results_[name].append(val)

    def _store_ct_results(self, X, y, i, outer_cv, dummy_results, ct_results_tmp):
        if self.multi_score:
            for score in self.scoring:
                self.ct_results_[f"train_{score}"].append(ct_results_tmp[f"train_{score}"].mean())
                self.ct_results_[f"test_{score}"].append(ct_results_tmp[f"test_{score}"].mean())
        else:
            self.ct_results_[f"train_score"].append(ct_results_tmp["train_score"].mean())
            self.ct_results_[f"test_score"].append(ct_results_tmp["test_score"].mean())

        self.ct_results_[f"fit_time"].append(ct_results_tmp["fit_time"].mean())
        self.ct_results_[f"predict_time"].append(ct_results_tmp["score_time"].mean())

        if self.coef_func:
            coef_tmp = np.zeros((self.n_folds, X.shape[1]))

        for j, (train, test) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train], X[test]

            current_estimator = ct_results_tmp["estimator"][j]
            if self.do_gs:
                current_estimator = current_estimator.best_estimator_

            if self.coef_func:
                coef_tmp[j,:] = self.coef_func(current_estimator).squeeze()
            y_pred = current_estimator.predict(X_test)
            self.predictions_["y_pred"][i, test] = y_pred

            if hasattr(current_estimator, "decision_function"):
                conf_scores = current_estimator.decision_function(X_test)
                self.predictions_["conf_scores"][i, test] = conf_scores

            if hasattr(current_estimator, "predict_proba"):
                proba = current_estimator.predict_proba(X_test)[:,1]
                self.predictions_["probability"][i, test] = proba
            
        mcn_table = mcnemar_table(y.ravel(),
                                  dummy_results.ravel(),
                                  self.predictions_["y_pred"][i].ravel())
        _, p_val = mcnemar(mcn_table)
        self.ct_results_["p_value"].append(p_val)
        if self.coef_func:
            self.coefs_[i,:] = coef_tmp.mean(axis=0)                                          
    
    def _fit_final_estimator(self, X, y):
        self.estimator_ = clone(self.estimator)

        if self.do_gs:
            params_final_model = {}

            for param, vals in self.param_results_.items():
                if np.asarray(vals).dtype == float:
                    val_final = np.mean(vals)
                else:
                    val_final = mode(vals)
                params_final_model[param] = val_final
            self.estimator_.set_params(**params_final_model)
        self.estimator_.fit(X, y)



