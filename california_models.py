"""Small explicit sklearn replacement for the regression operations used here.

Each estimator carries its preprocessing. Selection/tuning use training-only CV;
the externally supplied test frame is never used for fitting or early stopping.
"""
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class RegressionExperiment:
    def setup(self, data, target, session_id=42, normalize=False,
              transformation=False, fold=5, smoke=False):
        self.target = target
        self.X = data.drop(columns=target)
        self.y = data[target]
        if len(data) < fold:
            raise ValueError('Training rows must be at least the number of folds')
        self.seed, self.smoke = session_id, smoke
        self.cv = KFold(fold, shuffle=True, random_state=session_id)
        self.preprocess = [('impute', SimpleImputer())]
        if transformation:
            self.preprocess.append(('power', PowerTransformer()))
        if normalize:
            self.preprocess.append(('scale', StandardScaler()))
        return self

    def compare_models(self, n_select=5):
        trees = 5 if self.smoke else 100
        models = [RandomForestRegressor(n_estimators=trees, random_state=self.seed),
                  LGBMRegressor(n_estimators=trees, random_state=self.seed, verbosity=-1, n_jobs=1),
                  ExtraTreesRegressor(n_estimators=trees, random_state=self.seed),
                  GradientBoostingRegressor(n_estimators=trees, random_state=self.seed),
                  DecisionTreeRegressor(random_state=self.seed)]
        ranked = []
        for model in models:
            pipe = Pipeline(self.preprocess + [('model', model)])
            score = cross_val_score(pipe, self.X, self.y, cv=self.cv,
                                    scoring='neg_mean_absolute_percentage_error', error_score='raise').mean()
            ranked.append((score, pipe))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [model.fit(self.X, self.y) for _, model in ranked[:n_select]]

    def ensemble_model(self, model):
        result = BaggingRegressor(estimator=clone(model), n_estimators=3 if self.smoke else 10,
                                  random_state=self.seed)
        return result.fit(self.X, self.y)

    def tune_model(self, model):
        # Tune the actual selected estimator, never assume rank two is LightGBM.
        params = model.get_params()
        grid = {'model__max_depth': [2, 4, 8, None]}
        if isinstance(model[-1], LGBMRegressor):
            grid = {'model__max_depth': [2, 4, 8, -1]}
        if 'model__n_estimators' in params:
            grid['model__n_estimators'] = [5, 10] if self.smoke else [50, 100, 200]
        search = RandomizedSearchCV(clone(model), grid, n_iter=2 if self.smoke else 8,
                                    cv=self.cv, scoring='neg_mean_absolute_percentage_error',
                                    random_state=self.seed, error_score='raise')
        return search.fit(self.X, self.y).best_estimator_

    def blend_models(self, models):
        return VotingRegressor([(str(i), clone(m)) for i, m in enumerate(models)]).fit(self.X, self.y)

    def stack_models(self, models):
        return StackingRegressor([(str(i), clone(m)) for i, m in enumerate(models)],
                                 final_estimator=Ridge(), cv=self.cv).fit(self.X, self.y)

    def predict_model(self, model, data):
        frame = data.copy()
        frame['prediction_label'] = model.predict(frame[self.X.columns])
        return frame


def check_metric(actual, prediction, metric='MAPE'):
    if metric.upper() != 'MAPE':
        raise ValueError('Only MAPE is used by this experiment')
    return mean_absolute_percentage_error(actual, prediction)


def save_model(model, name):
    path = Path('artifacts') / (name + '.joblib')
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    return path
