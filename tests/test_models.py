import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from california_models import RegressionExperiment, check_metric, save_model
from wine_models import text_pipeline


@pytest.mark.parametrize('transformed', [False, True])
def test_california_full_workflow(tmp_path, monkeypatch, transformed):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(1, 5, (60, 3)), columns=list('abc'))
    frame['price'] = frame.a * 3 + frame.b + 10
    frame.loc[0, 'b'] = np.nan
    train, test = frame.iloc[:48], frame.iloc[48:]
    exp = RegressionExperiment().setup(train, 'price', fold=2, smoke=True,
                                       normalize=transformed, transformation=transformed)
    models = exp.compare_models()
    assert len(models) == 5
    models += [exp.ensemble_model(models[0]), exp.tune_model(models[1]),
               exp.blend_models(models[1:4]), exp.stack_models(models[1:4])]
    for model in models:
        prediction = exp.predict_model(model, test)
        assert np.isfinite(check_metric(test.price, prediction.prediction_label))
    monkeypatch.chdir(tmp_path)
    model = models[-1]
    restored = joblib.load(save_model(model, 'smoke'))
    np.testing.assert_allclose(restored.predict(test.drop(columns='price')), model.predict(test.drop(columns='price')))


def test_text_pipeline_fold_fit_and_roundtrip(tmp_path):
    texts = ['red fruit berry', 'white citrus fresh', 'red oak dry', 'white light pear'] * 5
    target = np.array([20., 10., 30., 12.] * 5)
    pipeline = text_pipeline(Ridge(), ['the'])
    assert np.isfinite(cross_val_score(pipeline, texts, target, cv=2)).all()
    pipeline.fit(texts, target)
    assert 'unseen' not in pipeline['tfidf'].vocabulary_
    path = tmp_path / 'wine.joblib'
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(joblib.load(path).predict(['unseen berry']), pipeline.predict(['unseen berry']))


def test_notebook_helpers_use_selected_pipeline_even_without_display():
    nb = json.loads(Path('NLP_wine_reviews.ipynb').read_text(encoding='utf8'))
    namespace = {}
    from sklearn.model_selection import RandomizedSearchCV, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    namespace.update(locals())
    namespace.update(pd=pd, text_pipeline=text_pipeline, rnd=42, stopwords=[], SMOKE=False)
    train = pd.DataFrame({'description': ['berry red', 'citrus white', 'oak red', 'pear white'] * 4,
                          'price': [20, 10, 30, 12] * 4})
    namespace.update(train=train, test=train.copy(), y_train=train.price, y_test=train.price)
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if 'def simple_model(' in source or 'def rand(' in source:
            exec(source, namespace)
    simple = namespace['simple_model']('Ridge', Ridge(), cv=2, show=False, back=True)
    best = namespace['rand']('Ridge', Ridge(), {'alpha': [1, 2]}, iter=2, cv=2, show=False, back=True)
    assert namespace['models'][('tf-idf', 'simple', 'Ridge')] is simple
    assert namespace['models'][('tf-idf', 'tuned', 'Ridge')] is best
    assert np.isfinite(best.predict(['berry'])).all()


def test_lemmatized_pipeline_serializes_its_preprocessor(tmp_path):
    import nltk
    nltk.data.path.insert(0, str(Path('data/nltk_data').resolve()))
    from wine_models import lemmatize_texts
    # prepare_nltk.py is an explicit prerequisite, including in CI.
    assert lemmatize_texts(['berries and wines'])[0] == 'berry and wine'
    pipe = text_pipeline(Ridge(), [], lemm=True)
    texts = ['berries and wines', 'fresh apples', 'dry wines', 'ripe pears'] * 3
    pipe.fit(texts, [20, 10, 30, 12] * 3)
    path = tmp_path / 'lemma.joblib'
    joblib.dump(pipe, path)
    np.testing.assert_allclose(joblib.load(path).predict(texts), pipe.predict(texts))
