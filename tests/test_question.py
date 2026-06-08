import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    fetch_news_data, load_news_data, explore_data, encode_labels,
    split_data, build_bow_pipeline, train_model, evaluate_model,
    predict_article, top_fake_words, top_real_words,
    compare_bow_vs_tfidf, predict_batch, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar indirme/eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data_path():
    """İlk testte internet'ten indirir (~30MB), sonraki tüm testler cache'den okur."""
    return fetch_news_data()


@pytest.fixture(scope="module")
def raw_df(data_path):
    return load_news_data(data_path)


@pytest.fixture(scope="module")
def split(raw_df):
    df = encode_labels(raw_df)
    return split_data(df['text'], df['target'])


@pytest.fixture(scope="module")
def trained_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_bow_pipeline()
    return train_model(pipe, X_train, y_train)


# 1. fetch_news_data
def test_fetch_news_data_returns_path(data_path):
    assert isinstance(data_path, str)
    assert os.path.exists(data_path)
    assert data_path.endswith('fake_or_real_news.csv')


# 2. load_news_data
def test_load_news_data_shape(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    # ~6335 satır
    assert 6000 < raw_df.shape[0] < 6700
    assert list(raw_df.columns) == ['title', 'text', 'label']


# 3. load — label değerleri
def test_load_news_data_labels(raw_df):
    assert set(raw_df['label'].unique()) == {'REAL', 'FAKE'}


# 4. explore_data
def test_explore_data_structure(raw_df):
    info = explore_data(raw_df)
    assert set(info.keys()) >= {'total', 'real_count', 'fake_count', 'fake_rate'}
    assert info['total'] == len(raw_df)
    assert info['real_count'] + info['fake_count'] == info['total']
    # ~dengeli %50/50
    assert 0.45 < info['fake_rate'] < 0.55


# 5. encode_labels
def test_encode_labels_binary(raw_df):
    df = encode_labels(raw_df)
    assert 'target' in df.columns
    assert set(df['target'].unique()) == {0, 1}
    assert df['target'].dtype in (np.int64, np.int32, int)
    # FAKE → 1, REAL → 0
    assert df[df['label'] == 'FAKE']['target'].iloc[0] == 1
    assert df[df['label'] == 'REAL']['target'].iloc[0] == 0


# 6. split_data
def test_split_data_stratified(raw_df):
    df = encode_labels(raw_df)
    X_train, X_test, y_train, y_test = split_data(df['text'], df['target'])
    # 80/20
    assert abs(len(X_train) / len(df) - 0.8) < 0.01
    # Stratify: train ve test fake oranı yakın olmalı
    assert abs(y_train.mean() - y_test.mean()) < 0.01


# 7. build_bow_pipeline
def test_build_bow_pipeline_type():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    pipe = build_bow_pipeline()
    assert isinstance(pipe, Pipeline)
    names = dict(pipe.steps)
    assert 'cv' in names and 'nb' in names
    # Bag of Words → CountVectorizer (TF-IDF DEĞİL)
    assert isinstance(names['cv'], CountVectorizer)
    assert isinstance(names['nb'], MultinomialNB)
    assert names['cv'].stop_words == 'english'


# 8. train_model
def test_train_model_fits(trained_pipeline, split):
    X_train, X_test, y_train, y_test = split
    preds = trained_pipeline.predict(X_test[:5])
    assert len(preds) == 5


# 9. evaluate_model
def test_evaluate_model(trained_pipeline, split):
    _, X_test, _, y_test = split
    res = evaluate_model(trained_pipeline, X_test, y_test)
    assert set(res.keys()) >= {
        'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
    }
    assert 0 <= res['accuracy'] <= 1
    assert res['confusion_matrix'].shape == (2, 2)
    # Bag of Words + NB sahte haberde güçlü baseline
    assert res['f1'] > 0.85


# 10. predict_article — fake
def test_predict_article_fake(trained_pipeline):
    out = predict_article(
        trained_pipeline,
        "SHOCKING TRUTH they don't want you to know! The corrupt establishment "
        "elites are hiding this secret. Share now before they delete it!",
    )
    assert set(out.keys()) >= {'prediction', 'label', 'fake_probability'}
    assert 0 <= out['fake_probability'] <= 1
    # Sansasyonel/komplo tonlu metin fake yönünde olmalı
    assert out['fake_probability'] > 0.5


# 11. predict_article — real
def test_predict_article_real(trained_pipeline):
    out = predict_article(
        trained_pipeline,
        "The Senate committee held a hearing on Wednesday to review the proposed "
        "budget. Officials said the legislation would be debated next month.",
    )
    assert set(out.keys()) >= {'prediction', 'label', 'fake_probability'}
    # Ölçülü/olgusal haber real yönünde olmalı
    assert out['fake_probability'] < 0.5


# 12. top_fake_words
def test_top_fake_words(trained_pipeline):
    words = top_fake_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    assert all(isinstance(w, str) for w in words)


# 13. top_real_words
def test_top_real_words(trained_pipeline):
    words = top_real_words(trained_pipeline, n=10)
    assert isinstance(words, list)
    assert len(words) == 10
    # Real ve fake kelimeleri farklı olmalı
    fake_words = top_fake_words(trained_pipeline, n=10)
    assert set(words) != set(fake_words)


# 14. compare_bow_vs_tfidf
def test_compare_bow_vs_tfidf(split):
    X_train, X_test, y_train, y_test = split
    cmp = compare_bow_vs_tfidf(X_train, X_test, y_train, y_test)
    assert set(cmp.keys()) >= {'bow_f1', 'tfidf_f1'}
    # İkisi de iyi performans göstermeli; BoW güçlü baseline
    assert cmp['bow_f1'] > 0.8
    assert cmp['tfidf_f1'] > 0.8


# 15. predict_batch
def test_predict_batch(trained_pipeline):
    articles = [
        "BREAKING! You won't believe what the secret cabal of elites is hiding "
        "from you. Wake up sheeple, the truth is out there, share before deleted!",
        "The mayor announced a new public transit plan on Monday, with funding "
        "approved by the city council for the next fiscal year.",
    ]
    out = predict_batch(trained_pipeline, articles)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all('prediction' in o for o in out)
    # İlki fake yönünde, ikincisi real yönünde
    assert out[0]['fake_probability'] > out[1]['fake_probability']


# 16. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'fake_rate', 'test_f1', 'top_fake_word',
        'sample_fake_pred', 'sample_real_pred'
    }
    assert result['test_f1'] > 0.85
    assert 0.45 < result['fake_rate'] < 0.55
    # Örnek tahminler doğru yönde
    assert result['sample_fake_pred']['fake_probability'] > \
        result['sample_real_pred']['fake_probability']
    # En fake kelime mantıklı (boş olmayan string)
    assert isinstance(result['top_fake_word'], str)
    assert len(result['top_fake_word']) > 0


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
