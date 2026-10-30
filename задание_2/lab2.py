#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 2 «Умный поиск по документам»
Вариант 1 — Ботанический справочник (другой вариант: python lab2.py 03_Кулинарная_энциклопедия).

Запуск:   python lab2.py [папка_варианта] [папка_вывода]
Выход:    search_results.csv, experiment_metrics.csv, baseline_results.csv,
          error_analysis.txt, metrics_comparison.png

Стек: pandas, numpy, re, nltk (только список стоп-слов), pymorphy3, scikit-learn, matplotlib.
Метрики считаются ТОЛЬКО на 8 размеченных (dev) запросах; параметры по ним не подбираются.
"""

# ПУНКТ 1. ИМПОРТЫ, КОНФИГУРАЦИЯ ОКРУЖЕНИЯ И ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_K = 5
ENC = "utf-8-sig"  # UTF-8 with BOM

# ------------------------------------------------------------------ 0. Стоп-слова
try:
    from nltk.corpus import stopwords
    try:
        STOP_WORDS = set(stopwords.words("russian"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("russian"))
except Exception:  # запасной вариант, если nltk/корпус недоступны
    STOP_WORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
    ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
    нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
    сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь
    этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть
    после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между""".split())
# нормализуем «ё» так же, как в тексте
STOP_WORDS = {w.replace("ё", "е") for w in STOP_WORDS}

MORPH = pymorphy3.MorphAnalyzer()
_LEMMA_CACHE = {}

TOKEN_RE = re.compile(r"[а-яa-z0-9]+")


# ------------------------------------------------------------------ 1. Загрузка и исследование
def load_data(folder: Path):
    docs = pd.read_csv(folder / "documents.csv", encoding=ENC)
    queries = pd.read_csv(folder / "queries.csv", encoding=ENC)
    rel = pd.read_csv(folder / "dev_relevance.csv", encoding=ENC)
    return docs, queries, rel


def explore(docs, queries, rel, log):
    log("=== ИССЛЕДОВАНИЕ КОРПУСА ===")
    log(f"Документов: {len(docs)}; поля: {list(docs.columns)}")
    log(f"Запросов: {len(queries)}; поля: {list(queries.columns)}")
    log(f"Размеченных запросов (dev): {rel['query_id'].nunique()}, пар: {len(rel)}")
    log("Типы данных:\n" + docs.dtypes.to_string())
    log("Пропуски в documents:\n" + docs.isna().sum().to_string())
    log("Пропуски в queries:\n" + queries.isna().sum().to_string())
    log(f"ID документов уникальны: {docs['document_id'].is_unique}")
    log(f"ID запросов уникальны: {queries['query_id'].is_unique}")
    log(f"Дубликаты текстов: {docs['text'].duplicated().sum()}; "
        f"дубликаты заголовков: {docs['title'].duplicated().sum()}")
    log(f"Пустые/пробельные тексты: {(docs['text'].fillna('').str.strip() == '').sum()}")
    lens = docs["text"].fillna("").str.split().str.len()
    log(f"Длина текста (слов): min={lens.min()}, mean={lens.mean():.1f}, max={lens.max()}")
    log("Разделы:\n" + docs["section"].value_counts().to_string())
    log("Сложность запросов:\n" + queries["difficulty"].value_counts().to_string())
    missing = set(rel["relevant_document_id"]) - set(docs["document_id"])
    log(f"Эталонные ID, отсутствующие в корпусе: {missing or 'нет'}")
    log("")

# ПУНКТ 2. NLP-ПРЕДОБРАБОТКА И БАЗОВЫЙ ПОИСК (BASELINE)

# ------------------------------------------------------------------ 2. Предобработка
def clean(text) -> str:
    """Очистка: нижний регистр, ё->е, всё кроме букв/цифр -> пробел."""
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е")


def tokenize(text) -> list:
    return TOKEN_RE.findall(clean(text))


def lemma(word: str) -> str:
    if word not in _LEMMA_CACHE:
        _LEMMA_CACHE[word] = MORPH.parse(word)[0].normal_form.replace("ё", "е")
    return _LEMMA_CACHE[word]


def preprocess(text, lemmatize: bool, remove_stop: bool) -> list:
    """Единая обработка и для документов, и для запросов."""
    tokens = tokenize(text)
    if remove_stop:
        # стоп-слова удаляем и до, и после лемматизации (напр. «какие» -> «какой»)
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if lemmatize:
        tokens = [lemma(t) for t in tokens]
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


# ------------------------------------------------------------------ 3. Базовый поиск
class BaselineSearch:
    """Поиск по совпадению слов: score = число различных слов запроса, найденных в документе."""

    def __init__(self, docs, lemmatize=True, remove_stop=True):
        self.ids = docs["document_id"].tolist()
        self.opts = (lemmatize, remove_stop)
        full = (docs["title"].fillna("") + " " + docs["text"].fillna(""))
        self.sets = [set(preprocess(t, *self.opts)) for t in full]

    def search(self, query, k=TOP_K):
        q = set(preprocess(query, *self.opts))
        if not q:  # пустой запрос -> пустая выдача
            return []
        scores = np.array([len(q & s) for s in self.sets], dtype=float)
        # стабильная сортировка: при равенстве баллов — порядок в корпусе
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

# ПУНКТ 3. TF-IDF ВЕКТОРИЗАЦИЯ И МЕТРИКИ РАНЖИРОВАНИЯ

# ------------------------------------------------------------------ 4. TF-IDF + cosine
CONFIGS = {
    "A": dict(lemmatize=False, remove_stop=False, ngram=(1, 1),
              desc="без лемматизации, стоп-слова остаются, униграммы"),
    "B": dict(lemmatize=True, remove_stop=False, ngram=(1, 1),
              desc="лемматизация, стоп-слова остаются, униграммы"),
    "C": dict(lemmatize=True, remove_stop=True, ngram=(1, 1),
              desc="лемматизация, стоп-слова удалены, униграммы"),
    "D": dict(lemmatize=True, remove_stop=True, ngram=(1, 2),
              desc="лемматизация, стоп-слова удалены, униграммы + биграммы"),
}


class TfidfSearch:
    def __init__(self, docs, lemmatize, remove_stop, ngram):
        self.ids = docs["document_id"].tolist()
        self.opts = (lemmatize, remove_stop)
        full = (docs["title"].fillna("") + " " + docs["text"].fillna(""))
        corpus = [" ".join(preprocess(t, *self.opts)) for t in full]
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,
            ngram_range=ngram, sublinear_tf=True)
        # fit_transform ТОЛЬКО на документах
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, k=TOP_K):
        q_tokens = preprocess(query, *self.opts)
        if not q_tokens:
            return []
        # transform тем же векторизатором (без повторного fit)
        q_vec = self.vectorizer.transform([" ".join(q_tokens)])
        if q_vec.nnz == 0:  # ни одного известного слова
            return []
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


# ------------------------------------------------------------------ 5. Метрики
def evaluate(engine, queries, rel_map):
    """Hit@5 и MRR@5 только по размеченным запросам."""
    hits, rr, per_query = [], [], []
    for qid, qtext in zip(queries["query_id"], queries["query"]):
        if qid not in rel_map:
            continue
        ranked = [d for d, _ in engine.search(qtext, TOP_K)]
        rank = next((i + 1 for i, d in enumerate(ranked) if d in rel_map[qid]), None)
        hits.append(1.0 if rank else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
        per_query.append((qid, rank, ranked))
    return float(np.mean(hits)), float(np.mean(rr)), per_query


def run_all(engine, queries):
    rows = []
    for qid, qtext in zip(queries["query_id"], queries["query"]):
        res = engine.search(qtext, TOP_K)
        for r, (d, s) in enumerate(res, 1):
            rows.append((qid, r, d, round(s, 6)))
    return pd.DataFrame(rows, columns=["query_id", "rank", "document_id", "score"])

# ПУНКТ 4. АНАЛИЗ ОШИБОК, ЗАПУСК ПАЙПЛЛАЙНА И ЭКСПОРТ АРТЕФАКТОВ

# ------------------------------------------------------------------ 6. Анализ ошибок
def error_analysis(name, engine, docs, queries, rel_map, per_query, log, max_cases=3):
    title = docs.set_index("document_id")["title"].to_dict()
    qtext = queries.set_index("query_id")["query"].to_dict()
    bad = [p for p in per_query if p[1] != 1]
    log(f"=== АНАЛИЗ ОШИБОК (конфигурация {name}) ===")
    if not bad:
        log("На dev-запросах все релевантные документы на 1-м месте.\n")
        return
    for qid, rank, ranked in bad[:max_cases]:
        gold = list(rel_map[qid])[0]
        log(f"Запрос {qid}: {qtext[qid]}")
        log(f"  Эталон: {gold} «{title[gold]}» -> позиция {rank if rank else 'вне ТОП-5'}")
        scores = dict(engine.search(qtext[qid], TOP_K))
        for i, d in enumerate(ranked, 1):
            log(f"   {i}. {d} «{title[d]}» score={scores[d]:.4f}")
    log("")


# ------------------------------------------------------------------ main
def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("01_Ботанический_справочник")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    docs, queries, rel = load_data(folder)
    explore(docs, queries, rel, log)
    rel_map = rel.groupby("query_id")["relevant_document_id"].apply(set).to_dict()

    # --- пример согласованной предобработки
    sample = queries["query"].iloc[0]
    log("=== ПРЕДОБРАБОТКА (пример запроса) ===")
    log(f"Исходный: {sample}")
    log(f"Токены:   {tokenize(sample)}")
    log(f"Лемм.+стоп-слова удалены: {preprocess(sample, True, True)}\n")

    # --- базовый поиск
    base = BaselineSearch(docs)
    b_hit, b_mrr, _ = evaluate(base, queries, rel_map)
    log("=== БАЗОВЫЙ ПОИСК (совпадение слов) ===")
    log(f"Hit@5={b_hit:.3f}  MRR@5={b_mrr:.3f}")
    log(f"Пустой запрос -> {base.search('')}  |  только стоп-слова -> {base.search('и в на')}\n")
    base_df = run_all(base, queries)
    base_df.to_csv(out / "baseline_results.csv", index=False, encoding=ENC)

    # --- 4 конфигурации
    engines, rows, perq = {}, [], {}
    log("=== TF-IDF + COSINE: СРАВНЕНИЕ КОНФИГУРАЦИЙ (8 dev-запросов) ===")
    for name, cfg in CONFIGS.items():
        eng = TfidfSearch(docs, cfg["lemmatize"], cfg["remove_stop"], cfg["ngram"])
        hit, mrr, pq = evaluate(eng, queries, rel_map)
        engines[name], perq[name] = eng, pq
        rows.append(dict(configuration=name, hit_at_5=round(hit, 4), mrr_at_5=round(mrr, 4),
                         n_features=eng.matrix.shape[1], description=cfg["desc"]))
        log(f"{name}: Hit@5={hit:.3f}  MRR@5={mrr:.3f}  признаков={eng.matrix.shape[1]}  ({cfg['desc']})")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "experiment_metrics.csv", index=False, encoding=ENC)

    # --- выбор итоговой конфигурации: лучшая по MRR, затем Hit; при равенстве — более простая (раньше по алфавиту)
    best = metrics.sort_values(["mrr_at_5", "hit_at_5", "configuration"],
                               ascending=[False, False, True]).iloc[0]["configuration"]
    log(f"\nИтоговая конфигурация для search_results.csv: {best}\n")

    # --- результаты по всем 30 запросам
    res = run_all(engines[best], queries)
    res.to_csv(out / "search_results.csv", index=False, encoding=ENC)
    log(f"search_results.csv: {len(res)} строк, запросов: {res['query_id'].nunique()}")

    # --- анализ ошибок: для A (самая слабая по MRR) и итоговой
    worst = metrics.sort_values(["mrr_at_5", "configuration"]).iloc[0]["configuration"]
    for n in dict.fromkeys([worst, best]):
        error_analysis(n, engines[n], docs, queries, rel_map, perq[n], log)

    # --- график
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(metrics)); w = 0.38
    ax.bar(x - w / 2, metrics["hit_at_5"], w, label="Hit@5")
    ax.bar(x + w / 2, metrics["mrr_at_5"], w, label="MRR@5")
    ax.set_xticks(x); ax.set_xticklabels(metrics["configuration"])
    ax.set_ylim(0, 1.05); ax.set_title("Конфигурации A–D (8 dev-запросов)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "metrics_comparison.png", dpi=150)

    (out / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
