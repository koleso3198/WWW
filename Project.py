import os
import re
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pymorphy3
import emoji

# 1. ЗАГРУЗКА РЕСУРСОВ NLTK И НАСТРОЙКА ЛЕММАТИЗАТОРА

nltk.download("stopwords", quiet=True)

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))

# 2. ФИЛЬТРАЦИЯ СПАМА

SPAM_KEYWORDS = {
    "казино", "выигрыш", "ставка", "ставки", "займ", "микрозайм",
    "кредит", "крипта", "инвестиции", "заработок", "работа на дому",
    "скидка 90%", "выиграй", "акция только сегодня"
}


def is_spam(text: str) -> bool:
    """Определяет спам-сообщения по ключевым словам и бессмысленному содержимому."""
    if not isinstance(text, str):
        return True

    text_lower = text.lower()

    # Поиск спам-триггеров
    if any(keyword in text_lower for keyword in SPAM_KEYWORDS):
        return True

    # Сообщения без буквенного смысла (состоящие только из цифр, ссылок и знаков)
    letters = re.findall(r"[a-zа-яё]", text_lower)
    if len(letters) < 5:
        return True

    return False

# 3. АНОНИМИЗАЦИЯ ДАННЫХ И ОЧИСТКА ОТ ШУМА

def mask_and_clean_text(text: str) -> str:
    """
    Заменяет персональные данные и метаданные специальными тегами:
    <URL>, <EMAIL>, <PHONE>, <DATE>, <PRICE>, <ORDER_ID>,
    а также удаляет эмодзи и посторонние спецсимволы.
    """
    if not isinstance(text, str):
        return ""

    # 1. Ссылки и URL
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text, flags=re.I)

    # 2. E-mail адреса
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<EMAIL>", text)

    # 3. Номера телефонов (+7, 8, скобки, дефисы, пробелы)
    phone_pattern = r"(?:\+7|8)[\s\-(]?\s?\(?\d{3}\)?[\s\)-]?\s?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    text = re.sub(phone_pattern, "<PHONE>", text)

    # 4. Даты (2026-11-01, 05/10/2026, 28-06-26, 15.03.2026)
    date_pattern = r"\b(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b"
    text = re.sub(date_pattern, "<DATE>", text)

    # 5. Номера заказов и отправлений (№74990, #10335)
    text = re.sub(r"[№#]\s*\d+", "<ORDER_ID>", text)

    # 6. Цены и финансовые списания (рубли, RUB, ₽, валюты)
    price_pattern = r"\b\d+(?:[\s.,]\d+)?\s*(?:рублей|рубля|руб\.?|rub|usd|\$|eur|€|р)(?!\w)|\d+(?:[\s.,]\d+)?\s*₽"
    text = re.sub(price_pattern, "<PRICE>", text, flags=re.I)

    # 7. Удаление эмодзи
    text = emoji.replace_emoji(text, replace="")

    # 8. Удаление лишних спецсимволов (сохраняем теги <...>, слова и пробелы)
    text = re.sub(r"[^\w\s<>]+", " ", text)

    # 9. Схлопывание повторяющихся пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 4. ТОКЕНИЗАЦИЯ И ЛЕММАТИЗАЦИЯ

def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Выделяет теги-плейсхолдеры и русские слова,
    удаляет стоп-слова и приводит слова к нормальной форме (лемме).
    """
    raw_tokens = re.findall(r"<[A-Z_]+>|[а-яёА-ЯЁ]+", text)
    lemmatized_tokens = []

    for token in raw_tokens:
        # Теги вида <PHONE>, <URL> сохраняются без изменений
        if token.startswith("<") and token.endswith(">"):
            lemmatized_tokens.append(token)
            continue

        token_lower = token.lower()
        # Пропуск стоп-слов и слишком коротких токенов
        if token_lower in russian_stopwords or len(token_lower) <= 2:
            continue

        # Лемматизация слова
        normal_form = morph.parse(token_lower)[0].normal_form
        lemmatized_tokens.append(normal_form)

    return lemmatized_tokens

# 5. ОСНОВНОЙ ПАЙПЛАЙН ВЫПОЛНЕНИЯ

def main():
    input_file = "03_dostavka_i_logistika.csv"
    output_file = "03_dostavka_i_logistika_processed.csv"

    if not os.path.exists(input_file):
        print(f"Ошибка: файл '{input_file}' не найден в текущей директории.")
        return

    print(f"1. Загрузка базы данных из {input_file}...")
    df = pd.read_csv(input_file, encoding="utf-8-sig", sep=",")
    initial_count = len(df)
    print(f"   Загружено строк: {initial_count}")

    # Удаление явных дубликатов и пропусков
    df = df.dropna(subset=["message"]).drop_duplicates(subset=["message"]).reset_index(drop=True)
    print(f"   После удаления дубликатов: {len(df)} строк")

    # Удаление спама
    df["is_spam"] = df["message"].apply(is_spam)
    clean_df = df[~df["is_spam"]].copy().reset_index(drop=True)
    print(f"   Удалено спам-сообщений: {df['is_spam'].sum()}")
    print(f"   Осталось чистых сообщений: {len(clean_df)}")

    # Анонимизация персональных данных и очистка
    print("\n2. Анонимизация персональных данных (<PHONE>, <EMAIL>, <PRICE>)...")
    clean_df["anonymized_message"] = clean_df["message"].apply(mask_and_clean_text)

    # Токенизация и лемматизация
    print("3. Токенизация и лемматизация сообщений...")
    clean_df["tokens"] = clean_df["anonymized_message"].apply(tokenize_and_lemmatize)
    clean_df["lemmatized_message"] = clean_df["tokens"].apply(lambda words: " ".join(words))

    # Сбор статистики
    all_tokens = [tok for tokens in clean_df["tokens"] for tok in tokens]
    entity_tags = ["<PHONE>", "<EMAIL>", "<URL>", "<DATE>", "<PRICE>", "<ORDER_ID>"]

    print("\nСтатистика замененных персональных сущностей:")
    for tag in entity_tags:
        print(f"   {tag:<12}: {all_tokens.count(tag)} шт.")

    top_lemmas = Counter([t for t in all_tokens if t not in entity_tags]).most_common(10)
    print("\nТоп-10 ключевых лемм предметной области:")
    for rank, (word, count) in enumerate(top_lemmas, 1):
        print(f"   {rank}. {word:<15} — {count} раз")

    # Сохранение результата в CSV
    export_columns = [
        "message_id",
        "channel",
        "created_at",
        "message",  # Исходное сообщение
        "anonymized_message",  # Суррогатные данные: <PHONE>, <EMAIL> и т.д.
        "lemmatized_message"  # Финальный лемматизированный текст
    ]
    clean_df[export_columns].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[✓] Результат сохранен в файл: {output_file}")

    # Демонстрация результата первой строки
    print("\nПример обработки первой записи:")
    sample = clean_df.iloc[0]
    print(f"Исходное     : {sample['message']}")
    print(f"Анонимизация : {sample['anonymized_message']}")
    print(f"Леммы        : {sample['lemmatized_message']}")


if __name__ == "__main__":
    main()