# YouTube Recommendation System

Персональная рекомендательная система на основе истории просмотров YouTube.  
Цель проекта — попытаться **дистиллировать YouTube**: собрать срез платформы и построить на нём рекомендации, персонализированные под конкретного пользователя через его реальную историю просмотров.

---

## Архитектура пайплайна

```
YouTube Data API v3
       ↓
  Сбор данных (146 поисковых запросов, ~25 000 видео)
       ↓
  Препроцессинг (glob CSV-шардов, фильтрация, full_text)
       ↓
  Google Takeout (watch-history.html)
       ↓
  Обогащение каталога (добор просмотренных видео через API)
       ↓
  Эмбеддинги BAAI/bge-m3 + FAISS-индекс
       ↓
  ┌─────────────┬──────────────────────┐
  │  Baseline   │  CatBoost Ranking    │
  │  (cosine)   │  (двухэтапный)       │
  └─────────────┴──────────────────────┘
       ↓
  Оценка: Hit Rate / Precision / Recall / NDCG @ K
```

---

## Структура проекта

```
.
├── youtube_recommender.ipynb       # Основной ноутбук (8 шагов)
├── youtube_large_dataset*.csv      # Сырые данные (CSV-шарды от API)
├── youtube_clean_filtered.parquet  # Очищенный каталог видео
├── parsed_watch_history_filtered.parquet  # История просмотров
├── youtube_faiss.index             # FAISS-индекс эмбеддингов
├── catboost_ranker_leakfree.cbm    # Обученная модель CatBoost
└── история-просмотров.html         # Google Takeout (не включать в git)
```

---

## Шаги пайплайна

### Шаг 1 — Сбор данных
Сбор видео через YouTube Data API v3. 146 поисковых запросов по тематикам: ML/программирование, игры (Dota 2, CS2, Minecraft), спорт, юмор, наука, кулинария, новости и др. Данные сохраняются в несколько CSV-шардов.

Требуется API-ключ: [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → YouTube Data API v3 → Credentials.

```python
API_KEY = "YOUR_YOUTUBE_API_KEY"
TARGET_VIDEOS = 25_000
```

### Шаг 2 — Препроцессинг
Glob-объединение всех шардов `youtube_large_dataset*.csv`, дедупликация по `video_id`, фильтрация шортс и спама, фильтр по длительности (180–18 000 с), сборка поля:

```
full_text = title + description + tags
```

Выход: `youtube_clean_filtered.parquet`

### Шаг 3 — История просмотров
Парсинг `watch-history.html` из Google Takeout через BeautifulSoup. Извлечение `video_id` из ссылок вида `youtube.com/watch?v=...`.

Скачать историю: [myaccount.google.com](https://myaccount.google.com) → Data & privacy → Download your data → YouTube → `Takeout/YouTube/history/watch-history.html`

### Шаг 4 — Обогащение каталога
Добор просмотренных видео, отсутствующих в каталоге, через API батчами по 50. Решает проблему малого пересечения (было 241 совпадение → стало 408).

```python
ENRICH_N = 2500  # максимум видео для добора
```

### Шаг 5 — Эмбеддинги и FAISS-индекс
Кодирование `full_text` моделью `BAAI/bge-m3` (мультиязычная sentence-transformer). Нормализованные векторы + `IndexFlatIP` = cosine similarity.

```python
model = SentenceTransformer("BAAI/bge-m3", device=device)
index = faiss.IndexFlatIP(dim)
```

После изменения каталога (Шаг 4) необходимо пересобрать индекс.

### Шаг 6 — Baseline (косинусная близость)
Профиль пользователя = среднее эмбеддингов просмотренных видео. Поиск ближайших в FAISS.

### Шаг 7 — CatBoost Ranking (основная модель)

**Stage 1 — Recall:** FAISS достаёт 200 кандидатов по семантической близости к профилю.

**Stage 2 — Ranking:** CatBoost ранжирует кандидатов по 5 признакам:

| Признак | Описание |
|---|---|
| `similarity_score` | Cosine similarity к профилю пользователя |
| `log_views` | log(1 + view_count) |
| `log_likes` | log(1 + like_count) |
| `log_comments` | log(1 + comment_count) |
| `duration_seconds` | Длина видео в секундах |

**Схема обучения (без утечки данных):**
- Позитивы: train-часть истории (80%)
- Негативы: 5× случайных видео, не входящих ни в train, ни в test
- CatBoost никогда не видит test-видео до оценки

### Шаг 8 — Оценка
Leave-20%-out: профиль и обучение CatBoost — на 80% просмотров, проверка попадания — на оставшихся 20%.

---

## Результаты

| Модель | Hit Rate@10 | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| Baseline (cosine) | 0.000 | 0.000 | 0.000 | 0.000 |
| CatBoost (leak-free) | **1.000** | **0.200** | **0.024** | **0.202** |

Baseline даёт нули: каталог (~25K видео) семантически не пересекается с конкретными видео из личной истории. CatBoost за счёт признаков популярности и вовлечённости находит релевантные видео там, где косинусная близость не справляется.

**Feature Importance:**

```
log_views          26.5%  ██████████████
log_comments       22.8%  ███████████
duration_seconds   19.6%  █████████
log_likes          16.4%  ████████
similarity_score   14.7%  ███████
```

---

## Установка зависимостей

```bash
pip install pandas numpy faiss-cpu sentence-transformers catboost \
            scikit-learn isodate pyarrow beautifulsoup4 \
            google-api-python-client
```

Для GPU:
```bash
pip install faiss-gpu
```

---


## Известные ограничения

- **Маленький каталог** (~66K видео) — недостаточен для полноценной дистилляции YouTube
- **Короткая история просмотров** — аккаунту около года, 408 совпадений вместо тысяч
- **Квота API** — ежедневный лимит YouTube Data API требует ротации ключей при большом сборе
