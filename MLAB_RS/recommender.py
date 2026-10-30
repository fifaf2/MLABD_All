import os
import re
import sqlite3

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ml-latest-small")
DB_PATH = os.path.join(BASE_DIR, "movies.db")

_model = None


def get_model():
    global _model
    if _model is None:
        print("загрузка модели фильмов...", flush=True)
        model = Recommender()
        model.prepare()
        _model = model
        print("модель готова", flush=True)
    return _model


class Recommender:
    def __init__(self):
        self.db_path = DB_PATH
        self.top_popular = []
        self.movie_ids = []
        self.user_ids = []
        self.movie_index = {}
        self.user_index = {}
        self.meta = {}
        self.user_item = None
        self.item_user = None
        self.nn_user = None
        self.nn_item = None
        self.vectorizer = None
        self.content_matrix = None
        self.analysis = {"lines": []}

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def prepare(self):
        conn = self.connect()
        self.create_tables(conn)
        loaded = conn.execute("SELECT value FROM meta WHERE key = 'loaded'").fetchone()
        if loaded is None:
            print("первый запуск, пишу csv в базу", flush=True)
            movies, ratings = self.fill_database(conn)
        else:
            print("читаю базу", flush=True)
            movies = pd.read_sql("SELECT movie_id, title, genres, year FROM movies", conn)
            ratings = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
        extra = self.ensure_demo_users(conn)
        conn.close()
        if extra is not None:
            ratings = pd.concat([ratings, extra], ignore_index=True)
        self.fit(movies, ratings)

    def create_tables(self, conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY,
                title TEXT,
                genres TEXT,
                year TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER,
                movie_id INTEGER,
                rating REAL,
                PRIMARY KEY (user_id, movie_id)
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                user_id INTEGER PRIMARY KEY,
                name TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id)")
        conn.commit()

    def fill_database(self, conn):
        movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
        movies["movie_id"] = movies["movieId"].astype(int)
        movies["title"] = movies["title"].fillna("").astype(str).str.strip()
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies = movies[movies["genres"] != "(no genres listed)"]
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].fillna("")
        movies = movies[(movies["title"] != "")]
        movies = movies.drop_duplicates("movie_id")
        movies = movies[["movie_id", "title", "genres", "year"]]

        ratings["user_id"] = ratings["userId"].astype(int)
        ratings["movie_id"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings = ratings[ratings["rating"] > 0]
        ratings = ratings.groupby(["user_id", "movie_id"], as_index=False)["rating"].mean()
        ratings = ratings[["user_id", "movie_id", "rating"]]

        conn.executemany(
            "INSERT INTO movies(movie_id, title, genres, year) VALUES (?,?,?,?)",
            list(zip(movies["movie_id"].tolist(), movies["title"].tolist(), movies["genres"].tolist(), movies["year"].tolist())),
        )
        conn.executemany(
            "INSERT INTO ratings(user_id, movie_id, rating) VALUES (?,?,?)",
            list(zip(ratings["user_id"].tolist(), ratings["movie_id"].tolist(), ratings["rating"].tolist())),
        )
        conn.execute("INSERT INTO meta(key, value) VALUES ('loaded', '1')")
        conn.commit()
        print("база собрана", len(movies), len(ratings), flush=True)
        return movies, ratings

    def ensure_demo_users(self, conn):
        self.demo_names = {}
        ready = conn.execute("SELECT user_id, name FROM app_users").fetchall()
        for row in ready:
            self.demo_names[int(row["user_id"])] = row["name"]
        if self.demo_names:
            return None
        people = [
            (9001, "Аня", [(1, 5), (3114, 5), (78499, 5), (4306, 5), (6377, 5), (364, 5), (588, 5), (4886, 5), (5218, 4.5), (595, 4.5), (356, 4), (318, 3.5), (480, 3.5), (2571, 2), (296, 2.5), (1721, 3), (1407, 1), (1982, 1)]),
            (9002, "Максим", [(2571, 5), (260, 5), (1196, 5), (480, 4.5), (1240, 5), (3578, 4.5), (780, 4), (4993, 4.5), (1, 3), (318, 3), (356, 3), (296, 4), (1721, 2), (8533, 1), (597, 1.5)]),
            (9003, "Илья", [(318, 5), (858, 5), (356, 4.5), (3147, 5), (296, 4.5), (2959, 4.5), (593, 4), (50, 4.5), (1, 3), (2571, 3.5), (1721, 3.5), (480, 3), (54503, 2)]),
            (9004, "Олег", [(54503, 5), (4816, 4.5), (2706, 4.5), (104, 4.5), (344, 4.5), (4306, 4), (1, 4), (2571, 3), (318, 3), (356, 3.5), (1721, 2.5), (296, 3.5), (480, 3)]),
            (9005, "Мария", [(1721, 5), (8533, 5), (597, 4.5), (6942, 5), (356, 4.5), (595, 4), (1, 3.5), (2571, 2), (318, 4), (296, 2.5), (480, 2.5), (1407, 1.5)]),
            (9006, "Денис", [(1407, 5), (1982, 4.5), (1997, 5), (5679, 4.5), (593, 4.5), (1214, 4.5), (1200, 4.5), (2959, 4), (1, 1.5), (2571, 4), (318, 3), (356, 2.5), (1721, 1.5), (296, 4), (480, 3.5)]),
            (9007, "Катя", [(4993, 5), (5952, 5), (7153, 5), (260, 4), (595, 4), (588, 3.5), (3578, 3.5), (1, 3.5), (2571, 4), (318, 4), (356, 3.5), (1721, 3), (296, 3), (480, 3)]),
            (9008, "Павел", [(318, 5), (356, 5), (2571, 4.5), (296, 4.5), (1, 4), (480, 4), (1721, 4), (260, 4), (2959, 4.5), (858, 4.5)]),
        ]
        rows = []
        for user_id, name, marks in people:
            conn.execute("INSERT INTO app_users(user_id, name) VALUES (?, ?)", (user_id, name))
            self.demo_names[user_id] = name
            for movie_id, rating in marks:
                rows.append((user_id, movie_id, float(rating)))
        conn.executemany(
            "INSERT OR REPLACE INTO ratings(user_id, movie_id, rating) VALUES (?,?,?)",
            rows,
        )
        conn.commit()
        print("добавлены пользователи", ", ".join(self.demo_names.values()), flush=True)
        return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])

    def fit(self, movies, ratings):
        movies = movies.copy()
        ratings = ratings.copy()
        movies["movie_id"] = movies["movie_id"].astype(int)
        movies["title"] = movies["title"].fillna("").astype(str)
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies["year"] = movies["year"].fillna("").astype(str)
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["movie_id"] = ratings["movie_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings = ratings[ratings["rating"] > 0]

        merged = ratings.merge(movies, on="movie_id", how="inner")
        print("оценок после соединения", len(merged), flush=True)

        user_counts = merged.groupby("user_id").size()
        active_users = user_counts[user_counts > 200].index
        part = merged[merged["user_id"].isin(active_users)]
        movie_counts = part.groupby("movie_id").size()
        part = part[part["movie_id"].isin(movie_counts[movie_counts >= 5].index)]
        part = part.groupby(["user_id", "movie_id"], as_index=False).agg(
            rating=("rating", "mean"),
            title=("title", "first"),
            genres=("genres", "first"),
            year=("year", "first"),
        )
        print("в модели пользователей", part["user_id"].nunique(), "фильмов", part["movie_id"].nunique(), flush=True)

        meta_rows = part.drop_duplicates("movie_id")
        self.meta = {}
        for row in meta_rows.itertuples(index=False):
            self.meta[int(row.movie_id)] = {
                "title": row.title,
                "genres": row.genres,
                "year": row.year,
            }

        self.user_ids = sorted(int(x) for x in part["user_id"].unique())
        self.movie_ids = sorted(int(x) for x in part["movie_id"].unique())
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        row_idx = part["user_id"].astype(int).map(self.user_index).to_numpy()
        col_idx = part["movie_id"].astype(int).map(self.movie_index).to_numpy()
        data = part["rating"].to_numpy(dtype=float)
        self.user_item = csr_matrix((data, (row_idx, col_idx)), shape=(len(self.user_ids), len(self.movie_ids)))
        centered = self.user_item.copy()
        for i in range(centered.shape[0]):
            start, end = centered.indptr[i], centered.indptr[i + 1]
            if end > start:
                centered.data[start:end] = centered.data[start:end] - centered.data[start:end].mean()
        self.item_user = centered.T.tocsr()

        self.nn_user = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_user.fit(self.user_item)
        self.nn_item = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_item.fit(self.item_user)
        self.item_sim = cosine_similarity(self.item_user).astype(np.float32)

        texts = [self.content_text(mid) for mid in self.movie_ids]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.content_matrix = self.vectorizer.fit_transform(texts)

        self.catalog = {}
        for row in movies.itertuples(index=False):
            self.catalog[int(row.movie_id)] = {
                "title": row.title,
                "genres": row.genres,
                "year": "" if row.year is None else str(row.year),
            }
        self.all_ratings = {}
        for uid, mid, rating in zip(ratings["user_id"], ratings["movie_id"], ratings["rating"]):
            self.all_ratings.setdefault(int(uid), {})[int(mid)] = float(rating)

        self.top_popular = self.calc_popular(merged)
        self.analysis = self.make_analysis(movies, ratings, merged, user_counts, part)

    def content_text(self, movie_id):
        info = self.meta.get(int(movie_id), {})
        genres = info.get("genres", "").replace("|", " ")
        genres = genres.replace("(no genres listed)", "")
        return (info.get("title", "") + " " + genres).strip()

    def calc_popular(self, merged):
        pop = merged.groupby("movie_id").agg(
            title=("title", "first"),
            genres=("genres", "first"),
            year=("year", "first"),
            avg=("rating", "mean"),
            votes=("rating", "count"),
        )
        pop = pop[pop["votes"] >= 20]
        m = float(pop["votes"].quantile(0.90))
        c = float(pop["avg"].mean())
        pop["score"] = (pop["votes"] * pop["avg"] + m * c) / (pop["votes"] + m)
        pop = pop.sort_values("score", ascending=False).head(30)
        self.popular_m = m
        result = []
        for movie_id, row in pop.iterrows():
            result.append(
                {
                    "movie_id": int(movie_id),
                    "title": row["title"],
                    "genres": row["genres"],
                    "year": row["year"],
                    "score": round(float(row["score"]), 3),
                    "votes": int(row["votes"]),
                }
            )
        return result

    def make_analysis(self, movies, ratings, merged, user_counts, part):
        genres = movies["genres"].str.split("|").explode()
        genres = genres[genres != "(no genres listed)"]
        top = genres.value_counts().head(5)
        genre_text = ", ".join(f"{name} ({int(count)})" for name, count in top.items())
        cleaned = movies["title"].apply(lambda x: re.sub(r"[\W_]+", " ", str(x)).strip())
        changed = int((cleaned != movies["title"]).sum())
        lines = [
            "Датасет MovieLens Latest Small. На Kaggle он лежит как grouplens/movielens-latest-small.",
            "Файлы: movies.csv и ratings.csv.",
            f"Фильмов: {len(movies)}",
            f"Оценок: {len(ratings)}",
            f"Пользователей: {ratings['user_id'].nunique()}",
            f"Оценка от {ratings['rating'].min()} до {ratings['rating'].max()}, среднее {round(float(ratings['rating'].mean()), 3)}",
            f"После соединения фильмов и оценок осталось строк: {len(merged)}",
            f"Названия почищены как в ноутбуке, изменилось строк: {changed}. В интерфейсе оставлено исходное название.",
            f"Пользователей с числом оценок больше 200: {int((user_counts > 200).sum())}. Модель CF строится по ним, как в ноутбуке.",
            f"В матрице: пользователей {part['user_id'].nunique()}, фильмов {part['movie_id'].nunique()}, оценок {len(part)}",
            "Фильм попал в матрицу, если среди этих пользователей его оценили хотя бы 5 раз.",
            f"Популярность: взвешенный рейтинг, в расчет фильмы с 20+ оценками, порог m = {round(self.popular_m, 2)}",
            "Похожие фильмы считаются по косинусу, но оценка пользователя сначала сдвигается на его среднее, иначе в топ лезут просто популярные фильмы.",
            "Контент: TF-IDF по названию и жанрам. Отдельного описания сюжета в файле нет.",
            "Персональный список и CF по пользователям считаются в API: твои оценки сравниваются с оценками других людей, в том числе добавленных пользователей "
            + ", ".join(self.demo_names.values())
            + ".",
            f"Частые жанры: {genre_text}",
        ]
        return {
            "lines": lines,
            "movies": int(len(movies)),
            "ratings": int(len(ratings)),
            "users": int(ratings["user_id"].nunique()),
            "model_users": int(part["user_id"].nunique()),
            "model_movies": int(part["movie_id"].nunique()),
        }

    def pack(self, movie_id, score):
        info = self.meta.get(int(movie_id)) or self.catalog.get(int(movie_id)) or {}
        return {
            "movie_id": int(movie_id),
            "title": info.get("title", ""),
            "genres": info.get("genres", ""),
            "year": info.get("year", ""),
            "score": round(float(score), 3),
        }

    def resolve(self, title):
        title = (title or "").strip()
        if not title:
            return None
        low = title.casefold()
        exact = [mid for mid, info in self.meta.items() if info["title"].casefold() == low]
        if exact:
            return exact[0]
        hits = [mid for mid, info in self.meta.items() if low in info["title"].casefold()]
        if not hits:
            return None
        hits.sort(key=lambda mid: len(self.meta[mid]["title"]))
        return hits[0]

    def popular(self, n=10):
        n = max(1, min(int(n), 20))
        return self.top_popular[:n]

    def similar_movies(self, title, n=10):
        n = max(1, min(int(n), 20))
        movie_id = self.resolve(title)
        if movie_id is None:
            return None
        idx = self.movie_index[movie_id]
        k = min(n + 1, len(self.movie_ids))
        dist, ind = self.nn_item.kneighbors(self.item_user.getrow(idx), n_neighbors=k)
        out = []
        for d, j in zip(dist[0], ind[0]):
            other = self.movie_ids[int(j)]
            if other == movie_id:
                continue
            out.append(self.pack(other, max(0.0, 1.0 - float(d))))
            if len(out) >= n:
                break
        return out

    def pair_sim(self, mine, other, common):
        common = list(common)
        a = np.array([mine[movie_id] for movie_id in common], dtype=float)
        b = np.array([other[movie_id] for movie_id in common], dtype=float)
        if len(common) == 1:
            return max(0.0, 1.0 - abs(float(a[0] - b[0])) / 4.5)
        raw_gap = abs(float(a.mean() - b.mean()))
        a = a - a.mean()
        b = b - b.mean()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return max(0.0, 1.0 - raw_gap / 4.5)
        return float(np.dot(a, b) / (na * nb))

    def user_cf(self, user_id, n=10):
        n = max(1, min(int(n), 20))
        user_id = int(user_id)
        mine = self.all_ratings.get(user_id)
        if not mine:
            return None
        if len(mine) < 8:
            pool = {uid: self.all_ratings[uid] for uid in self.demo_names if uid in self.all_ratings}
            min_common = 1
        else:
            pool = self.all_ratings
            min_common = 2
        neighbors = []
        my_keys = set(mine)
        for other_id, other in pool.items():
            if other_id == user_id:
                continue
            common = my_keys.intersection(other)
            if len(common) < min_common:
                continue
            sim = self.pair_sim(mine, other, common)
            if sim <= 0.05:
                continue
            neighbors.append((sim, other_id))
        if not neighbors:
            return []
        neighbors.sort(reverse=True)
        scores = {}
        if len(mine) < 3:
            take = 1
        elif len(mine) < 8:
            take = 3
        else:
            take = 15
        for sim, other_id in neighbors[:take]:
            for movie_id, rating in self.all_ratings[other_id].items():
                if movie_id in mine or rating < 3.5:
                    continue
                bucket = scores.get(movie_id)
                if bucket is None:
                    bucket = [0.0, 0.0, 0]
                    scores[movie_id] = bucket
                bucket[0] += sim * float(rating)
                bucket[1] += sim
                bucket[2] += 1
        ranked = []
        for movie_id, (num, den, support) in scores.items():
            if den <= 0:
                continue
            ranked.append((num / den, num, movie_id))
        ranked.sort(reverse=True)
        out = []
        for pred, weight, movie_id in ranked:
            if movie_id not in self.catalog and movie_id not in self.meta:
                continue
            out.append(self.pack(movie_id, pred))
            if len(out) >= n:
                break
        return out

    def similar_by_user(self, user_id, n=10):
        return self.user_cf(user_id, n)

    def content_by_title(self, title, n=10):
        n = max(1, min(int(n), 20))
        movie_id = self.resolve(title)
        if movie_id is not None:
            text = self.content_text(movie_id)
        else:
            text = self.text_from_db(title)
            if not text:
                text = (title or "").strip()
        if not text:
            return []
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.content_matrix).ravel()
        if movie_id is not None:
            sims[self.movie_index[movie_id]] = -1
        out = []
        for j in np.argsort(-sims):
            if sims[j] <= 0:
                break
            out.append(self.pack(self.movie_ids[int(j)], sims[j]))
            if len(out) >= n:
                break
        return out

    def text_from_db(self, title):
        conn = self.connect()
        row = conn.execute(
            "SELECT title, genres FROM movies WHERE title = ? COLLATE NOCASE LIMIT 1",
            (title,),
        ).fetchone()
        conn.close()
        if row is None:
            return ""
        genres = (row["genres"] or "").replace("|", " ").replace("(no genres listed)", "")
        return (row["title"] + " " + genres).strip()

    def rated_ids(self, user_id):
        conn = self.connect()
        rows = conn.execute(
            "SELECT movie_id FROM ratings WHERE user_id = ?",
            (int(user_id),),
        ).fetchall()
        conn.close()
        return {int(row["movie_id"]) for row in rows}

    def user_rating_rows(self, user_id):
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT r.movie_id AS movie_id, m.title AS title, r.rating AS rating
            FROM ratings r
            JOIN movies m ON m.movie_id = r.movie_id
            WHERE r.user_id = ?
            """,
            (int(user_id),),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def personal(self, user_id, n=10):
        n = max(1, min(int(n), 20))
        by_users = self.user_cf(user_id, n)
        if by_users:
            return by_users
        rows = self.user_rating_rows(user_id)
        if not rows:
            return []
        vec = np.zeros(len(self.movie_ids), dtype=np.float32)
        known = np.zeros(len(self.movie_ids), dtype=bool)
        outside = []
        for row in rows:
            movie_id = int(row["movie_id"])
            if movie_id not in self.movie_index:
                outside.append(row)
                continue
            idx = self.movie_index[movie_id]
            vec[idx] = float(row["rating"]) - 3.0
            known[idx] = True
        if known.any():
            scores = self.item_sim.dot(vec)
            scores[known] = -1e9
            out = []
            for j in np.argsort(-scores):
                if scores[j] <= 0:
                    break
                out.append(self.pack(self.movie_ids[int(j)], scores[j]))
                if len(out) >= n:
                    break
            if out:
                return out
        liked = sorted(outside or rows, key=lambda row: float(row["rating"]), reverse=True)
        liked = [row for row in liked if float(row["rating"]) >= 4][:5]
        seen = {int(row["movie_id"]) for row in rows}
        result = []
        for movie in liked:
            for item in self.content_by_title(movie["title"], n):
                if int(item["movie_id"]) in seen:
                    continue
                seen.add(int(item["movie_id"]))
                result.append(item)
                if len(result) >= n:
                    return result
        return result

    def search_movies(self, q, limit=20):
        limit = max(1, min(int(limit), 50))
        q = (q or "").strip()
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT movie_id, title, genres, year
            FROM movies
            WHERE title LIKE ?
            ORDER BY CASE WHEN title LIKE ? THEN 0 ELSE 1 END, length(title)
            LIMIT ?
            """,
            ("%" + q + "%", q + "%", limit),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def top_users(self, limit=20):
        limit = max(1, min(int(limit), 50))
        result = []
        seen = set()
        for user_id, name in self.demo_names.items():
            result.append(
                {
                    "user_id": int(user_id),
                    "ratings": len(self.all_ratings.get(user_id, {})),
                    "name": name,
                }
            )
            seen.add(int(user_id))
        counts = np.diff(self.user_item.indptr)
        order = np.argsort(-counts)
        added = 0
        for i in order:
            user_id = int(self.user_ids[int(i)])
            if user_id in seen:
                continue
            result.append({"user_id": user_id, "ratings": int(counts[int(i)]), "name": ""})
            added += 1
            if added >= limit:
                break
        return result

    def user_info(self, user_id):
        user_id = int(user_id)
        conn = self.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ratings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()
        return {
            "user_id": user_id,
            "ratings": int(row["cnt"]) if row is not None else 0,
            "in_model": user_id in self.user_index,
        }

    def user_ratings(self, user_id, limit=40):
        limit = max(1, min(int(limit), 100))
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT r.movie_id AS movie_id, m.title AS title, m.genres AS genres, m.year AS year, r.rating AS rating
            FROM ratings r
            JOIN movies m ON m.movie_id = r.movie_id
            WHERE r.user_id = ?
            ORDER BY r.rating DESC
            LIMIT ?
            """,
            (int(user_id), limit),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def add_rating(self, user_id, movie_id, rating):
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = float(rating)
        if rating < 0.5 or rating > 5:
            return None, "оценка должна быть от 0.5 до 5"
        conn = self.connect()
        movie = conn.execute(
            "SELECT title, genres, year FROM movies WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
        if movie is None:
            conn.close()
            return None, "такого фильма нет в базе"
        conn.execute(
            "INSERT OR REPLACE INTO ratings(user_id, movie_id, rating) VALUES (?,?,?)",
            (user_id, movie_id, rating),
        )
        conn.commit()
        conn.close()
        self.all_ratings.setdefault(user_id, {})[movie_id] = rating
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "title": movie["title"],
            "genres": movie["genres"],
            "year": movie["year"],
            "rating": rating,
        }, None
