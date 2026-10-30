from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from recommender import get_model


@asynccontextmanager
async def lifespan(_app):
    get_model()
    yield


app = FastAPI(title="Рекомендательная система фильмов", lifespan=lifespan)

HELP = [
    {
        "method": "GET",
        "path": "/analysis",
        "params": "нет",
        "about": "Краткий разбор датасета: размеры, среднее, фильтр пользователей и что попало в модель.",
    },
    {
        "method": "GET",
        "path": "/popular",
        "params": "n — сколько фильмов вернуть, по умолчанию 10 (максимум 20)",
        "about": "Топ популярных фильмов. Взвешенный рейтинг по формуле из ноутбука.",
    },
    {
        "method": "GET",
        "path": "/collab/item",
        "params": "title — название фильма; n — сколько рекомендаций, по умолчанию 10",
        "about": "Коллаборативная фильтрация по фильмам. NearestNeighbors, косинус, как в ноутбуке.",
    },
    {
        "method": "GET",
        "path": "/collab/user",
        "params": "user_id — id пользователя; n — сколько рекомендаций, по умолчанию 10",
        "about": "Коллаборативная фильтрация по пользователям. Твои оценки сравниваются с оценками других людей, берутся фильмы, которые понравились похожим.",
    },
    {
        "method": "GET",
        "path": "/content",
        "params": "title — название фильма; n — сколько рекомендаций, по умолчанию 10",
        "about": "10 фильмов по контенту. Сравниваются название и жанры.",
    },
    {
        "method": "GET",
        "path": "/personal",
        "params": "user_id — id пользователя; n — сколько рекомендаций, по умолчанию 10",
        "about": "Персональные рекомендации. В API считается коллаборативная фильтрация: по твоим оценкам ищутся похожие пользователи и их фильмы.",
    },
    {
        "method": "GET",
        "path": "/movies",
        "params": "q — кусок названия, минимум 2 символа; limit — сколько строк, по умолчанию 20",
        "about": "Поиск фильма.",
    },
    {
        "method": "GET",
        "path": "/users",
        "params": "limit — сколько пользователей, по умолчанию 20",
        "about": "Сначала добавленные пользователи с оценками, потом люди из датасета с большим числом оценок.",
    },
    {
        "method": "GET",
        "path": "/user/{user_id}",
        "params": "user_id — id пользователя",
        "about": "Сколько оценок у пользователя и есть ли он в модели.",
    },
    {
        "method": "GET",
        "path": "/ratings",
        "params": "user_id — id пользователя; limit — сколько оценок показать, по умолчанию 40",
        "about": "Оценки пользователя из базы.",
    },
    {
        "method": "POST",
        "path": "/rate",
        "params": "json: user_id, movie_id, rating. rating от 0.5 до 5",
        "about": "Сохранить оценку фильма в SQLite.",
    },
    {
        "method": "GET",
        "path": "/help",
        "params": "нет",
        "about": "Список команд и параметров.",
    },
]


class RateBody(BaseModel):
    user_id: int
    movie_id: int
    rating: float


def norm_n(n: int) -> int:
    if n < 1:
        return 1
    if n > 20:
        return 20
    return n


@app.get("/")
def root():
    return {
        "project": "Рекомендательная система фильмов",
        "author": "Чернов Илья Дмитриевич 23п-4",
        "help": "/help",
        "docs": "/docs",
    }


@app.get("/help")
def help_api():
    return {"commands": HELP}


@app.get("/analysis")
def analysis():
    return get_model().analysis


@app.get("/popular")
def popular(n: int = 10):
    return get_model().popular(norm_n(n))


@app.get("/collab/item")
def collab_item(title: str, n: int = 10):
    if not title.strip():
        raise HTTPException(status_code=400, detail="нужен параметр title")
    items = get_model().similar_movies(title, norm_n(n))
    if items is None:
        raise HTTPException(
            status_code=404,
            detail="Фильм не попал в модель CF. Возьми фильм из топа, например Shawshank или Matrix.",
        )
    return items


@app.get("/collab/user")
def collab_user(user_id: int, n: int = 10):
    items = get_model().similar_by_user(user_id, norm_n(n))
    if items is None:
        raise HTTPException(
            status_code=404,
            detail="Нет оценок у этого id или не нашлось похожих пользователей. Поставь несколько оценок.",
        )
    return items


@app.get("/content")
def content(title: str, n: int = 10):
    if not title.strip():
        raise HTTPException(status_code=400, detail="нужен параметр title")
    items = get_model().content_by_title(title, norm_n(n))
    if not items:
        raise HTTPException(status_code=404, detail="Не получилось подобрать фильмы по названию и жанрам")
    return items


@app.get("/personal")
def personal(user_id: int, n: int = 10):
    items = get_model().personal(user_id, norm_n(n))
    if not items:
        raise HTTPException(
            status_code=404,
            detail="Нет оценок для этого id. Найди фильмы и поставь несколько оценок, лучше от 4.",
        )
    return items


@app.get("/movies")
def movies(q: str = "", limit: int = 20):
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="параметр q, минимум 2 символа")
    if limit < 1:
        limit = 1
    if limit > 50:
        limit = 50
    return get_model().search_movies(q, limit)


@app.get("/users")
def users(limit: int = 20):
    if limit < 1:
        limit = 1
    if limit > 50:
        limit = 50
    return get_model().top_users(limit)


@app.get("/user/{user_id}")
def user_info(user_id: int):
    return get_model().user_info(user_id)


@app.get("/ratings")
def ratings(user_id: int, limit: int = 40):
    return get_model().user_ratings(user_id, limit)


@app.post("/rate")
def rate(body: RateBody):
    saved, err = get_model().add_rating(body.user_id, body.movie_id, body.rating)
    if err:
        code = 400 if "оценка" in err else 404
        raise HTTPException(status_code=code, detail=err)
    return saved


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
