import json
import sys
import urllib.error
import urllib.request
from urllib.parse import urlencode

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

API = "http://127.0.0.1:8000"


def request_json(path, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            detail = json.loads(raw).get("detail", raw)
        except Exception:
            detail = raw
        return None, str(detail)
    except Exception as exc:
        return None, "API недоступен. Запусти python api.py или python main.py.\n" + str(exc)


class TextDialog(QDialog):
    def __init__(self, title, text):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(720, 480)
        layout = QVBoxLayout(self)
        box = QTextEdit()
        box.setReadOnly(True)
        box.setPlainText(text)
        layout.addWidget(box)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рекомендательная система фильмов")
        self.resize(1040, 720)
        self.selected_id = 0
        self.selected_title = ""
        self.tries = 0

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        layout.addWidget(QLabel("Рекомендательная система фильмов"))

        user_row = QHBoxLayout()
        user_row.addWidget(QLabel("ID пользователя:"))
        self.user_edit = QLineEdit()
        self.user_edit.setText("1000")
        self.user_edit.setPlaceholderText("свой id, например 1000")
        self.user_edit.setFixedWidth(220)
        user_row.addWidget(self.user_edit)
        self.user_combo = QComboBox()
        self.user_combo.setMinimumWidth(240)
        self.user_combo.currentIndexChanged.connect(self.on_user_combo)
        user_row.addWidget(self.user_combo)
        connect_btn = QPushButton("Подключиться к API")
        connect_btn.clicked.connect(self.try_load_users)
        user_row.addWidget(connect_btn)
        user_row.addStretch()
        layout.addLayout(user_row)

        self.user_label = QLabel("Данные пользователя появятся после выбора id")
        layout.addWidget(self.user_label)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Поиск фильма:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("часть названия")
        self.search_edit.returnPressed.connect(self.search_movies)
        search_row.addWidget(self.search_edit)
        search_btn = QPushButton("Найти")
        search_btn.clicked.connect(self.search_movies)
        search_row.addWidget(search_btn)
        layout.addLayout(search_row)

        self.search_table = self.make_table()
        self.search_table.setMaximumHeight(170)
        self.search_table.cellClicked.connect(self.pick_search_row)
        layout.addWidget(self.search_table)

        self.picked = QLabel("Фильм не выбран")
        layout.addWidget(self.picked)

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Оценка:"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.5, 5.0)
        self.spin.setSingleStep(0.5)
        self.spin.setValue(4.0)
        rate_row.addWidget(self.spin)
        rate_btn = QPushButton("Сохранить оценку")
        rate_btn.clicked.connect(self.save_rating)
        rate_row.addWidget(rate_btn)
        my_btn = QPushButton("Мои оценки")
        my_btn.clicked.connect(self.show_my_ratings)
        rate_row.addWidget(my_btn)
        rate_row.addStretch()
        layout.addLayout(rate_row)

        btn_row = QHBoxLayout()
        buttons = [
            ("Топ-10", self.show_popular),
            ("CF по фильмам", self.show_item_cf),
            ("CF по пользователям", self.show_user_cf),
            ("По контенту", self.show_content),
            ("Персональные", self.show_personal),
            ("Анализ", self.show_analysis),
            ("Справка", self.show_help),
        ]
        for text, func in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self.caption = QLabel("Результаты")
        layout.addWidget(self.caption)
        self.result_table = self.make_table()
        self.result_table.cellClicked.connect(self.pick_result_row)
        layout.addWidget(self.result_table)

        self.status = QLabel("Ожидание API. Первый запуск может занять немного времени.")
        layout.addWidget(self.status)
        layout.addWidget(QLabel("Чернов Илья Дмитриевич, группа 23п-4"))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.try_load_users)
        self.timer.start(2000)
        QTimer.singleShot(400, self.try_load_users)

    def make_table(self):
        table = QTableWidget(0, 0)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        return table

    def try_load_users(self):
        self.tries += 1
        data, err = request_json("/users?limit=25")
        if err:
            if self.tries >= 60:
                self.timer.stop()
                self.status.setText("API не ответил. Запусти python main.py или отдельно python api.py.")
            else:
                self.status.setText("Жду API, модель ещё считается...")
            return
        self.timer.stop()
        self.user_combo.blockSignals(True)
        self.user_combo.clear()
        for user in data:
            name = user.get("name") or ""
            if name:
                label = name + ", id " + str(user["user_id"])
            else:
                label = str(user["user_id"]) + " (в модели: " + str(user["ratings"]) + ")"
            self.user_combo.addItem(label, user["user_id"])
        self.user_combo.blockSignals(False)
        if self.user_edit.text().strip():
            self.load_user_info()
        self.status.setText("API подключен: " + API)

    def on_user_combo(self, index):
        uid = self.user_combo.itemData(index)
        if uid is None:
            return
        self.user_edit.setText(str(uid))
        self.load_user_info()

    def load_user_info(self):
        uid = self.need_user(silent=True)
        if uid is None:
            return
        data, err = request_json("/user/" + str(uid))
        if err or not data:
            self.user_label.setText("Не удалось прочитать пользователя")
            return
        if data.get("in_model"):
            flag = "этот id из обучающей выборки"
        else:
            flag = "свой id, список строится по твоим оценкам"
        self.user_label.setText("Оценок в базе: " + str(data.get("ratings", 0)) + ". " + flag)

    def need_user(self, silent=False):
        text = self.user_edit.text().strip()
        if not text.isdigit():
            if not silent:
                QMessageBox.information(self, "Пользователь", "Введи числовой id пользователя")
            return None
        return int(text)

    def need_title(self):
        if not self.selected_title:
            QMessageBox.information(self, "Фильм", "Сначала найди фильм и выбери строку")
            return False
        return True

    def call(self, path):
        data, err = request_json(path)
        if err:
            QMessageBox.warning(self, "Ошибка", err)
            return None
        return data

    def search_movies(self):
        q = self.search_edit.text().strip()
        if len(q) < 2:
            QMessageBox.information(self, "Поиск", "Введи хотя бы 2 символа")
            return
        data = self.call("/movies?" + urlencode({"q": q, "limit": 20}))
        if data is None:
            return
        self.search_table.clear()
        self.search_table.setColumnCount(4)
        self.search_table.setHorizontalHeaderLabels(["ID", "Название", "Жанры", "Год"])
        self.search_table.setRowCount(len(data))
        for i, row in enumerate(data):
            self.search_table.setItem(i, 0, QTableWidgetItem(str(row.get("movie_id", ""))))
            self.search_table.setItem(i, 1, QTableWidgetItem(str(row.get("title", ""))))
            self.search_table.setItem(i, 2, QTableWidgetItem(str(row.get("genres", ""))))
            self.search_table.setItem(i, 3, QTableWidgetItem(str(row.get("year", ""))))
        self.search_table.resizeColumnsToContents()
        self.search_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.status.setText("Найдено фильмов: " + str(len(data)))

    def pick_search_row(self, row, col):
        id_item = self.search_table.item(row, 0)
        title_item = self.search_table.item(row, 1)
        self.selected_id = int(id_item.text()) if id_item and id_item.text().isdigit() else 0
        self.selected_title = title_item.text() if title_item else ""
        self.picked.setText("Выбрано: " + self.selected_title)

    def pick_result_row(self, row, col):
        title_item = self.result_table.item(row, 0)
        if title_item is None:
            return
        self.selected_title = title_item.text()
        movie_id = title_item.data(Qt.ItemDataRole.UserRole) or 0
        try:
            self.selected_id = int(movie_id)
        except Exception:
            self.selected_id = 0
        self.picked.setText("Выбрано: " + self.selected_title)

    def show_rows(self, rows, caption):
        self.caption.setText(caption)
        self.result_table.clear()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Название", "Жанры", "Оценка"])
        self.result_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            title_item = QTableWidgetItem(str(row.get("title", "")))
            title_item.setData(Qt.ItemDataRole.UserRole, int(row.get("movie_id", 0) or 0))
            if "votes" in row:
                score_text = str(row.get("score", "")) + " (оценок: " + str(row["votes"]) + ")"
            elif "rating" in row and "score" not in row:
                score_text = str(row.get("rating", ""))
            else:
                score_text = str(row.get("score", ""))
            self.result_table.setItem(i, 0, title_item)
            self.result_table.setItem(i, 1, QTableWidgetItem(str(row.get("genres", ""))))
            self.result_table.setItem(i, 2, QTableWidgetItem(score_text))
        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status.setText(caption + ": " + str(len(rows)))

    def show_popular(self):
        data = self.call("/popular?n=10")
        if data is None:
            return
        self.show_rows(data, "Топ-10 популярных фильмов")

    def show_item_cf(self):
        if not self.need_title():
            return
        data = self.call("/collab/item?" + urlencode({"title": self.selected_title, "n": 10}))
        if data is None:
            return
        self.show_rows(data, "CF по фильмам: " + self.selected_title)

    def show_user_cf(self):
        uid = self.need_user()
        if uid is None:
            return
        data = self.call("/collab/user?" + urlencode({"user_id": uid, "n": 10}))
        if data is None:
            return
        self.show_rows(data, "CF по пользователям, id " + str(uid))

    def show_content(self):
        if not self.need_title():
            return
        data = self.call("/content?" + urlencode({"title": self.selected_title, "n": 10}))
        if data is None:
            return
        self.show_rows(data, "По контенту: " + self.selected_title)

    def show_personal(self):
        uid = self.need_user()
        if uid is None:
            return
        data = self.call("/personal?" + urlencode({"user_id": uid, "n": 10}))
        if data is None:
            return
        self.show_rows(data, "Персональные рекомендации, id " + str(uid))

    def show_my_ratings(self):
        uid = self.need_user()
        if uid is None:
            return
        data = self.call("/ratings?" + urlencode({"user_id": uid, "limit": 40}))
        if data is None:
            return
        self.show_rows(data, "Оценки пользователя " + str(uid))

    def save_rating(self):
        uid = self.need_user()
        if uid is None:
            return
        if not self.selected_id:
            QMessageBox.information(self, "Оценка", "Выбери фильм в таблице поиска.")
            return
        data, err = request_json(
            "/rate",
            method="POST",
            payload={"user_id": uid, "movie_id": self.selected_id, "rating": self.spin.value()},
        )
        if err:
            QMessageBox.warning(self, "Ошибка", err)
            return
        self.status.setText("Оценка сохранена: " + str(data.get("title", "")) + " = " + str(data.get("rating")))
        self.load_user_info()

    def show_analysis(self):
        data = self.call("/analysis")
        if data is None:
            return
        text = "\n".join(data.get("lines", []))
        dlg = TextDialog("Анализ данных", text)
        dlg.exec()

    def show_help(self):
        data = self.call("/help")
        if data is None:
            return
        lines = [
            "Команды API рекомендательной системы фильмов",
            "Базовый адрес: http://127.0.0.1:8000",
            "Документация FastAPI: http://127.0.0.1:8000/docs",
            "",
        ]
        for cmd in data.get("commands", []):
            lines.append(cmd["method"] + " " + cmd["path"])
            lines.append("Параметры: " + cmd["params"])
            lines.append(cmd["about"])
            lines.append("")
        dlg = TextDialog("Справка", "\n".join(lines))
        dlg.exec()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
