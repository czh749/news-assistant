from types import SimpleNamespace

import pytest

from services.scheduler_service import SchedulerService


class FakeUserService:
    def get_user(self, user_id):
        return SimpleNamespace(name="Tester")

    def get_user_interests(self, user_id):
        return [SimpleNamespace(interest_keyword="AI", weight=1.0)]


class FakeEmbedding:
    def embed_query(self, text):
        return [0.1, 0.2]


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [{
            "news_id": "news-1",
            "title": "Title",
            "source": "sina",
            "publish_time": "2026-09-12T08:00:00",
            "distance": 0.9,
        }]


class FakeMinio:
    def list_news(self, prefix="news/"):
        return ["news/2026-09-12/sina/news-1.json"]

    def download_news(self, object_name):
        return {
            "id": "news-1",
            "title": "Title",
            "source": "sina",
            "summary": "Summary",
        }


class FakeLLM:
    def chat(self, *args, **kwargs):
        return "Briefing"


class FakeDatabase:
    def __init__(self):
        self.inserts = []
        self.updates = []

    def insert(self, table, data):
        self.inserts.append((table, data))
        return 42

    def update(self, table, data, where, params):
        self.updates.append((table, data, where, params))
        return 1


def make_service(sender=lambda user_id, content: True):
    milvus = FakeMilvus()
    database = FakeDatabase()
    service = SchedulerService(
        user_service=FakeUserService(),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
        milvus_client=milvus,
        minio_client=FakeMinio(),
        mysql_client=database,
        message_sender=sender,
    )
    return service, milvus, database


def test_date_filter_rejects_invalid_dates():
    assert SchedulerService._build_date_filter("2026-09-12") == (
        'publish_time like "2026-09-12%"'
    )
    with pytest.raises(ValueError):
        SchedulerService._build_date_filter('2026-09-12" or true')


def test_manual_briefing_uses_keyword_and_yesterday_filter():
    service, milvus, _ = make_service()

    result = service.generate_briefing_for_user("user-1")

    assert result == "Briefing"
    assert milvus.calls[0]["filter_expr"].startswith('publish_time like "')
    assert milvus.calls[0]["filter_expr"].endswith('%"')


def test_successful_push_transitions_pending_to_sent():
    service, milvus, database = make_service()

    service._generate_user_briefing(
        {"user_id": "user-1", "interests": [{"keyword": "AI", "weight": 1.0}]},
        "2026-09-12",
    )

    assert milvus.calls[0]["filter_expr"] == 'publish_time like "2026-09-12%"'
    assert database.inserts[0][1]["status"] == "pending"
    assert database.updates[-1][1] == {"status": "sent"}


def test_failed_push_is_recorded_and_raised():
    service, _, database = make_service(sender=lambda user_id, content: False)

    with pytest.raises(RuntimeError, match="发送失败"):
        service._generate_user_briefing(
            {"user_id": "user-1", "interests": [{"keyword": "AI", "weight": 1.0}]},
            "2026-09-12",
        )

    assert database.updates[-1][1] == {"status": "failed"}
