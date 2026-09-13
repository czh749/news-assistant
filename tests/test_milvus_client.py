from storage.milvus_client import MilvusClient


class FakeHit:
    id = "news-1"
    distance = 0.95
    entity = {
        "news_id": "news-1",
        "title": "Title",
        "source": "sina",
        "publish_time": "2026-09-12T08:00:00",
    }


class FakeCollection:
    def __init__(self):
        self.search_kwargs = None

    def load(self):
        return None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[FakeHit()]]


def test_search_passes_filter_and_returns_requested_fields():
    client = MilvusClient.__new__(MilvusClient)
    client._collection = FakeCollection()

    result = client.search(
        query_vector=[0.1, 0.2],
        output_fields=["news_id", "publish_time"],
        filter_expr='publish_time like "2026-09-12%"',
    )

    assert client._collection.search_kwargs["expr"] == 'publish_time like "2026-09-12%"'
    assert result == [{
        "id": "news-1",
        "distance": 0.95,
        "news_id": "news-1",
        "publish_time": "2026-09-12T08:00:00",
    }]
