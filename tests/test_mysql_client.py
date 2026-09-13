from contextlib import contextmanager

from storage.mysql_client import MySQLClient


class FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return {"id": 1}

    def fetchall(self):
        return [{"id": 1}, {"id": 2}]


def make_client():
    client = MySQLClient.__new__(MySQLClient)
    cursor = FakeCursor()

    @contextmanager
    def get_cursor():
        yield cursor

    client.get_cursor = get_cursor
    return client


def test_canonical_fetch_methods_match_legacy_aliases():
    client = make_client()

    assert client.fetch_one("SELECT 1") == {"id": 1}
    assert client.fetchone("SELECT 1") == {"id": 1}
    assert client.fetch_all("SELECT 1") == [{"id": 1}, {"id": 2}]
    assert client.fetchall("SELECT 1") == [{"id": 1}, {"id": 2}]
