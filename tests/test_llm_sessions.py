from types import SimpleNamespace

import LLM as llm_module


class FakeChatModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return SimpleNamespace(content=f"reply-{len(self.calls)}")


def make_llm(monkeypatch):
    model = FakeChatModel()
    monkeypatch.setattr(llm_module.settings, "ZHIPU_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "ChatZhipuAI", lambda **kwargs: model)
    llm = llm_module.ZhipuLLM(max_history_messages=6)
    llm.min_request_interval = 0
    return llm, model


def test_calls_without_session_are_stateless(monkeypatch):
    llm, model = make_llm(monkeypatch)

    llm.chat("first")
    llm.chat("second")

    assert [len(call) for call in model.calls] == [1, 1]
    assert llm.get_history() == []


def test_histories_are_isolated_by_session(monkeypatch):
    llm, model = make_llm(monkeypatch)

    llm.chat("u1-first", system_prompt="system", session_id="u1")
    llm.chat("u2-first", system_prompt="system", session_id="u2")
    llm.chat("u1-second", session_id="u1")

    assert [len(call) for call in model.calls] == [2, 2, 4]
    assert len(llm.get_history("u1")) == 5
    assert len(llm.get_history("u2")) == 3

    llm.clear_history("u1")
    assert llm.get_history("u1") == []
    assert len(llm.get_history("u2")) == 3
