import logging


class DummyTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "PROMPT"


def test_mlx_prompt_uses_roles_and_warns(caplog):
    from latentscore import models

    models._chat_role_warning_emitted = False
    caplog.set_level(logging.WARNING, logger="latentscore.models")

    tokenizer = DummyTokenizer()
    prompt = models._build_chat_prompt(
        system_prompt="SYS",
        vibe="glow",
        tokenizer=tokenizer,
    )
    assert prompt == "PROMPT"
    assert tokenizer.messages[0]["role"] == "system"
    assert "<vibe>" in tokenizer.messages[1]["content"]
    assert "chat roles" in caplog.text
