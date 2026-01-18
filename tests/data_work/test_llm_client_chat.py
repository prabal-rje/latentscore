from data_work.lib import llm_client


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 2
        self.pad_token = None
        self.padding_side = "left"
        self.chat_template = None
        self.last_kwargs = None

    def apply_chat_template(self, *args, **kwargs) -> str:
        _ = args
        self.last_kwargs = kwargs
        return "rendered"


def test_render_chat_prompt_disables_qwen_thinking() -> None:
    tokenizer = DummyTokenizer()
    llm_client.render_chat_prompt(
        system_prompt="sys",
        user_prompt="hello",
        tokenizer=tokenizer,
        model_name="unsloth/Qwen3-0.6B",
        add_generation_prompt=True,
    )
    assert tokenizer.last_kwargs is not None
    assert tokenizer.last_kwargs["enable_thinking"] is False
    assert tokenizer.last_kwargs["add_generation_prompt"] is True


def test_render_chat_prompt_omits_enable_thinking_for_non_qwen() -> None:
    tokenizer = DummyTokenizer()
    llm_client.render_chat_prompt(
        system_prompt="sys",
        user_prompt="hello",
        tokenizer=tokenizer,
        model_name="unsloth/gemma-3-270m-it",
        add_generation_prompt=False,
    )
    assert tokenizer.last_kwargs is not None
    assert "enable_thinking" not in tokenizer.last_kwargs
    assert tokenizer.last_kwargs["add_generation_prompt"] is False


def test_normalize_tokenizer_sets_pad_token_and_template() -> None:
    tokenizer = DummyTokenizer()
    llm_client.normalize_tokenizer_for_model(
        tokenizer=tokenizer,
        model_name="unsloth/gemma-3-270m-it",
    )
    assert tokenizer.pad_token is not None or tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.padding_side == "right"
    assert tokenizer.chat_template == llm_client.GEMMA_CHAT_TEMPLATE


def test_render_chat_prompt_positional_only_drops_unknown_kwargs() -> None:
    class PositionalOnlyTokenizer:
        def __init__(self) -> None:
            self.last_args = None
            self.last_kwargs = None

        def apply_chat_template(self, messages, /, *, tokenize: bool) -> str:
            self.last_args = (messages,)
            self.last_kwargs = {"tokenize": tokenize}
            return "rendered"

    tokenizer = PositionalOnlyTokenizer()
    llm_client.render_chat_prompt(
        system_prompt="sys",
        user_prompt="hello",
        tokenizer=tokenizer,
        model_name="unsloth/Qwen3-0.6B",
        add_generation_prompt=True,
    )
    assert tokenizer.last_args is not None
    assert tokenizer.last_kwargs == {"tokenize": False}
