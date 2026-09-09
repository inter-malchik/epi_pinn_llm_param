"""utils/RetryParser.py and utils/PromptLogger.py.

Reading guide
-------------
* These are characterization tests: they pin the CURRENT behavior of the code,
  quirks included. A comment starting with ``# NOTE: current behavior`` marks a
  place where the pinned behavior looks like a bug. Fix the code first, then
  update the test on purpose - never "fix" the test to hide the change.
* Test names read as sentences (``test_zero_beta_is_treated_as_missing``); a
  test class groups the tests of one function, method or scenario.
* Only external boundaries are faked (LLM providers, the network). The SIRD
  solver, the PINN, LangGraph and the file system are real. Shared fixtures
  live in ``tests/conftest.py``, fake LLM clients and JSON reply builders in
  ``tests/support.py``.
"""
import json
import re
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from formats.data_formats import CriticOutput, EpiParameters
from tests.support import critic_json, make_episode, params_json
from utils.PromptLogger import PromptLogger
from utils.RetryParser import RetryParser

# --------------------------------------------------------------------------- RetryParser


class RecordingLLM:
    """Minimal stand-in exposing the two things RetryParser needs: bind() and invoke()."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # (prompt, kwargs)

    def bind(self, **kwargs):
        def _invoke(prompt, **kw):
            self.calls.append((prompt, {**kwargs, **kw}))
            return self.replies.pop(0)

        return RunnableLambda(_invoke)


@pytest.fixture
def params_parser():
    return PydanticOutputParser(pydantic_object=EpiParameters)


class TestRetryParser:
    """Retry/back-off logic around a Pydantic output parser; sleeps are patched out by conftest."""

    def test_first_attempt_success_does_not_call_the_llm(self, params_parser):
        llm = RecordingLLM([])
        parser = RetryParser(llm, params_parser)
        out = parser.parse(params_json(0.1, 0.05, 0.005), "prompt")
        assert isinstance(out, EpiParameters) and out.beta == 0.1
        assert llm.calls == []

    def test_defaults(self, params_parser):
        parser = RetryParser(RecordingLLM([]), params_parser)
        assert parser.max_retries == 3 and parser.delay == 1

    def test_retry_uses_lower_temperature_and_the_original_prompt(self, params_parser):
        llm = RecordingLLM([params_json(0.2, 0.05, 0.005)])
        parser = RetryParser(llm, params_parser, retry_temperature=0.3)
        out = parser.parse("not json at all", "the original prompt")
        assert out.beta == 0.2
        assert llm.calls == [("the original prompt", {"temperature": 0.3})]

    def test_backoff_schedule_is_one_then_two_seconds(self, params_parser, monkeypatch):
        import utils.RetryParser as retry_module

        sleeps = []
        monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)
        llm = RecordingLLM(["still garbage", params_json(0.3, 0.05, 0.005)])
        out = RetryParser(llm, params_parser, delay=1).parse("garbage", "prompt")
        assert out.beta == 0.3
        assert sleeps == [1, 2]
        assert len(llm.calls) == 2

    def test_custom_delay_scales_the_backoff(self, params_parser, monkeypatch):
        import utils.RetryParser as retry_module

        sleeps = []
        monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)
        llm = RecordingLLM(["garbage", params_json(0.3, 0.05, 0.005)])
        RetryParser(llm, params_parser, delay=5).parse("garbage", "prompt")
        assert sleeps == [5, 10]

    def test_gives_up_after_max_retries(self, params_parser):
        llm = RecordingLLM(["bad", "bad", "bad"])
        with pytest.raises(OutputParserException) as excinfo:
            RetryParser(llm, params_parser, max_retries=3).parse("bad", "prompt")
        assert "Failed to parse after 3 attempts" in str(excinfo.value)
        # attempts 1 and 2 re-ask the model; the third failure raises
        assert len(llm.calls) == 2

    def test_without_prompt_text_there_is_no_retry(self, params_parser, capsys):
        # NOTE: current behavior — the message still claims "after 3 attempts"
        llm = RecordingLLM([params_json(0.3, 0.05, 0.005)])
        with pytest.raises(OutputParserException) as excinfo:
            RetryParser(llm, params_parser).parse("bad")
        assert "Failed to parse after 3 attempts" in str(excinfo.value)
        assert llm.calls == []
        assert "⚠️ Critic retry 1/3" in capsys.readouterr().out

    def test_diagnostic_is_always_labelled_critic(self, params_parser, capsys):
        # NOTE: current behavior — the generator uses this class too, but the log says "Critic retry"
        llm = RecordingLLM([params_json(0.3, 0.05, 0.005)])
        RetryParser(llm, params_parser).parse("bad", "prompt")
        out = capsys.readouterr().out
        assert "⚠️ Critic retry 1/3" in out
        assert "⏳ Waiting 1 seconds" in out

    def test_retry_response_with_content_attribute(self, params_parser, chat_llm):
        llm = chat_llm(params_json(0.4, 0.05, 0.005))
        out = RetryParser(llm, params_parser).parse("bad", "prompt")
        assert out.beta == 0.4

    def test_validation_error_is_retried_like_a_parse_error(self, params_parser):
        llm = RecordingLLM([params_json(0.5, 0.05, 0.005)])
        out = RetryParser(llm, params_parser).parse(params_json(5.0, 0.05, 0.005), "prompt")
        assert out.beta == 0.5

    def test_works_with_critic_output_parser(self):
        parser = PydanticOutputParser(pydantic_object=CriticOutput)
        out = RetryParser(RecordingLLM([]), parser).parse(critic_json("Accept"), "p")
        assert out.decision == "accept"


# --------------------------------------------------------------------------- PromptLogger


class TestPromptLoggerSetup:
    """Directory layout and timestamp format of PromptLogger."""

    def test_creates_generator_and_critic_dirs_under_cwd(self, tmp_path, capsys):
        PromptLogger()
        assert (tmp_path / "logs" / "prompts" / "generator").is_dir()
        assert (tmp_path / "logs" / "prompts" / "critic").is_dir()
        assert "📝 PromptLogger initialized. Logs:" in capsys.readouterr().out

    def test_custom_log_dir(self, tmp_path):
        logger = PromptLogger(log_dir="custom/dir")
        assert logger.log_dir == Path("custom/dir")
        assert (tmp_path / "custom" / "dir" / "critic").is_dir()

    def test_timestamp_format(self):
        ts = PromptLogger()._get_timestamp()
        assert re.fullmatch(r"\d{8}_\d{6}_\d{3}", ts)


class TestPromptLoggerMetadata:
    """Which model metadata gets attached to a log record - always read from the config module, never from the LLM object."""

    def test_huggingface_metadata_comes_from_config(self, config_env):
        cfg = config_env(MODEL_NAME_HF="org/m", MODEL_TEMPERATURE_HF="0.4", MAX_TOKENS="12", HF_USE_API="false", HF_DEVICE="mps")
        assert PromptLogger()._extract_llm_metadata(object()) == {
            "provider": "huggingface",
            "model": "org/m",
            "temperature": 0.4,
            "max_tokens": 12,
            "use_api": False,
            "device": "mps",
        }
        assert cfg.LLM_PROVIDER == "huggingface"

    def test_openai_metadata(self, config_env):
        config_env(LLM_PROVIDER="openai", OPENAI_MODEL="gpt-x", MODEL_TEMPERATURE_HF="0.2")
        assert PromptLogger()._extract_llm_metadata(object()) == {
            "provider": "openai",
            "model": "gpt-x",
            "temperature": 0.2,
            "max_tokens": 1024,
        }

    def test_vllm_metadata(self, config_env):
        config_env(LLM_PROVIDER="vllm", VLLM_MODEL="v", VLLM_TEMPERATURE="0.6", VLLM_MAX_TOKENS="8", VLLM_TENSOR_PARALLEL="4")
        assert PromptLogger()._extract_llm_metadata(object()) == {
            "provider": "vllm",
            "model": "v",
            "temperature": 0.6,
            "max_tokens": 8,
            "tensor_parallel_size": 4,
        }

    def test_lmstudio_metadata(self, config_env):
        config_env(LLM_PROVIDER="lmstudio", LMSTUDIO_MODEL="q", LMSTUDIO_TEMPERATURE="0.1", LMSTUDIO_MAX_TOKENS="7", LMSTUDIO_BASE_URL="http://h:1/v1")
        assert PromptLogger()._extract_llm_metadata(object()) == {
            "provider": "lmstudio",
            "model": "q",
            "temperature": 0.1,
            "max_tokens": 7,
            "base_url": "http://h:1/v1",
        }

    def test_unknown_provider_gives_only_the_generic_block(self, config_env):
        config_env(LLM_PROVIDER="banana", MODEL_NAME_HF="org/m")
        assert PromptLogger()._extract_llm_metadata(object()) == {
            "provider": "banana",
            "model": "org/m",
            "temperature": 0.0,
            "max_tokens": 1024,
        }

    def test_the_llm_argument_itself_is_ignored(self, config_env):
        # NOTE: current behavior — metadata is read from `config`, never from the LLM object
        config_env(MODEL_NAME_HF="org/m")
        logger = PromptLogger()

        class Other:
            model_name = "runtime-model"
            temperature = 0.99

        assert logger._extract_llm_metadata(Other()) == logger._extract_llm_metadata("anything")


def _read_only_json(directory: Path) -> dict:
    files = list(directory.glob("*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


class TestPromptLoggerWrites:
    """File naming and JSON payload of the three log methods, including the same-millisecond collision."""

    def test_generator_log_file_and_payload(self, tmp_path, config_env):
        config_env(MODEL_NAME_HF="org/m")
        logger = PromptLogger()
        parsed = EpiParameters(beta=0.1, gamma=0.05, mu=0.005, reasoning="why", confidence="low")
        path = logger.log_generator_prompt(
            prompt_text="PROMPT", response_text="RAW", parsed_output=parsed,
            context={"expert_comment": "Нужен пик выше"}, iteration=3,
            metadata={"note": "x", "model": "overridden-by-config"}, llm=object(),
        )
        assert Path(path).name.startswith("generator_iter_003_")
        assert not Path(path).is_absolute()  # relative to cwd
        assert Path(path).resolve().parent == (tmp_path / "logs" / "prompts" / "generator").resolve()
        data = _read_only_json(logger.generator_dir)
        assert data["agent"] == "generator"
        assert data["iteration"] == 3
        assert data["prompt"] == "PROMPT" and data["response_raw"] == "RAW"
        assert data["parsed_output"] == {"beta": 0.1, "gamma": 0.05, "mu": 0.005, "reasoning": "why", "confidence": "low"}
        assert data["context"] == {"expert_comment": "Нужен пик выше"}
        # config metadata wins over caller metadata on key collisions
        assert data["metadata"]["note"] == "x"
        assert data["metadata"]["model"] == "org/m"
        assert data["metadata"]["provider"] == "huggingface"
        assert "Нужен пик выше" in Path(path).read_text(encoding="utf-8")  # ensure_ascii=False

    def test_generator_log_without_llm_has_only_caller_metadata(self):
        logger = PromptLogger()
        logger.log_generator_prompt("p", "r", {"beta": 1}, {}, metadata={"a": 1})
        assert _read_only_json(logger.generator_dir)["metadata"] == {"a": 1}

    def test_generator_log_without_any_metadata(self):
        logger = PromptLogger()
        logger.log_generator_prompt("p", "r", {"beta": 1}, {})
        data = _read_only_json(logger.generator_dir)
        assert data["metadata"] == {} and data["iteration"] == 0

    def test_dict_parsed_output_is_stored_verbatim(self):
        logger = PromptLogger()
        logger.log_generator_prompt("p", "r", {"any": "thing"}, {})
        assert _read_only_json(logger.generator_dir)["parsed_output"] == {"any": "thing"}

    def test_non_serializable_context_is_stringified(self):
        logger = PromptLogger()
        ep = make_episode(timestamp="ts")
        logger.log_generator_prompt("p", "r", {}, {"episode": ep})
        assert _read_only_json(logger.generator_dir)["context"]["episode"] == str(ep)

    def test_critic_log_file_and_payload(self, tmp_path):
        logger = PromptLogger()
        parsed = CriticOutput(reasoning="r", decision="reject", confidence="low", issues=["x"])
        path = logger.log_critic_prompt("P", "R", parsed, {"k": 1}, iteration=2, metadata={"decision": "reject"})
        assert Path(path).name.startswith("critic_iter_002_")
        assert Path(path).resolve().parent == (tmp_path / "logs" / "prompts" / "critic").resolve()
        data = _read_only_json(logger.critic_dir)
        assert data["agent"] == "critic"
        assert data["parsed_output"] == {"reasoning": "r", "decision": "reject", "confidence": "low", "issues": ["x"]}
        assert data["metadata"] == {"decision": "reject"}

    def test_attempt_log(self, tmp_path):
        logger = PromptLogger()
        path = logger.log_generation_attempt(iteration=5, params={"beta": 0.1}, reasoning="why", decision="accept")
        assert Path(path).name.startswith("attempt_iter_005_")
        assert Path(path).resolve().parent == (tmp_path / "logs" / "prompts" / "generator").resolve()
        data = _read_only_json(logger.generator_dir)
        assert data["agent"] == "generator" and data["type"] == "attempt"
        assert data["params"] == {"beta": 0.1}
        assert data["reasoning"] == "why" and data["decision"] == "accept"
        assert data["metadata"] == {}

    def test_distinct_timestamps_give_distinct_files(self, monkeypatch):
        logger = PromptLogger()
        stamps = iter(["20260101_000000_001", "20260101_000000_002"])
        monkeypatch.setattr(logger, "_get_timestamp", lambda: next(stamps))
        logger.log_generator_prompt("p", "r", {}, {}, iteration=1)
        logger.log_generator_prompt("p", "r", {}, {}, iteration=1)
        assert sorted(f.name for f in logger.generator_dir.glob("*.json")) == [
            "generator_iter_001_20260101_000000_001.json",
            "generator_iter_001_20260101_000000_002.json",
        ]

    def test_writes_within_the_same_millisecond_overwrite_each_other(self, monkeypatch):
        # NOTE: current behavior — possible bug: the filename has millisecond resolution only,
        # so two logs in the same millisecond collide and the first one is lost
        logger = PromptLogger()
        monkeypatch.setattr(logger, "_get_timestamp", lambda: "20260101_000000_001")
        logger.log_generator_prompt("first", "r", {}, {}, iteration=1)
        logger.log_generator_prompt("second", "r", {}, {}, iteration=1)
        assert _read_only_json(logger.generator_dir)["prompt"] == "second"

    def test_saved_message_is_printed(self, capsys):
        logger = PromptLogger()
        logger.log_critic_prompt("p", "r", {}, {})
        assert "💾 Critic log saved:" in capsys.readouterr().out
