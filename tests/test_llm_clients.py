"""agents/BaseLLMClient.py, agents/LLMClients.py, agents/LLMFactory.py.

Every provider SDK (langchain_openai, langchain_huggingface, transformers,
vllm, openai) is replaced by a stub module injected into ``sys.modules`` —
the clients import them lazily inside ``__init__``.

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
import sys
import types

import pytest
import torch

import agents.LLMFactory as factory_module
from agents.BaseLLMClient import BaseLLMClient, LLMResponse
from agents.LLMClients import HuggingFaceClient, LMStudioClient, LocalVLLMClient, OpenAIClient
from agents.LLMFactory import LLMFactory
from tests.support import ScriptedClient

# --------------------------------------------------------------------------- stubs


class _Reply:
    """Minimal LangChain-like reply object: .content plus optional .usage_metadata."""
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture
def fake_langchain_openai(monkeypatch):
    """Stub `langchain_openai.ChatOpenAI` that records its kwargs and replays replies."""

    class ChatOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.invoked = []
            self.reply = _Reply("openai says hi", usage={"input_tokens": 3, "output_tokens": 2})
            ChatOpenAI.instances.append(self)

        def invoke(self, payload):
            self.invoked.append(payload)
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

        def stream(self, prompt):
            return iter(["a", "b"])

    monkeypatch.setitem(sys.modules, "langchain_openai", _module("langchain_openai", ChatOpenAI=ChatOpenAI))
    ChatOpenAI.instances.clear()
    return ChatOpenAI


@pytest.fixture
def fake_langchain_huggingface(monkeypatch):
    class HuggingFaceEndpoint:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            HuggingFaceEndpoint.instances.append(self)

    class ChatHuggingFace:
        instances = []

        def __init__(self, llm):
            self.llm = llm
            self.invoked = []
            self.reply = _Reply("hf says hi")
            ChatHuggingFace.instances.append(self)

        def invoke(self, payload):
            self.invoked.append(payload)
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        _module("langchain_huggingface", HuggingFaceEndpoint=HuggingFaceEndpoint, ChatHuggingFace=ChatHuggingFace),
    )
    HuggingFaceEndpoint.instances.clear()
    ChatHuggingFace.instances.clear()
    return HuggingFaceEndpoint, ChatHuggingFace


class _Batch(dict):
    """What a HF tokenizer returns: a mapping with .to(device)."""

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def fake_transformers(monkeypatch):
    class Tokenizer:
        calls = []
        eos_token_id = 2

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            cls.calls.append(("tokenizer", name, kwargs))
            return cls()

        def __call__(self, prompt, return_tensors=None):
            self.last_prompt = prompt
            return _Batch(input_ids=[[1, 2, 3]])

        def decode(self, ids, skip_special_tokens=False):
            return self.last_prompt + "  the model answer  "

    class Model:
        calls = []
        device = "cpu"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            cls.calls.append(("model", name, kwargs))
            return cls()

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return [[7, 8, 9]]

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    Tokenizer.calls.clear()
    Model.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _module("transformers", AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model, BitsAndBytesConfig=BitsAndBytesConfig),
    )
    return Tokenizer, Model, BitsAndBytesConfig


@pytest.fixture
def fake_vllm(monkeypatch):
    class LLM:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.generated = []
            LLM.instances.append(self)

        def generate(self, prompts, sampling_params):
            self.generated.append((prompts, sampling_params))
            out = types.SimpleNamespace(outputs=[types.SimpleNamespace(text="vllm says hi")])
            return [out]

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    LLM.instances.clear()
    monkeypatch.setitem(sys.modules, "vllm", _module("vllm", LLM=LLM, SamplingParams=SamplingParams))
    return LLM, SamplingParams


@pytest.fixture
def fake_openai(monkeypatch):
    class Completions:
        def __init__(self):
            self.calls = []
            self.reply = "lmstudio says hi"

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("stream"):
                chunks = []
                for piece in ["he", None, "llo"]:
                    delta = types.SimpleNamespace(content=piece)
                    chunks.append(types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)]))
                return iter(chunks)
            if isinstance(self.reply, Exception):
                raise self.reply
            message = types.SimpleNamespace(content=self.reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())
            OpenAI.instances.append(self)

    OpenAI.instances.clear()
    monkeypatch.setitem(sys.modules, "openai", _module("openai", OpenAI=OpenAI))
    return OpenAI


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "prev"},
    {"role": "tool", "content": "ignored"},
]


# --------------------------------------------------------------------------- base


class TestBase:
    """BaseLLMClient / LLMResponse: abstractness, defaults and model info."""

    def test_base_client_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMClient("m")

    def test_get_model_info(self):
        client = ScriptedClient(["x"], model_name="m", temperature=0.4, max_tokens=12)
        assert client.get_model_info() == {"model_name": "m", "temperature": 0.4, "max_tokens": 12, "type": "ScriptedClient"}

    def test_base_defaults(self):
        client = ScriptedClient(["x"])
        assert (client.temperature, client.max_tokens) == (0.0, 64)
        assert BaseLLMClient.__init__.__defaults__ == (0.7, 1000)

    def test_llm_response_defaults(self):
        resp = LLMResponse(content="c", model_name="m")
        assert resp.raw_response is None and resp.usage == {}

    def test_llm_response_requires_model_name(self):
        with pytest.raises(Exception):
            LLMResponse(content="c")


# --------------------------------------------------------------------------- OpenAIClient


class TestOpenAIClient:
    """OpenAIClient against a stubbed langchain_openai.ChatOpenAI."""

    def test_constructor_passes_settings_through(self, fake_langchain_openai):
        client = OpenAIClient(model_name="gpt-x", temperature=0.2, max_tokens=50, api_key="sk")
        assert fake_langchain_openai.instances[0].kwargs == {"model": "gpt-x", "temperature": 0.2, "max_tokens": 50, "api_key": "sk"}
        assert client.model_name == "gpt-x"

    def test_defaults(self, fake_langchain_openai):
        OpenAIClient()
        assert fake_langchain_openai.instances[0].kwargs == {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000, "api_key": None}

    def test_missing_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "langchain_openai", None)
        with pytest.raises(ImportError, match="pip install langchain-openai"):
            OpenAIClient()

    def test_invoke(self, fake_langchain_openai):
        resp = OpenAIClient().invoke("hello")
        assert isinstance(resp, LLMResponse)
        assert resp.content == "openai says hi"
        assert resp.usage == {"input_tokens": 3, "output_tokens": 2}
        assert resp.model_name == "gpt-4"
        assert resp.raw_response.content == "openai says hi"
        assert fake_langchain_openai.instances[0].invoked == ["hello"]

    def test_invoke_without_usage_metadata(self, fake_langchain_openai):
        client = OpenAIClient()
        fake_langchain_openai.instances[0].reply = _Reply("x", usage=None)
        assert client.invoke("q").usage == {}

    def test_invoke_reraises_errors(self, fake_langchain_openai):
        client = OpenAIClient()
        fake_langchain_openai.instances[0].reply = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            client.invoke("q")

    def test_messages_are_converted_and_unknown_roles_dropped(self, fake_langchain_openai):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        client = OpenAIClient()
        resp = client.invoke_with_messages(MESSAGES)
        assert resp.content == "openai says hi"
        sent = fake_langchain_openai.instances[0].invoked[0]
        assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage]
        assert [m.content for m in sent] == ["sys", "hi", "prev"]

    def test_stream_delegates(self, fake_langchain_openai):
        assert list(OpenAIClient().stream("p")) == ["a", "b"]


# --------------------------------------------------------------------------- HuggingFaceClient (API)


class TestHuggingFaceApiClient:
    """HuggingFaceClient in API mode (HuggingFaceEndpoint + ChatHuggingFace stubs)."""

    def test_constructor_builds_endpoint_and_chat(self, fake_langchain_huggingface, capsys):
        Endpoint, Chat = fake_langchain_huggingface
        client = HuggingFaceClient("org/m", temperature=0.1, max_tokens=33, use_api=True, api_token="tok")
        assert Endpoint.instances[0].kwargs == {"repo_id": "org/m", "huggingfacehub_api_token": "tok", "temperature": 0.1, "max_new_tokens": 33}
        assert Chat.instances[0].llm is Endpoint.instances[0]
        assert client.use_api is True and client.api_token == "tok"
        assert "✅ Using HuggingFace API with model: org/m" in capsys.readouterr().out

    def test_missing_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
        with pytest.raises(ImportError, match="pip install langchain-huggingface"):
            HuggingFaceClient("org/m", use_api=True)

    def test_invoke(self, fake_langchain_huggingface):
        _, Chat = fake_langchain_huggingface
        resp = HuggingFaceClient("org/m", use_api=True).invoke("hello")
        assert resp.content == "hf says hi" and resp.model_name == "org/m" and resp.usage == {}
        assert Chat.instances[0].invoked == ["hello"]

    def test_invoke_prints_and_reraises(self, fake_langchain_huggingface, capsys):
        _, Chat = fake_langchain_huggingface
        client = HuggingFaceClient("org/m", use_api=True)
        Chat.instances[0].reply = RuntimeError("down")
        with pytest.raises(RuntimeError):
            client.invoke("q")
        assert "HuggingFace invoke error: down" in capsys.readouterr().out

    def test_messages_converted(self, fake_langchain_huggingface):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        _, Chat = fake_langchain_huggingface
        HuggingFaceClient("org/m", use_api=True).invoke_with_messages(MESSAGES)
        sent = Chat.instances[0].invoked[0]
        assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage]

    def test_stream_not_implemented(self, fake_langchain_huggingface):
        with pytest.raises(NotImplementedError):
            HuggingFaceClient("org/m", use_api=True).stream("p")


# --------------------------------------------------------------------------- HuggingFaceClient (local)


class TestHuggingFaceLocalClient:
    """HuggingFaceClient in local mode: how transformers is called, quantization branches, prompt stripping."""

    def test_plain_cpu_load(self, fake_transformers, capsys):
        Tokenizer, Model, _ = fake_transformers
        HuggingFaceClient("org/m", device="cpu", use_api=False, api_token="tok")
        assert Tokenizer.calls == [("tokenizer", "org/m", {"token": "tok"})]
        assert Model.calls == [("model", "org/m", {"torch_dtype": torch.float32, "device_map": None, "token": "tok"})]
        assert "✅ Using local model on cpu: org/m" in capsys.readouterr().out

    def test_cuda_load_uses_fp16_and_device_map(self, fake_transformers):
        _, Model, _ = fake_transformers
        HuggingFaceClient("org/m", device="cuda", use_api=False)
        assert Model.calls[0][2] == {"torch_dtype": torch.float16, "device_map": "auto", "token": None}

    def test_8bit_load(self, fake_transformers):
        _, Model, BnB = fake_transformers
        HuggingFaceClient("org/m", use_api=False, load_in_8bit=True)
        kwargs = Model.calls[0][2]
        assert isinstance(kwargs["quantization_config"], BnB)
        assert kwargs["quantization_config"].kwargs == {"load_in_8bit": True}
        assert kwargs["device_map"] == "auto"

    def test_4bit_load(self, fake_transformers):
        _, Model, BnB = fake_transformers
        HuggingFaceClient("org/m", use_api=False, load_in_4bit=True)
        assert Model.calls[0][2]["quantization_config"].kwargs == {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}

    def test_8bit_wins_over_4bit(self, fake_transformers):
        _, Model, _ = fake_transformers
        HuggingFaceClient("org/m", use_api=False, load_in_8bit=True, load_in_4bit=True)
        assert Model.calls[0][2]["quantization_config"].kwargs == {"load_in_8bit": True}

    def test_missing_transformers(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match="Please install required packages"):
            HuggingFaceClient("org/m", use_api=False)

    def test_invoke_strips_the_prompt_from_the_decoded_text(self, fake_transformers):
        client = HuggingFaceClient("org/m", use_api=False, temperature=0.0, max_tokens=17)
        resp = client.invoke("What is R0?")
        assert resp.content == "the model answer"
        assert resp.raw_response == "the model answer"
        gen = client.model.generate_kwargs
        assert gen["max_new_tokens"] == 17
        assert gen["temperature"] == 0.0
        assert gen["do_sample"] is False
        assert gen["pad_token_id"] == 2
        assert gen["input_ids"] == [[1, 2, 3]]

    def test_positive_temperature_enables_sampling(self, fake_transformers):
        client = HuggingFaceClient("org/m", use_api=False, temperature=0.5)
        client.invoke("q")
        assert client.model.generate_kwargs["do_sample"] is True

    def test_messages_are_flattened_into_a_transcript(self, fake_transformers):
        client = HuggingFaceClient("org/m", use_api=False)
        client.invoke_with_messages(MESSAGES)
        assert client.tokenizer.last_prompt == "System: sys\nUser: hi\nAssistant: prev\nAssistant: "

    def test_stream_not_implemented(self, fake_transformers):
        with pytest.raises(NotImplementedError):
            HuggingFaceClient("org/m", use_api=False).stream("p")


# --------------------------------------------------------------------------- LocalVLLMClient


class TestLocalVLLMClient:
    """LocalVLLMClient against a stubbed vllm package."""

    def test_constructor(self, fake_vllm):
        LLM, _ = fake_vllm
        client = LocalVLLMClient("org/m", temperature=0.3, max_tokens=9, tensor_parallel_size=2)
        assert LLM.instances[0].kwargs == {"model": "org/m", "tensor_parallel_size": 2}
        assert client.sampling_params.kwargs == {"temperature": 0.3, "max_tokens": 9}

    def test_missing_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vllm", None)
        with pytest.raises(ImportError, match="pip install vllm"):
            LocalVLLMClient("org/m")

    def test_invoke(self, fake_vllm):
        LLM, _ = fake_vllm
        client = LocalVLLMClient("org/m")
        resp = client.invoke("q")
        assert resp.content == "vllm says hi" and resp.raw_response == "vllm says hi"
        prompts, params = LLM.instances[0].generated[0]
        assert prompts == ["q"] and params is client.sampling_params

    def test_invoke_reraises(self, fake_vllm):
        LLM, _ = fake_vllm
        client = LocalVLLMClient("org/m")
        LLM.instances[0].generate = lambda *a: (_ for _ in ()).throw(RuntimeError("oom"))
        with pytest.raises(RuntimeError, match="oom"):
            client.invoke("q")

    def test_messages_flattened(self, fake_vllm):
        LLM, _ = fake_vllm
        LocalVLLMClient("org/m").invoke_with_messages(MESSAGES)
        assert LLM.instances[0].generated[0][0] == ["System: sys\nUser: hi\nAssistant: prev\nAssistant: "]

    def test_stream_not_supported(self, fake_vllm):
        with pytest.raises(NotImplementedError):
            LocalVLLMClient("org/m").stream("p")


# --------------------------------------------------------------------------- LMStudioClient


class TestLMStudioClient:
    """LMStudioClient against a stubbed openai SDK (OpenAI-compatible local server)."""

    def test_constructor(self, fake_openai, capsys):
        client = LMStudioClient(model_name="qwen", temperature=0.2, max_tokens=8, base_url="http://h:1/v1")
        assert fake_openai.instances[0].kwargs == {"base_url": "http://h:1/v1", "api_key": "not-needed"}
        assert client.base_url == "http://h:1/v1"
        assert "✅ Connected to LM Studio at http://h:1/v1" in capsys.readouterr().out

    def test_defaults(self, fake_openai):
        client = LMStudioClient()
        assert (client.model_name, client.temperature, client.max_tokens, client.base_url) == ("local-model", 0.7, 1000, "http://127.0.0.1:1234/v1")

    def test_missing_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="pip install openai"):
            LMStudioClient()

    def test_invoke_wraps_prompt_as_a_user_message(self, fake_openai):
        client = LMStudioClient(model_name="qwen", temperature=0.2, max_tokens=8)
        resp = client.invoke("q")
        assert resp.content == "lmstudio says hi" and resp.model_name == "qwen"
        assert client.client.chat.completions.calls == [
            {"model": "qwen", "messages": [{"role": "user", "content": "q"}], "temperature": 0.2, "max_tokens": 8}
        ]

    def test_invoke_reraises(self, fake_openai):
        client = LMStudioClient()
        client.client.chat.completions.reply = ConnectionError("refused")
        with pytest.raises(ConnectionError):
            client.invoke("q")

    def test_messages_passed_through_untouched(self, fake_openai):
        client = LMStudioClient()
        client.invoke_with_messages(MESSAGES)
        assert client.client.chat.completions.calls[0]["messages"] is MESSAGES

    def test_stream_yields_non_empty_deltas(self, fake_openai):
        client = LMStudioClient()
        assert list(client.stream("q")) == ["he", "llo"]
        assert client.client.chat.completions.calls[0]["stream"] is True


# --------------------------------------------------------------------------- LLMFactory


@pytest.fixture
def recording_clients(monkeypatch):
    """Replace the four client classes inside LLMFactory with recorders."""
    made = []

    def _recorder(name):
        def factory(**kwargs):
            made.append((name, kwargs))
            return ScriptedClient(["x"], model_name=kwargs["model_name"])

        return factory

    for name in ("OpenAIClient", "HuggingFaceClient", "LocalVLLMClient", "LMStudioClient"):
        monkeypatch.setattr(factory_module, name, _recorder(name))
    return made


class TestLLMFactory:
    """Provider routing, per-provider defaults and the from_config() unpacking of LLMFactory."""

    def test_openai_routing_and_defaults(self, recording_clients, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        LLMFactory.create_client("openai")
        assert recording_clients == [("OpenAIClient", {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000, "api_key": "env-key"})]

    def test_openai_explicit_key_wins(self, recording_clients, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        LLMFactory.create_client("openai", model_name="gpt-x", api_key="explicit")
        assert recording_clients[0][1]["api_key"] == "explicit"
        assert recording_clients[0][1]["model_name"] == "gpt-x"

    def test_huggingface_routing_and_defaults(self, recording_clients):
        LLMFactory.create_client("huggingface")
        assert recording_clients == [
            (
                "HuggingFaceClient",
                {
                    "model_name": "microsoft/phi-2",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "device": "cpu",
                    "use_api": False,
                    "api_token": None,
                    "load_in_8bit": False,
                    "load_in_4bit": False,
                },
            )
        ]

    def test_huggingface_kwargs_forwarded(self, recording_clients):
        LLMFactory.create_client("huggingface", model_name="org/m", temperature=0.1, max_tokens=5, device="cuda", use_api=True, api_token="t", load_in_4bit=True)
        assert recording_clients[0][1] == {
            "model_name": "org/m",
            "temperature": 0.1,
            "max_tokens": 5,
            "device": "cuda",
            "use_api": True,
            "api_token": "t",
            "load_in_8bit": False,
            "load_in_4bit": True,
        }

    def test_vllm_routing_and_defaults(self, recording_clients):
        LLMFactory.create_client("vllm")
        assert recording_clients == [("LocalVLLMClient", {"model_name": "meta-llama/Llama-2-7b-chat-hf", "temperature": 0.7, "max_tokens": 1000, "tensor_parallel_size": 1})]

    def test_lmstudio_routing_and_defaults(self, recording_clients):
        LLMFactory.create_client("lmstudio")
        assert recording_clients == [("LMStudioClient", {"model_name": "local-model", "temperature": 0.7, "max_tokens": 1000, "base_url": "http://127.0.0.1:1234/v1"})]

    def test_lmstudio_base_url_forwarded(self, recording_clients):
        LLMFactory.create_client("lmstudio", base_url="http://h:9/v1")
        assert recording_clients[0][1]["base_url"] == "http://h:9/v1"

    def test_unknown_provider_message_omits_lmstudio(self, recording_clients):
        # NOTE: current behavior — lmstudio is supported but not listed in the error text
        with pytest.raises(ValueError) as excinfo:
            LLMFactory.create_client("banana")
        assert str(excinfo.value) == "Unknown provider: banana. Supported: openai, huggingface, vllm"
        assert recording_clients == []

    def test_provider_is_case_sensitive(self, recording_clients):
        with pytest.raises(ValueError):
            LLMFactory.create_client("OpenAI")

    def test_from_config_unpacks_kwargs(self, recording_clients):
        cfg = {"provider": "lmstudio", "model_name": "q", "temperature": 0.2, "max_tokens": 9, "kwargs": {"base_url": "http://h:1/v1"}}
        client = LLMFactory.from_config(cfg)
        assert recording_clients == [("LMStudioClient", {"model_name": "q", "temperature": 0.2, "max_tokens": 9, "base_url": "http://h:1/v1"})]
        assert client.model_name == "q"

    def test_from_config_defaults_to_huggingface(self, recording_clients):
        LLMFactory.from_config({})
        assert recording_clients[0][0] == "HuggingFaceClient"
        assert recording_clients[0][1]["model_name"] == "microsoft/phi-2"
        assert recording_clients[0][1]["temperature"] == 0.7

    def test_from_config_with_the_real_config_module(self, recording_clients, config_env):
        cfg = config_env(LLM_PROVIDER="lmstudio", LMSTUDIO_MODEL="q")
        LLMFactory.from_config(cfg.LLM_CONFIG)
        assert recording_clients[0] == ("LMStudioClient", {"model_name": "q", "temperature": 0.0, "max_tokens": 1024, "base_url": "http://127.0.0.1:1234/v1"})


# --------------------------------------------------------------------------- error paths & abstract bodies


class TestAbstractBodies:
    """The abstract methods of BaseLLMClient have `pass` bodies: calling them through the
    base class on a concrete instance returns None."""

    def test_abstract_methods_do_nothing(self):
        client = ScriptedClient(["x"])
        assert BaseLLMClient.invoke(client, "p") is None
        assert BaseLLMClient.invoke_with_messages(client, []) is None
        assert BaseLLMClient.stream(client, "p") is None


class TestErrorPropagation:
    """Every client logs and re-raises provider errors instead of swallowing them."""

    def test_openai_messages_error(self, fake_langchain_openai):
        client = OpenAIClient()
        fake_langchain_openai.instances[0].reply = RuntimeError("quota")
        with pytest.raises(RuntimeError, match="quota"):
            client.invoke_with_messages(MESSAGES)

    def test_lmstudio_messages_error(self, fake_openai):
        client = LMStudioClient()
        client.client.chat.completions.reply = RuntimeError("offline")
        with pytest.raises(RuntimeError, match="offline"):
            client.invoke_with_messages(MESSAGES)

    def test_lmstudio_stream_error(self, fake_openai, monkeypatch):
        client = LMStudioClient()

        def boom(**kwargs):
            raise RuntimeError("stream broke")

        monkeypatch.setattr(client.client.chat.completions, "create", boom)
        with pytest.raises(RuntimeError, match="stream broke"):
            list(client.stream("p"))
