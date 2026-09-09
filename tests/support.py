"""Helpers shared by the characterization tests (importable as ``tests.support``)."""
import json
from typing import Callable, List, Union

from agents.BaseLLMClient import BaseLLMClient, LLMResponse
from formats.data_formats import Episode

# --------------------------------------------------------------------------- JSON replies


def intent_json(
    cares_position: bool = True,
    position: str = "later",
    cares_height: bool = True,
    height: str = "lower",
    primary: str = "both",
    reasoning: str = "scripted intent",
) -> str:
    """A valid ``ExpertIntent`` JSON document, as the intent LLM should answer."""
    return json.dumps(
        {
            "cares_about_position": cares_position,
            "position_direction": position,
            "cares_about_height": cares_height,
            "height_direction": height,
            "primary_metric": primary,
            "reasoning": reasoning,
        }
    )


def params_json(
    beta: float,
    gamma: float,
    mu: float,
    reasoning: str = "scripted parameters",
    confidence: str = "high",
) -> str:
    """A valid ``EpiParameters`` JSON document, as the generator LLM should answer."""
    return json.dumps(
        {
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "reasoning": reasoning,
            "confidence": confidence,
        }
    )


def critic_json(
    decision: str = "accept",
    reasoning: str = "scripted verdict",
    confidence: str = "high",
    issues: Union[List[str], None] = None,
) -> str:
    """A valid ``CriticOutput`` JSON document, as the LLM critic should answer."""
    return json.dumps(
        {
            "reasoning": reasoning,
            "decision": decision,
            "confidence": confidence,
            "issues": issues or [],
        }
    )


GENERATOR_MARKER = "SIRD model parameter optimization"  # only in the generator's system prompt


def route_by_prompt(intent_reply: str, params_reply: Union[str, Callable[[str], str]]):
    """Build a prompt -> reply function: generator prompts get ``params_reply``,
    every other prompt (intent parser, critic fallback) gets ``intent_reply``."""

    def _reply(prompt: str) -> str:
        if GENERATOR_MARKER in prompt:
            return params_reply(prompt) if callable(params_reply) else params_reply
        return intent_reply

    return _reply


# --------------------------------------------------------------------------- fake clients


class ScriptedClient(BaseLLMClient):
    """``BaseLLMClient`` stand-in: answers from a script and records every prompt.

    ``script`` is either a list of strings (consumed in order, the last one
    repeats) or a callable ``prompt -> str``.
    """

    def __init__(self, script, model_name: str = "scripted-model", temperature: float = 0.0, max_tokens: int = 64):
        super().__init__(model_name, temperature, max_tokens)
        self.script = script
        self.prompts: List[str] = []
        self._i = 0

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        if callable(self.script):
            content = self.script(prompt)
        else:
            content = self.script[min(self._i, len(self.script) - 1)]
            self._i += 1
        return LLMResponse(content=content, raw_response=content, model_name=self.model_name)

    def invoke_with_messages(self, messages, **kwargs) -> LLMResponse:
        return self.invoke("\n".join(m["content"] for m in messages))

    def stream(self, prompt: str, **kwargs):
        yield self.invoke(prompt).content


# --------------------------------------------------------------------------- episodes


def make_episode(**overrides) -> Episode:
    """An ``Episode`` with realistic defaults (baseline-like) that can be overridden."""
    fields = dict(
        beta=0.091,
        gamma=0.0553,
        mu=0.0085,
        reasoning="Baseline parameters from initial input",
        peak_position=120.0,
        peak_height=10000.0,
        total_deaths=500.0,
        expert_comment="Need higher peak",
        accepted=True,
        iteration=0,
    )
    fields.update(overrides)
    return Episode(**fields)
