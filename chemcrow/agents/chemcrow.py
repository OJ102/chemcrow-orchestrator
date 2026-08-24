import os
from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools

# Local model served by Ollama, used by default. Ollama exposes an
# OpenAI-compatible /v1 endpoint, so no OpenAI account/key is needed.
#
# qwen3.8:27b requires a newer Ollama server than the system-wide install at
# /usr/local/bin/ollama (root-owned, no sudo available to upgrade it), so it
# runs on a separate user-space Ollama daemon on port 11435 instead of the
# default 11434 -- see /home3/jadhavor/ollama-local (binary + its own model
# store). That daemon must be running for this to work; it is not a system
# service, so it won't survive a reboot on its own. Start it with:
#   OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/home3/jadhavor/ollama-local/models \
#     /home3/jadhavor/ollama-local/extracted/bin/ollama serve
CHEMCROW_MODEL = os.getenv("CHEMCROW_MODEL", "qwen3.8:27b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435/v1")


def _make_llm(
    model, temp, api_key, streaming: bool = False, ollama_base_url: Optional[str] = None
):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )
    elif ollama_base_url:
        # Any other model name is treated as a local Ollama model. Ollama's
        # /v1 endpoint speaks the OpenAI chat-completions API, so the same
        # ChatOpenAI client works unmodified as transport -- this avoids
        # needing `ChatOllama`, which doesn't exist at this package's pinned
        # langchain version.
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key or "ollama",
            openai_api_base=ollama_base_url,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model=CHEMCROW_MODEL,
        tools_model=CHEMCROW_MODEL,
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        local_rxn: bool = False,
        ollama_base_url: Optional[str] = OLLAMA_BASE_URL,
    ):
        """Initialize ChemCrow agent.

        By default this runs entirely against a local Ollama model
        (`CHEMCROW_MODEL`, served at `ollama_base_url`) and needs no OpenAI
        API key. Pass a `gpt-*`/`text-*` model name (and `openai_api_key`)
        to use OpenAI instead; pass `ollama_base_url=None` to disable the
        Ollama fallback and get the original "Invalid model name" error for
        unrecognized model strings.
        """

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming, ollama_base_url)
        except ValidationError:
            raise ValueError("Invalid OpenAI API key")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(
                tools_model, temp, openai_api_key, streaming, ollama_base_url
            )
            tools = make_tools(tools_llm, api_keys=api_keys, local_rxn=local_rxn, verbose=verbose)

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt, callbacks=None):
        outputs = self.agent_executor({"input": prompt}, callbacks=callbacks)
        return outputs["output"]
