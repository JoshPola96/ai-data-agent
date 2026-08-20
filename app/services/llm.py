# app/services/llm.py

"""
LLM Service v5.1
Separated tool calling from structured output enforcement
"""

import json
import logging
import asyncio
import random
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from google import genai
from google.genai import types
from app.core.config import get_settings
import uuid

from app.utils.schemas import FinalResponseSchema

settings = get_settings()
logger = logging.getLogger(__name__)

# Retry decided by HTTP status alone. Both providers surface the code on the exception (`status_code` for OpenAI, `code` for google-genai)
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient(exc: Exception) -> bool:
    """True when retrying the same provider has a realistic chance of succeeding."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def _provider_reason(exc: Optional[Exception]) -> str:
    """
    A short, honest description of why a provider refused, safe to show a user.

    The provider's own message is the only place the distinction lives — an exhausted
    key and a rate limit are both 429 — but those messages run to several lines of JSON
    with URLs, so only the leading sentence is surfaced. None of this drives control
    flow; retry and failover decide on status alone.
    """
    if exc is None:
        return "no response"
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = " ".join(str(exc).split())
    if len(text) > 140:
        text = text[:137] + "..."
    return f"HTTP {status}: {text}" if isinstance(status, int) else text or type(exc).__name__


async def _with_retry(call, label: str):
    """Retry transient provider faults with exponential backoff and jitter."""
    for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
        try:
            return await call()
        except Exception as e:
            if attempt >= settings.LLM_MAX_RETRIES or not _is_transient(e):
                raise
            wait = settings.LLM_RETRY_BASE_DELAY * 2 ** (attempt - 1) + random.uniform(
                0, 0.3
            )
            logger.warning(
                f"⚠️ {label} transient fault ({type(e).__name__}), retry {attempt}/{settings.LLM_MAX_RETRIES - 1} in {wait:.1f}s"
            )
            await asyncio.sleep(wait)


# Non-strict json_schema: strict mode requires fully-specified object schemas, which the free-form `custom_data` tool argument cannot satisfy.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_response",
        "schema": FinalResponseSchema.model_json_schema(),
    },
}


# Request parameters a model may refuse. Reasoning models reject any temperature but the
# default; older ones reject json_schema response formats or the newer token limit. Which
# is which changes with every release, so the set is discovered per model at runtime rather
# than maintained as a compatibility matrix that is wrong the week after it is written.
_DROPPABLE = ("temperature", "response_format", "max_completion_tokens", "tool_choice")

# Query expansion is the one call that wants variety: its whole job is to phrase the same
# question several ways so lexical search has more than one shot at the corpus. The agent
# loop runs at LLM_TEMPERATURE, which defaults to zero, because there repeatability is the
# point — identical questions should not get differently reasoned answers.
_EXPANSION_TEMPERATURE = 0.3


class LLMService:
    def __init__(self):
        self.openai_client: Optional[AsyncOpenAI] = None
        self.gemini_client: Optional[genai.Client] = None
        self.model = settings.DEFAULT_MODEL
        self.fallback_model = settings.FALLBACK_MODEL
        self._initialized = False
        # {model: {parameter it rejected}}, learned from the provider's own complaint
        self._unsupported: Dict[str, set] = {}

    async def _openai_create(self, kwargs: Dict[str, Any]):
        """
        Call OpenAI, dropping any parameter this model says it will not accept.

        The provider names the offending parameter in the 400, and it is the only thing
        that knows. Reading it is safe in a way that reading provider prose generally is
        not: the worst case is that nothing matches, the retry is skipped, and the call
        fails over exactly as it would have. Nothing silently changes meaning.
        """
        model = kwargs["model"]
        for parameter in self._unsupported.get(model, ()):
            kwargs.pop(parameter, None)

        try:
            return await self.openai_client.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                raise
            complaint = str(e).lower()
            offending = next(
                (p for p in _DROPPABLE if p in kwargs and f"'{p}'" in complaint), None
            )
            if not offending:
                raise
            logger.warning(f"⚠️ {model} rejects '{offending}', retrying without it")
            self._unsupported.setdefault(model, set()).add(offending)
            kwargs.pop(offending)
            return await self.openai_client.chat.completions.create(**kwargs)

    async def initialize(self):
        """Initialize API clients"""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("🧠 INITIALIZING LLM SERVICE")
        logger.info("=" * 60)

        if settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("✓ OpenAI client initialized")

        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info("✓ Gemini client initialized")

        self._initialized = True
        logger.info(f"✓ Default model: {self.model}")
        logger.info("=" * 60)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        # Defaulted from settings rather than a literal, so a caller that omits it cannot
        # silently run at a different temperature than the one configured.
        temperature: float = settings.LLM_TEMPERATURE,
        json_mode: bool = False,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Universal chat wrapper with intelligent routing"""
        await self.initialize()

        active_model = model_override or self.model

        logger.debug(f"🤖 LLM Call: model={active_model}, temp={temperature}")
        logger.debug(
            f"   Messages: {len(messages)}, Tools: {len(tools) if tools else 0}"
        )

        # Route based on model
        if "gemini" in active_model.lower():
            result = await self._chat_gemini(
                messages,
                system_prompt,
                tools,
                temperature,
                json_mode,
                active_model,
            )
        else:
            result = await self._chat_openai(
                messages,
                system_prompt,
                tools,
                temperature,
                json_mode,
                active_model,
            )

        # Report which model actually answered
        result.setdefault("model", active_model)
        return result

    async def generate_query_variants(self, query: str, n: int) -> List[str]:
        """Expand a query into n total phrasings to widen retrieval recall."""
        if n <= 1:
            return [query]

        await self.initialize()

        prompt = (
            f"Rewrite the search query below into {n - 1} alternative phrasings that a semantic "
            "search engine would match against differently worded passages of the same intent.\n"
            "Keep the original language, entities, numbers and dates exactly as written.\n"
            'Return JSON: {"queries": ["...", "..."]}\n\n'
            f"Query: {query}"
        )

        try:
            raw = await self._complete_json(prompt)
            variants = json.loads(raw).get("queries", [])
        except Exception as e:
            logger.warning(f"⚠️ Query expansion failed, searching original only: {e}")
            return [query]

        expanded, seen = [query], {query.strip().lower()}
        for v in variants:
            key = str(v).strip().lower()
            if key and key not in seen:
                seen.add(key)
                expanded.append(str(v).strip())

        expanded = expanded[:n]
        logger.info(f"🔀 Query expanded into {len(expanded)} phrasings")
        return expanded

    async def _complete_json(self, prompt: str) -> str:
        """Single-turn JSON completion that bypasses the tool-calling and final-answer schema paths."""
        if "gemini" in self.model.lower():
            if not self.gemini_client:
                raise RuntimeError("Gemini client unavailable")

            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=self.model.split(":")[-1],
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=_EXPANSION_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""

        if not self.openai_client:
            raise RuntimeError("OpenAI client unavailable")

        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    async def _chat_openai(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict]],
        temperature: float,
        json_mode: bool,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        OpenAI implementation with separated tool/schema phases
        """
        if not self.openai_client:
            return await self._try_fallback(
                messages, system_prompt, tools, temperature, json_mode, model_name,
                RuntimeError("no OPENAI_API_KEY configured"),
            )

        # Format messages
        formatted_msgs = []
        if system_prompt:
            formatted_msgs.append({"role": "system", "content": system_prompt})

        for m in messages:
            msg_to_add = {"role": m["role"], "content": m.get("content")}

            if m.get("tool_calls"):
                msg_to_add["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else tc["arguments"],
                        },
                    }
                    for tc in m["tool_calls"]
                ]

            if m.get("tool_call_id"):
                msg_to_add["tool_call_id"] = m.get("tool_call_id") or m.get("id")

            formatted_msgs.append(msg_to_add)

        # Tools and the response schema travel together, so the model either calls a tool or emits schema-shaped JSON in one round-trip.
        kwargs = {
            "model": model_name,
            "messages": formatted_msgs,
            "temperature": temperature,
            "response_format": RESPONSE_FORMAT,
            "max_completion_tokens": settings.LLM_MAX_TOKENS,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await _with_retry(
                lambda: self._openai_create(dict(kwargs)), "OpenAI"
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = tc.function.arguments

                    tool_calls.append(
                        {"id": tc.id, "name": tc.function.name, "arguments": args}
                    )

                logger.debug(f"🔧 Extracted {len(tool_calls)} tool calls")
                return {"content": msg.content or "", "tool_calls": tool_calls}

            logger.info(f"✅ Structured answer ({len(msg.content or '')} chars)")
            return {"content": msg.content or "", "tool_calls": []}

        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}", exc_info=True)
            return await self._try_fallback(
                messages, system_prompt, tools, temperature, json_mode, model_name, e
            )

    async def _chat_gemini(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict]],
        temperature: float,
        json_mode: bool,
        model_name: str,
    ) -> Dict[str, Any]:
        """Gemini implementation with strict schema enforcement"""
        if not self.gemini_client:
            logger.warning("⚠️ Gemini client not available, trying fallback")
            return await self._try_fallback(
                messages, system_prompt, tools, temperature, json_mode, model_name,
                RuntimeError("no GEMINI_API_KEY configured"),
            )

        # A thought_signature belongs to the model that issued it. Gemini rejects a
        # replayed function call whose signature is missing, and equally rejects one
        # carrying another model's — so a conversation cannot be handed over natively.
        # Mid-conversation failover used to die here on a 400 with no fallback left:
        # "Function call is missing a thought_signature in functionCall parts".
        # When the calls are not ours to replay, the exchange travels as text instead.
        # The new model loses the ability to continue the call, and keeps the facts.
        # The test is who issued the call, not whether a signature came with it. Models
        # differ: gemini-2.5-flash returns no thought_signature and needs none, while
        # gemini-flash-latest rejects a replayed call that lacks one. Requiring a signature
        # here degraded every multi-turn request on the model that never sends them.
        replayable = all(
            tc.get("signature_model") == model_name
            for m in messages
            if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])
        )
        if not replayable:
            logger.info("↻ Replaying tool history as text: the calls are another model's")

        # Map history to Gemini format
        gemini_msgs = []
        for m in messages:
            role = "user" if m["role"] in ["user", "system"] else "model"
            parts = []

            if m.get("content"):
                parts.append(types.Part(text=str(m["content"])))

            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if replayable:
                        part = types.Part(
                            function_call=types.FunctionCall(
                                name=tc["name"], args=tc["arguments"]
                            )
                        )
                        if tc.get("thought_signature"):
                            part.thought_signature = tc["thought_signature"]
                        parts.append(part)
                    else:
                        parts.append(
                            types.Part(text=f"[called {tc['name']}({tc['arguments']})]")
                        )

            if m["role"] == "tool":
                name = m.get("name", "unknown")
                if replayable:
                    # Gemini carries function responses on the user turn; a literal
                    # "function" role is rejected as an invalid role
                    part = types.Part(
                        function_response=types.FunctionResponse(
                            name=name, response={"result": m["content"]}
                        )
                    )
                else:
                    # A function_response with no function_call before it is invalid too
                    part = types.Part(text=f"[{name} returned: {m['content']}]")
                gemini_msgs.append(types.Content(role="user", parts=[part]))
                continue

            if parts:
                gemini_msgs.append(types.Content(role=role, parts=parts))

        # Configure tools
        tool_config = None
        if tools:
            funcs = [
                types.FunctionDeclaration(
                    name=t["function"]["name"],
                    description=t["function"]["description"],
                    parameters=t["function"]["parameters"],
                )
                for t in tools
            ]
            tool_config = [types.Tool(function_declarations=funcs)]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=settings.LLM_MAX_TOKENS,
            tools=tool_config,
            # Business documents trip low-confidence filters; block only high-confidence harm
            safety_settings=[
                types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
                for c in (
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                )
            ],
        )

        # JSON mode only: the Developer API rejects response_schema containing additionalProperties, which Pydantic emits for this schema's Dict[str, Any] fields. The shape is specified in the system prompt and parsed tolerantly.
        if not tools:
            logger.info("🔒 JSON mode (Gemini final answer)")
            config.response_mime_type = "application/json"

        try:
            clean_model = model_name.split(":")[-1] if ":" in model_name else model_name
            logger.debug(
                f"📤 Calling Gemini {clean_model} with {len(gemini_msgs)} messages"
            )

            response = await _with_retry(
                lambda: asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=clean_model,
                    contents=gemini_msgs,
                    config=config,
                ),
                "Gemini",
            )

            if not response.candidates:
                logger.warning("⚠️ No candidates in Gemini response")
                return await self._try_fallback(
                    messages, system_prompt, tools, temperature, json_mode, model_name,
                    RuntimeError("provider returned no candidates"),
                )

            candidate = response.candidates[0]

            # Safety check for blocked/empty content
            if not candidate.content or not candidate.content.parts:
                logger.warning("⚠️ Empty or blocked Gemini response")
                return await self._try_fallback(
                    messages, system_prompt, tools, temperature, json_mode, model_name,
                    RuntimeError("response empty or blocked by safety filters"),
                )

            # Extract text
            content_text = ""
            for part in candidate.content.parts:
                if part.text:
                    content_text += part.text

            # Extract tool calls
            tool_calls = []
            for part in candidate.content.parts:
                if part.function_call:
                    tool_calls.append(
                        {
                            "id": f"call_{part.function_call.name}_{uuid.uuid4().hex[:4]}",
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                            "thought_signature": getattr(
                                part, "thought_signature", None
                            ),
                            "signature_model": model_name,
                        }
                    )

            if tool_calls:
                logger.debug(f"🔧 Extracted {len(tool_calls)} Gemini tool calls")
                return {"content": content_text, "tool_calls": tool_calls}

            # Gemini rejects response_schema alongside tools, so the answer is shaped in a second, tool-free pass that can carry the schema
            if tools and content_text:
                logger.info("🔒 Reformatting Gemini answer under schema")
                return await self._chat_gemini(
                    messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Return this analysis as the required JSON, written in the "
                                "language of the user's question quoted in your "
                                "instructions. That governs `answer`, every "
                                "`key_insights` entry, and every chart or diagram title. "
                                "If the analysis below is in a different language from the "
                                "question, translate it into the question's language now.\n\n"
                                f"{content_text}"
                            ),
                        }
                    ],
                    system_prompt,
                    None,
                    temperature,
                    json_mode,
                    model_name,
                )

            if content_text:
                logger.info("✅ Received Gemini structured output")
                try:
                    json.loads(content_text)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Gemini output is malformed JSON: {e}")

            return {"content": content_text, "tool_calls": tool_calls}

        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}", exc_info=True)
            return await self._try_fallback(
                messages, system_prompt, tools, temperature, json_mode, model_name, e
            )

    async def _try_fallback(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict]],
        temperature: float,
        json_mode: bool,
        failed_model: str = None,
        cause: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """
        Fallback mechanism; one hop only, so a dead fallback cannot loop.

        The reason travels with the answer. Falling back silently once meant a reply
        produced by Gemini was presented as coming from gpt-4o, hiding both the switch
        and the exhausted key behind it.
        """
        if (
            not self.fallback_model
            or self.fallback_model == failed_model
            or self.fallback_model.lower() == "none"
        ):
            logger.error("❌ All LLM options exhausted.")
            detail = _provider_reason(cause)
            return {
                "content": json.dumps(
                    {
                        "answer": (
                            f"⚠️ No model could answer this. {failed_model or 'The model'} "
                            f"failed ({detail}) and no usable fallback is configured. "
                            "The request was not processed."
                        ),
                        "key_insights": [f"{failed_model}: {detail}"],
                        "visualizations": [],
                        "sources_used": [],
                    }
                ),
                "tool_calls": [],
                "degraded": f"{failed_model} failed ({detail}); no fallback available",
            }

        logger.info(f"⚠️ Falling back to: {self.fallback_model}")
        result = await self.chat(
            messages,
            system_prompt,
            tools,
            temperature,
            json_mode,
            model_override=self.fallback_model,
        )
        result["degraded"] = (
            f"{failed_model} failed ({_provider_reason(cause)}); "
            f"answered with {self.fallback_model} instead"
        )
        return result
