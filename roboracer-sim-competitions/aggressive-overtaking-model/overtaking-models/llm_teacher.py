"""Optional training-only LLM review; never imported by the racing controller.

An LLM is a source of suggestions, not ground truth. Its proposed maneuver is
validated against geometric feasibility and must still be tested by rollouts.
No missing-key fallback is silently presented as an LLM result. API responses,
model IDs, usage and rejection reasons are cached with their scenario hash.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


PROMPT_VERSION = "overtaking-review-v2-early-pass"
ACTIONS = ("LEFT", "RIGHT", "WAIT", "PROBE")
SYSTEM_PROMPT = """You are an offline teacher for a simulated 1/10-scale racing car.
Choose between supplied feasible candidates using their estimated completion
time, collision clearance, uncertainty, and ability to return to the reference
racing line. LEFT means pass on the obstacle's left (obstacle as right wall).
RIGHT means pass on its right (obstacle as left wall). WAIT means follow or stop
if the passage is blocked. PROBE means approach an already feasible inside
space while preserving a stop/retreat margin; never assume an opponent yields,
never aim at an occupied or nonexistent gap, and never permit contact. Use the
provided predicted motion and uncertainty for moving obstacles. Never invent
free space. Do not select a candidate whose feasible field is false. If no
passing candidate is feasible, choose WAIT. Steering is in radians, positive
left; target_speed is m/s, not throttle. Candidate controls are bounded hints,
and the simulator must validate your recommendation before it becomes a label.
Prefer the earliest safe completed pass over waiting for most of a lap. Start a feasible lateral move early enough to avoid braking beside the opponent. Do not treat a larger safety margin as inherently better when the smaller margin still clears the physical bodies and respects prediction uncertainty. The opponent is not required to yield. Return a short rationale, not detailed internal reasoning.\n"""


class TeacherError(RuntimeError):
    pass


def load_local_env(path=None):
    """Load only two training settings, without shell evaluation or printing.

    Existing process environment values win. Quotes are stripped but never
    interpreted. Unknown keys, comments, blank and malformed lines are ignored.
    """
    path = Path(path) if path is not None else Path(__file__).with_name(".env")
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?(OPENAI_API_KEY|OPENAI_MODEL)\s*=\s*(.*?)\s*", line)
        if not match:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not value or value.lower() in {"your_api_key_here", "paste_your_key_here", "replace_me", "replace_with_your_key"}:
            continue
        if not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _canonical(scenario):
    try:
        return json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scenario must be finite JSON data.") from exc


def _candidates(scenario):
    candidates = scenario.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Scenario needs candidate records.")
    result = {}
    for c in candidates:
        if c.get("action") not in ACTIONS or c["action"] in result or not isinstance(c.get("feasible"), bool):
            raise ValueError("Candidates need distinct known actions and boolean feasibility.")
        result[c["action"]] = c
    if "WAIT" not in result:
        raise ValueError("A WAIT candidate must always be supplied.")
    return result


def geometric_teacher(scenario):
    """Cheap, explicitly non-LLM bootstrap label from caller-provided candidates.

    Feasibility/time values are computed by the planner. This does not certify
    dynamic success; callers should record independent rollout validation.
    """
    _canonical(scenario)
    candidates = _candidates(scenario)
    feasible = [c for c in candidates.values() if c["feasible"]]
    if not feasible:
        chosen = candidates["WAIT"]
    else:
        chosen = min(feasible, key=lambda c: float(c.get("estimated_time", 1e6)))
    return {"action": chosen["action"], "confidence": 1.,
            "reason": "Lowest estimated time among geometrically feasible candidates.",
            "steering": float(chosen.get("steering", 0.)),
            "target_speed": float(chosen.get("target_speed", 0.)),
            "source": "geometric_bootstrap", "llm_used": False,
            "rollout_verified": False}


class LLMTeacher:
    """Bounded, cached requests to the official Responses endpoint.

    The API key is only read from OPENAI_API_KEY (or an explicit constructor
    value) and is never persisted or included in exception text. No requests
    are made by constructing the teacher or importing this module.
    """

    def __init__(self, model, cache_path, *, max_calls, api_key=None, timeout=30., max_output_tokens=500, max_cost_usd=None):
        if not model or max_calls < 0 or timeout <= 0 or max_output_tokens < 100:
            raise ValueError("Supply a model, nonnegative call limit and valid timeout/token limit.")
        self.model, self.cache_path, self.max_calls = str(model), Path(cache_path), int(max_calls)
        if api_key is None:
            load_local_env()
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout, self.max_output_tokens = float(timeout), int(max_output_tokens)
        self.calls = 0
        self.max_cost_usd = max_cost_usd
        self.reserved_cost_usd = 0.
        self.usage_cost_usd = 0.
        if max_cost_usd is not None:
            if self.model not in ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"):
                raise ValueError("Cost guard needs verified pricing for this model.")
            if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
                raise ValueError("Cost ceiling must be positive and finite.")
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.reserved_cost_usd += float(record.get("reserved_cost_usd", 0.))
                    self.usage_cost_usd += float(record.get("estimated_usage_cost_usd", 0.))
                    if record.get("status") == "accepted":
                        self.cache[record["cache_key"]] = record

    def _record(self, record):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")

    def label(self, scenario):
        canonical = _canonical(scenario)
        candidates = _candidates(scenario)
        cache_key = hashlib.sha256((PROMPT_VERSION + self.model + canonical).encode()).hexdigest()
        if cache_key in self.cache:
            result = dict(self.cache[cache_key]["label"])
            result["cache_hit"] = True
            return result
        if not self.api_key:
            raise TeacherError("OPENAI_API_KEY is not configured. No LLM label was generated.")
        if self.calls >= self.max_calls:
            raise TeacherError("LLM call limit reached; no additional request was sent.")
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"action": {"type": "string", "enum": list(candidates)},
                                 "confidence": {"type": "number"},
                                 "reason": {"type": "string"},
                                 "steering": {"type": "number"},
                                 "target_speed": {"type": "number"}},
                  "required": ["action", "confidence", "reason", "steering", "target_speed"]}
        payload = {"model": self.model, "instructions": SYSTEM_PROMPT, "input": canonical,
                   "store": False, "max_output_tokens": self.max_output_tokens,
                   "text": {"format": {"type": "json_schema", "name": "overtaking_teacher",
                                        "strict": True, "schema": schema}}}
        # UTF-8 bytes upper-bound tokenized text; extra allowance covers schema
        # and protocol framing. Reserve full output, including failed requests.
        # This is a conservative application guard, not an account billing cap.
        reserved = ((len(json.dumps(payload).encode()) + 4096) * .40
                    + self.max_output_tokens * 1.60) / 1_000_000
        if self.max_cost_usd is not None and self.reserved_cost_usd + reserved > self.max_cost_usd:
            raise TeacherError("LLM cost limit reached; no additional request was sent.")
        self.reserved_cost_usd += reserved
        request = urllib.request.Request("https://api.openai.com/v1/responses",
                                         data=json.dumps(payload).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + self.api_key,
                                                  "Content-Type": "application/json"})
        self.calls += 1
        record = {"cache_key": cache_key, "prompt_version": PROMPT_VERSION,
                  "requested_model": self.model, "scenario": scenario, "source": "openai_responses",
                  "status": "pending", "wall_timestamp": time.time(), "reserved_cost_usd": reserved}
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
            record.update(response_id=body.get("id"), model=body.get("model"), usage=body.get("usage"),
                          latency_seconds=time.perf_counter() - start)
            usage = body.get("usage") or {}
            cost = (usage.get("input_tokens", 0) * .40 + usage.get("output_tokens", 0) * 1.60) / 1_000_000
            if self.model in ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"):
                record["estimated_usage_cost_usd"] = cost
                self.usage_cost_usd += cost
            if body.get("status") not in (None, "completed"):
                raise TeacherError("LLM response was incomplete; label rejected.")
            content = [c for item in body.get("output", []) for c in item.get("content", [])]
            if any(c.get("type") == "refusal" for c in content):
                raise TeacherError("LLM declined the training query; no label accepted.")
            text = "".join(c.get("text", "") for c in content if c.get("type") == "output_text")
            record["response_text"] = text
            label = json.loads(text)
            required = {"action", "confidence", "reason", "steering", "target_speed"}
            if set(label) != required or label["action"] not in candidates:
                raise TeacherError("LLM output has an invalid action or schema.")
            if not candidates[label["action"]]["feasible"]:
                raise TeacherError("LLM chose an infeasible maneuver; label rejected.")
            numbers = [label["confidence"], label["steering"], label["target_speed"]]
            if not all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in numbers):
                raise TeacherError("LLM controls or confidence are not finite numbers.")
            max_speed = float(scenario.get("max_speed", 6.))
            if not 0 <= label["confidence"] <= 1 or abs(label["steering"]) > .4189 or not 0 <= label["target_speed"] <= max_speed:
                raise TeacherError("LLM controls/confidence exceed the allowed bounds.")
            if not isinstance(label["reason"], str) or len(label["reason"]) > 2000:
                raise TeacherError("LLM rationale has invalid type or length.")
            label.update(source="openai_responses", llm_used=True, rollout_verified=False,
                         response_id=record["response_id"], model=record["model"],
                         cache_key=cache_key, cache_hit=False)
            record.update(status="accepted", label=label)
            self._record(record)
            self.cache[cache_key] = record
            return dict(label)
        except urllib.error.HTTPError as exc:
            # Keep only machine-readable service codes. Never echo the raw
            # error body, which might include request or authentication data.
            service_error = {}
            try:
                parsed = json.loads(exc.read().decode()).get("error", {})
                for name in ("code", "type"):
                    value = parsed.get(name)
                    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_\-]{1,100}", value):
                        service_error[name] = value
            except (ValueError, OSError, AttributeError):
                pass
            detail = service_error.get("code") or service_error.get("type")
            record.update(status="error", error=f"HTTP {exc.code}", service_error=service_error,
                          latency_seconds=time.perf_counter() - start)
            self._record(record)
            suffix = f" ({detail})" if detail else ""
            raise TeacherError(f"LLM API returned HTTP {exc.code}{suffix}; no label accepted.") from None
        except (OSError, TimeoutError, ValueError, KeyError, TypeError, TeacherError) as exc:
            message = str(exc) if isinstance(exc, TeacherError) else "Malformed response or network failure."
            record.update(status="rejected", error=message, latency_seconds=time.perf_counter() - start)
            self._record(record)
            raise TeacherError(message) from None
