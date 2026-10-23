from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from annaops.config import Settings, settings
from annaops.llm import LLMResponseError, LLMUnavailable, StructuredLLMClient
from annaops.schemas import ExtractedOffer, FoodItem


CRITICAL_FIELDS = ("food_type", "quantity", "unit", "prepared_at", "pickup_deadline", "donor_location")
QUESTION_BY_FIELD = {
    "food_type": "What food is available, and is it vegetarian, non-vegetarian, or mixed?",
    "quantity": "How much food is available?",
    "unit": "What unit describes the quantity (for example, portions or kilograms)?",
    "prepared_at": "When was the food prepared? Please include the date, time, and timezone if relevant.",
    "pickup_deadline": "What is the latest safe pickup time?",
    "donor_location": "What is the pickup location?",
    "storage_conditions": "How has the food been stored since preparation?",
}


class IntakeError(RuntimeError):
    pass


class IntakeAgent:
    """Extracts facts while keeping deterministic validation separate from language interpretation."""

    def __init__(self, config: Settings = settings):
        self.config = config
        self.llm = StructuredLLMClient(config)

    def extract(self, message: str) -> ExtractedOffer:
        result, _ = self.extract_with_trace(message)
        return result

    def extract_with_trace(self, message: str) -> tuple[ExtractedOffer, dict[str, Any]]:
        started = time.perf_counter()
        if self.llm.configured:
            try:
                result, invocation = self._extract_with_llm(message)
                return result, {
                    "mode": "llm",
                    "provider": invocation.provider,
                    "model": invocation.model,
                    "duration_ms": invocation.duration_ms,
                    "retry_count": 0,
                    "error": None,
                    "structured_output": result.model_dump(mode="json"),
                    "tools": [],
                    "selected_action": "request_clarification" if result.missing_fields or result.contradictions else "submit_for_safety",
                }
            except (LLMUnavailable, LLMResponseError, ValueError, ValidationError) as exc:
                result = self._extract_deterministically(message)
                result.extraction_source = "llm_error"
                result.notes = self._append_note(result.notes, f"Configured LLM failed safely: {type(exc).__name__}")
                if "Confirm the extracted details because the configured model was unavailable." not in result.clarification_questions:
                    result.clarification_questions.append(
                        "Confirm the extracted details because the configured model was unavailable."
                    )
                return result, {
                    "mode": "deterministic_fallback",
                    "provider": self.llm.provider,
                    "model": self.config.llm_model,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "retry_count": 0,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "structured_output": result.model_dump(mode="json"),
                    "tools": [],
                    "selected_action": "request_clarification",
                }
        result = self._extract_deterministically(message)
        return result, {
            "mode": "deterministic",
            "provider": None,
            "model": None,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "retry_count": 0,
            "error": None,
            "structured_output": result.model_dump(mode="json"),
            "tools": [],
            "selected_action": "request_clarification" if result.missing_fields or result.contradictions else "submit_for_safety",
        }

    def merge_clarification(
        self,
        current: ExtractedOffer,
        fields: dict[str, Any],
        reference_at: datetime | None = None,
    ) -> ExtractedOffer:
        permitted = set(ExtractedOffer.model_fields) - {
            "explicit_fields", "inferred_fields", "missing_fields", "contradictions",
            "clarification_questions", "extraction_source"
        }
        unknown = set(fields) - permitted
        if unknown:
            raise ValueError(f"Unknown clarification fields: {sorted(unknown)}")
        if not fields:
            raise ValueError("At least one clarification field is required")
        payload = current.model_dump()
        for key, value in fields.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Clarification value for '{key}' cannot be empty")
            inferred: list[str] = []
            if key in {"prepared_at", "pickup_deadline"}:
                value, inferred = self._normalize_datetime_answer(key, value, reference_at)
            elif isinstance(value, str):
                value = value.strip()
            payload[key] = value
            if key not in payload["explicit_fields"]:
                payload["explicit_fields"].append(key)
            for inference in inferred:
                if inference not in payload["inferred_fields"]:
                    payload["inferred_fields"].append(inference)
        payload["contradictions"] = []
        result = ExtractedOffer.model_validate(payload)
        return self._finalize(result)

    def upgrade_legacy_food_items(self, current: ExtractedOffer, original_message: str) -> ExtractedOffer:
        """Repair pre-itemization records from their immutable source message.

        Only food fields are changed, and only when the deterministic parser finds explicit
        quantity/unit/name groups. Existing clarification and safety facts are preserved.
        """
        if current.food_items:
            return current
        items = self._extract_food_items(original_message)
        if not items:
            return current
        payload = current.model_dump()
        payload["food_items"] = [item.model_dump() for item in items]
        payload["food_type"] = ", ".join(item.name for item in items)
        units = {item.unit for item in items}
        if len(items) == 1:
            payload["quantity"], payload["unit"] = items[0].quantity, items[0].unit
        elif len(units) == 1:
            payload["quantity"], payload["unit"] = sum(item.quantity for item in items), next(iter(units))
        else:
            payload["quantity"], payload["unit"] = None, None
        for field in ("food_items", "food_type", "quantity", "unit"):
            if field not in payload["explicit_fields"]:
                payload["explicit_fields"].append(field)
        return self._finalize(ExtractedOffer.model_validate(payload))

    def _extract_with_llm(self, message: str):
        prompt = (
            "Extract only explicitly stated surplus-food donation facts. Never invent values. "
            "Preserve each quantity, unit, and food as its own food_items entry. Use null for missing facts, "
            "record supplied field names in explicit_fields, distinguish any inference in inferred_fields, "
            "flag ambiguity in contradictions, ask only concise questions needed to resolve missing or ambiguous "
            "facts, and use ISO-8601 timestamps. Do not make a food-safety decision."
        )
        invocation = self.llm.complete_structured(
            ExtractedOffer,
            system_prompt=prompt,
            user_payload={"original_donor_message": message},
        )
        result = ExtractedOffer.model_validate(invocation.output)
        result.extraction_source = "configured_llm"
        return self._finalize(result), invocation

    def _extract_deterministically(self, message: str) -> ExtractedOffer:
        lower = message.lower()
        payload: dict[str, Any] = {"extraction_source": "deterministic_demo", "explicit_fields": []}

        food_items = self._extract_food_items(message)
        if food_items:
            payload["food_items"] = [item.model_dump() for item in food_items]
            payload["food_type"] = ", ".join(item.name for item in food_items)
            payload["explicit_fields"].extend(["food_items", "food_type", "quantity", "unit"])
            units = {item.unit for item in food_items}
            if len(food_items) == 1:
                payload["quantity"] = food_items[0].quantity
                payload["unit"] = food_items[0].unit
            elif len(units) == 1:
                payload["quantity"] = sum(item.quantity for item in food_items)
                payload["unit"] = next(iter(units))

        food_markers = [
            "biryani", "rice", "dal", "curry", "bread", "sandwich", "sandwiches", "meal", "meals",
            "fruit", "fruits", "vegetables", "dessert", "snacks", "soup", "pasta", "chapati", "rotis", "roti"
        ]
        found_foods = [item for item in food_markers if re.search(rf"\b{re.escape(item)}\b", lower)]
        if found_foods and not food_items:
            prefix = "vegetarian " if re.search(r"\b(vegetarian|veg)\b", lower) else ""
            payload["food_type"] = prefix + ", ".join(dict.fromkeys(found_foods))
            payload["explicit_fields"].append("food_type")

        has_non_veg = bool(re.search(r"\b(non[- ]?veg(?:etarian)?|chicken|meat|fish|egg)\b", lower))
        has_veg = bool(re.search(r"\b(vegetarian|veg)\b", lower)) and not bool(re.search(r"\bnon[- ]?veg", lower))
        if has_non_veg and has_veg:
            payload["food_category"] = "mixed"
        elif has_non_veg:
            payload["food_category"] = "non_vegetarian"
        elif has_veg:
            payload["food_category"] = "vegetarian"
        else:
            payload["food_category"] = "unknown"
        if has_non_veg or has_veg:
            payload["explicit_fields"].append("food_category")

        prepared = self._find_timestamp(message, ("prepared at", "cooked at", "prepared"))
        deadline = self._find_timestamp(message, ("pickup by", "collect by", "collection by", "deadline"))
        if prepared:
            payload["prepared_at"] = prepared
            payload["explicit_fields"].append("prepared_at")
        if deadline:
            payload["pickup_deadline"] = deadline
            payload["explicit_fields"].append("pickup_deadline")

        storage_patterns = [
            r"\b(kept hot(?: at [^,.;]+)?)\b", r"\b(refrigerated(?: at [^,.;]+)?)\b",
            r"\b(chilled(?: at [^,.;]+)?)\b", r"\b(frozen(?: at [^,.;]+)?)\b",
            r"\b(room temperature(?: since [^,.;]+)?)\b",
        ]
        for pattern in storage_patterns:
            match = re.search(pattern, lower)
            if match:
                payload["storage_conditions"] = match.group(1).strip()
                payload["explicit_fields"].append("storage_conditions")
                break

        location_match = re.search(
            r"\b(?:from|pickup location[: ]+)\s+(.+?)(?=,\s*coordinates?\b|[.;]|$)",
            message,
            re.IGNORECASE,
        )
        if location_match:
            candidate = location_match.group(1).strip()
            if not re.fullmatch(r"\d{1,2}:\d{2}(?:\s*[ap]m)?", candidate, re.IGNORECASE):
                payload["donor_location"] = candidate
                payload["explicit_fields"].append("donor_location")

        lat_lon = re.search(r"\b(?:coordinates?[: ]*)?(-?\d{1,2}\.\d+)\s*[,/]\s*(-?\d{1,3}\.\d+)\b", message)
        if lat_lon:
            payload["donor_latitude"] = float(lat_lon.group(1))
            payload["donor_longitude"] = float(lat_lon.group(2))
            payload["explicit_fields"].extend(["donor_latitude", "donor_longitude"])

        result = ExtractedOffer.model_validate(payload)
        return self._finalize(result)

    def _extract_food_items(self, message: str) -> list[FoodItem]:
        units = r"portions?|meals?|servings?|plates?|packs?|kgs?|kilograms?|litres?|liters?"
        stop = rf"(?=\s+(?:and|plus)\s+\d+(?:\.\d+)?\s*(?:{units})\b|,?\s*(?:prepared at|cooked at|pickup by|collect by|kept hot|refrigerated|chilled|frozen|room temperature|from|coordinates?)\b|[.;]|$)"
        pattern = re.compile(
            rf"\b(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>{units})\b\s*(?:of\s+)?(?P<food>.+?){stop}",
            re.IGNORECASE,
        )
        items: list[FoodItem] = []
        for match in pattern.finditer(message):
            food = re.sub(r"^(?:of\s+)", "", match.group("food").strip(" ,"), flags=re.IGNORECASE)
            if not food:
                continue
            unit = self._normalize_unit(match.group("unit"))
            lowered_food = food.lower()
            category = "non_vegetarian" if re.search(r"\b(chicken|meat|fish|egg|non[- ]?veg(?:etarian)?)\b", lowered_food) else (
                "vegetarian" if re.search(r"\b(vegetarian|veg)\b", lowered_food) else "unknown"
            )
            items.append(FoodItem(
                name=lowered_food,
                quantity=float(match.group("quantity")),
                unit=unit,
                category=category,
            ))
        return items

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        value = unit.lower()
        aliases = {
            "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
            "liter": "litres", "liters": "litres", "litre": "litres",
            "portion": "portions", "meal": "meals", "serving": "servings",
            "plate": "plates", "pack": "packs",
        }
        return aliases.get(value, value)

    def _normalize_datetime_answer(
        self,
        field: str,
        value: Any,
        reference_at: datetime | None,
    ) -> tuple[datetime, list[str]]:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            answer = value.strip()
            informal = re.fullmatch(r"(\d{1,2})(?::([0-5]\d))?\s*([ap])\.?m\.?", answer, re.IGNORECASE)
            if informal:
                if reference_at is None:
                    raise ValueError(f"{field} needs an explicit date because no submission date is available")
                hour = int(informal.group(1))
                if not 1 <= hour <= 12:
                    raise ValueError(f"Invalid 12-hour time for {field}: {answer}")
                minute = int(informal.group(2) or 0)
                meridiem = informal.group(3).lower()
                hour = (hour % 12) + (12 if meridiem == "p" else 0)
                local_reference = reference_at.astimezone(self.config.timezone) if reference_at.tzinfo else reference_at.replace(tzinfo=self.config.timezone)
                parsed = datetime.combine(local_reference.date(), datetime.min.time(), self.config.timezone).replace(hour=hour, minute=minute)
                return parsed, [f"{field}.date_from_submission", f"{field}.timezone_from_configuration"]
            if re.fullmatch(r"\d{1,2}(?::\d{2})?", answer):
                raise ValueError(f"Ambiguous time for {field}: include am/pm or a full ISO-8601 timestamp")
            try:
                parsed = datetime.fromisoformat(answer.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {field}: use a time such as '6 pm' or a full ISO-8601 timestamp with date"
                ) from exc
        else:
            raise ValueError(f"Invalid value type for {field}")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.config.timezone)
            return parsed, [f"{field}.timezone_from_configuration"]
        return parsed, []

    def _find_timestamp(self, message: str, labels: tuple[str, ...]) -> datetime | None:
        for label in labels:
            pattern = rf"{re.escape(label)}\s+(\d{{4}}-\d{{2}}-\d{{2}}[T ]\d{{2}}:\d{{2}}(?::\d{{2}})?(?:Z|[+-]\d{{2}}:\d{{2}})?)"
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                value = match.group(1).replace(" ", "T")
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=self.config.timezone)
        return None

    def _finalize(self, result: ExtractedOffer) -> ExtractedOffer:
        proposed_questions = [question.strip() for question in result.clarification_questions if question.strip()]
        missing = []
        for field in CRITICAL_FIELDS:
            if field in {"food_type", "quantity", "unit"} and result.food_items:
                continue
            if getattr(result, field) in (None, ""):
                missing.append(field)
        if result.storage_conditions in (None, ""):
            missing.append("storage_conditions")
        result.missing_fields = missing
        maximum_questions = len(missing) + (1 if result.contradictions else 0)
        if result.extraction_source == "configured_llm" and proposed_questions and len(proposed_questions) <= maximum_questions:
            result.clarification_questions = proposed_questions
        else:
            result.clarification_questions = [QUESTION_BY_FIELD[field] for field in missing]
        if result.contradictions:
            conflict_question = "Please correct the conflicting or ambiguous details before this offer continues."
            if not result.clarification_questions or conflict_question not in result.clarification_questions:
                result.clarification_questions.append(conflict_question)
        return result

    @staticmethod
    def _append_note(existing: str | None, addition: str) -> str:
        return f"{existing}; {addition}" if existing else addition

