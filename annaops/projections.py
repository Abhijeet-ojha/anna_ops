from __future__ import annotations

from datetime import datetime
from typing import Any


def project_workflow(donation: dict[str, Any]) -> dict[str, Any]:
    events = donation.get("audit_events", [])
    current_status = donation["status"]
    parsed = donation.get("parsed") or {}
    safety = donation.get("safety") or {}
    match = donation.get("match") or {}
    approvals = donation.get("approval_decisions") or []
    attempts = donation.get("dispatch_attempts") or []

    def latest(*types: str) -> dict[str, Any] | None:
        for event in reversed(events):
            if event["event_type"] in types:
                return event
        return None

    stages: list[dict[str, Any]] = []

    def add(
        key: str, label: str, component: str, status: str, explanation: str,
        event: dict[str, Any] | None = None, input_data: Any = None,
        output_data: Any = None, fallback: str | None = None,
    ) -> None:
        timestamp = (event.get("created_at") or event.get("timestamp")) if event else None
        previous_timestamp = stages[-1]["timestamp"] if stages else None
        duration_ms = None
        if timestamp and previous_timestamp:
            try:
                t1 = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(previous_timestamp.replace("Z", "+00:00"))
                duration_ms = max(0, round((t1 - t2).total_seconds() * 1000))
            except Exception:
                duration_ms = None
        stages.append({
            "key": key, "label": label, "component": component, "status": status,
            "explanation": explanation, "timestamp": timestamp, "duration_ms": duration_ms,
            "input": input_data, "output": output_data, "fallback": fallback,
        })

    created_event = latest("DONATION_CREATED")
    add(
        "donor", "Donor message", "Human input", "completed",
        "Original donor message persisted as the immutable source record.", created_event,
        input_data=donation["original_message"], output_data={"donation_id": donation["id"]},
    )

    intake_event = latest("INTAKE_PARSED", "INTAKE_CLARIFICATION_REQUIRED", "INTAKE_FAILED")
    intake_status = "completed" if parsed else ("failed" if current_status == "ERROR" else "pending")
    source_name = "Configured LLM" if parsed.get("extraction_source") == "configured_llm" else parsed.get("extraction_source", "unknown")
    intake_explanation = (
        f"Extracted {len(parsed.get('food_items') or []) or 1} structured food line(s) using {source_name}."
        if parsed else ("Extraction failed; preserved safely for retry." if current_status == "ERROR" else "Waiting for structured extraction.")
    )
    fallback = None
    if parsed.get("extraction_source") == "deterministic_demo":
        fallback = "Configured LLM unavailable; labelled demo parser used."
    elif current_status == "ERROR" and not parsed:
        fallback = f"LLM extraction failed: {donation.get('error_message') or 'Check agent executions'}"
    add(
        "intake", "Intake Agent", "Language extraction", intake_status, intake_explanation,
        intake_event, input_data=donation["original_message"],
        output_data={
            "food_type": parsed.get("food_type"), "food_items": parsed.get("food_items", []),
            "extraction_source": parsed.get("extraction_source"),
        } if parsed else None,
        fallback=fallback,
    )

    missing = parsed.get("missing_fields") or []
    contradictions = parsed.get("contradictions") or []
    clarification_event = latest("CLARIFICATION_SUBMITTED")
    if contradictions:
        validation_status, validation_explanation = "blocked", f"Contradictory facts detected: {'; '.join(contradictions)}"
    elif missing:
        validation_status, validation_explanation = "awaiting_human", f"Clarification required: {', '.join(missing)}."
    elif parsed:
        validation_status, validation_explanation = "completed", "Required operational facts are present and internally consistent."
    else:
        validation_status, validation_explanation = "pending", "Validation waits for intake output."
    add(
        "validation", "Validation", "Schema and completeness gate", validation_status,
        validation_explanation, clarification_event or intake_event,
        input_data={"explicit": parsed.get("explicit_fields", []), "inferred": parsed.get("inferred_fields", [])},
        output_data={"missing": missing, "contradictions": contradictions},
    )

    safety_event = latest("SAFETY_ASSESSED", "SAFETY_INVALIDATED")
    if safety:
        verdict = safety.get("verdict", "UNKNOWN")
        safety_status = "completed" if verdict == "SAFE_TO_PROCEED_TO_MATCHING" else "blocked"
        if verdict == "REQUIRES_HUMAN_REVIEW":
            safety_status = "awaiting_human"
        safety_explanation = "; ".join(safety.get("reasons") or [verdict])
    elif missing:
        safety_status, safety_explanation = "pending", "Safety has not run because clarification is still required."
    else:
        safety_status, safety_explanation = "pending", "Awaiting deterministic safety assessment."
    add(
        "safety", "Safety Agent", "Deterministic rule engine", safety_status,
        safety_explanation, safety_event, input_data=parsed or None,
        output_data={"verdict": safety.get("verdict"), "rule_version": safety.get("rule_version")} if safety else None,
    )

    match_event = latest("MATCHING_COMPLETED", "NGO_DECLINED_REPLANNED", "NGO_TIMEOUT_REPLANNED", "MATCHING_FAILED")
    candidates = match.get("candidates") or []
    exclusions = match.get("exclusions") or {}
    if match:
        match_status = "completed" if candidates else "blocked"
        match_explanation = f"Ranked {len(candidates)} eligible receiver(s); documented {len(exclusions)} exclusion(s)."
    elif current_status == "ERROR" and parsed and not match:
        match_status, match_explanation = "failed", f"Receiver recommendation failed: {donation.get('error_message') or 'Check agent executions'}"
    elif safety and safety.get("verdict") != "SAFE_TO_PROCEED_TO_MATCHING":
        match_status, match_explanation = "blocked", "Matching did not run because the safety gate did not pass."
    else:
        match_status, match_explanation = "pending", "Matching waits for a passing safety verdict."
    route_methods = sorted({candidate.get("route", {}).get("method") for candidate in candidates if candidate.get("route")})
    route_fallback = None
    if "straight_line_fallback" in route_methods:
        route_fallback = "Unverified straight-line distance is shown; no road ETA is claimed."
    elif "unavailable" in route_methods:
        route_fallback = "Route data unavailable; feasibility requires human confirmation."
    add(
        "matching", "Match & Route Agent", "Eligibility and explainable ranking", match_status,
        match_explanation, match_event, input_data={"quantity": parsed.get("quantity"), "category": parsed.get("food_category")},
        output_data={"candidate_count": len(candidates), "exclusion_count": len(exclusions), "selected_ngo_id": donation.get("selected_ngo_id")},
        fallback=route_fallback,
    )

    approval_event = latest("DONOR_APPROVED_AND_DISPATCHED", "DONOR_REJECTED_PROPOSAL")
    if current_status == "AWAITING_DONOR_APPROVAL":
        approval_status, approval_explanation = "awaiting_human", "A ranked proposal is waiting for explicit donor approval."
    elif approvals:
        decision = approvals[-1]["decision"]
        approval_status = "completed" if decision == "APPROVED" else "blocked"
        approval_explanation = f"Latest donor decision: {decision.lower()}."
    elif not match:
        approval_status, approval_explanation = "pending", "Approval waits for a receiver proposal."
    else:
        approval_status, approval_explanation = "pending", "No donor decision has been recorded."
    add(
        "approval", "Donor approval", "Human control gate", approval_status,
        approval_explanation, approval_event,
        input_data={"selected_ngo_id": donation.get("selected_ngo_id")},
        output_data=approvals[-1] if approvals else None,
    )

    dispatch_event = latest("DONOR_APPROVED_AND_DISPATCHED")
    if attempts:
        dispatch_status, dispatch_explanation = "completed", f"Sent {len(attempts)} simulated receiver request(s); no real message was transmitted."
    elif current_status in {"BLOCKED", "REQUIRES_HUMAN_REVIEW", "NEEDS_CLARIFICATION", "DONOR_REJECTED"}:
        dispatch_status, dispatch_explanation = "blocked", "Dispatch is prohibited by the current workflow state."
    else:
        dispatch_status, dispatch_explanation = "pending", "Dispatch requires an approved proposal."
    add(
        "dispatch", "Dispatch Agent", "Simulated receiver inbox", dispatch_status,
        dispatch_explanation, dispatch_event,
        input_data={"donor_approved": donation.get("donor_approved")},
        output_data={"attempt_count": donation.get("attempt_count"), "simulated": True},
    )

    response_event = latest(
        "NGO_ACCEPTED", "NGO_DECLINED_REPLANNED", "NGO_TIMEOUT_REPLANNED",
        "NGO_DECLINED_ESCALATED", "NGO_TIMEOUT_ESCALATED",
        "NGO_DECLINED_NO_ALTERNATIVE", "NGO_TIMEOUT_NO_ALTERNATIVE",
    )
    if attempts and attempts[-1]["status"] == "AWAITING_RESPONSE":
        response_status, response_explanation = "awaiting_human", "Waiting for a simulated NGO response."
    elif response_event:
        response_status, response_explanation = "completed", response_event.get("reason") or "Simulated NGO response recorded."
    else:
        response_status, response_explanation = "pending", "No simulated receiver response is due yet."
    add(
        "response", "NGO response", "Explicitly simulated external response", response_status,
        response_explanation, response_event,
        input_data=attempts[-1] if attempts else None,
        output_data={"latest_status": attempts[-1]["status"]} if attempts else None,
        fallback="Simulation only; no real NGO was contacted." if attempts else None,
    )

    replan_event = latest("NGO_DECLINED_REPLANNED", "NGO_TIMEOUT_REPLANNED")
    escalation_event = latest(
        "NGO_DECLINED_ESCALATED", "NGO_TIMEOUT_ESCALATED",
        "NGO_DECLINED_NO_ALTERNATIVE", "NGO_TIMEOUT_NO_ALTERNATIVE", "HUMAN_ESCALATION",
    )
    if replan_event:
        replan_status, replan_explanation = "completed", replan_event.get("reason") or "Replacement receiver proposal generated."
    elif escalation_event:
        replan_status, replan_explanation = "blocked", escalation_event.get("reason") or "Retry path escalated."
    else:
        replan_status, replan_explanation = "pending", "Replanning activates only after a decline or timeout."
    add(
        "replanning", "Replanning", "Bounded retry policy", replan_status,
        replan_explanation, replan_event or escalation_event,
        input_data={"attempted_ngo_ids": donation.get("attempted_ngo_ids", [])},
        output_data={"selected_ngo_id": donation.get("selected_ngo_id"), "attempt_count": donation.get("attempt_count")},
    )

    completion_event = latest("DELIVERY_CONFIRMED", "NGO_ACCEPTED")
    if current_status == "COMPLETED":
        completion_status, completion_explanation = "completed", "A separate demo delivery confirmation was recorded."
    elif current_status == "ACCEPTED":
        completion_status, completion_explanation = "awaiting_human", "Receiver accepted the simulated arrangement; delivery remains unconfirmed."
    elif current_status == "ESCALATED":
        completion_status, completion_explanation = "blocked", "Case escalated to a human coordinator."
    elif current_status in {"BLOCKED", "CANCELLED", "DONOR_REJECTED", "ERROR"}:
        completion_status, completion_explanation = "blocked", f"Completion is unavailable while status is {current_status}."
    else:
        completion_status, completion_explanation = "pending", "A separate verified delivery-confirmation action is required."
    add(
        "completion", "Outcome", "Delivery confirmation", completion_status,
        completion_explanation, completion_event,
        input_data=donation.get("final_outcome"), output_data={"current_status": current_status},
    )

    active = next((stage for stage in stages if stage["status"] in {"running", "awaiting_human"}), None)
    if not active:
        active = next((stage for stage in stages if stage["status"] == "pending"), stages[-1])
    return {
        "donation_id": donation["id"], "current_status": current_status,
        "current_stage": active["key"], "stages": stages,
        "generated_from_persisted_events": True,
    }
