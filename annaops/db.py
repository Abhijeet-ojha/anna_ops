from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from annaops.config import Settings, settings


class BudgetExceeded(RuntimeError):
    category = "budget_exhausted"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, config: Settings = settings):
        self.config = config
        self.path = Path(config.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS donations (
            id TEXT PRIMARY KEY,
            original_message TEXT NOT NULL,
            parsed_json TEXT,
            safety_json TEXT,
            match_json TEXT,
            status TEXT NOT NULL,
            selected_ngo_id TEXT,
            donor_approved INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            attempted_ngo_ids_json TEXT NOT NULL DEFAULT '[]',
            delivery_confirmed_at TEXT,
            final_outcome_json TEXT,
            error_message TEXT,
            synthetic INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS ngo_profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            service_area TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            accepted_categories_json TEXT NOT NULL,
            max_capacity REAL NOT NULL CHECK(max_capacity >= 0),
            available_capacity REAL NOT NULL CHECK(available_capacity >= 0),
            opens_at TEXT NOT NULL,
            closes_at TEXT NOT NULL,
            available INTEGER NOT NULL DEFAULT 1,
            minimum_quantity REAL NOT NULL DEFAULT 0,
            synthetic INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS ngo_demands (
            id TEXT PRIMARY KEY,
            ngo_id TEXT NOT NULL,
            item_name TEXT NOT NULL,
            requested_quantity REAL NOT NULL CHECK(requested_quantity > 0),
            remaining_quantity REAL NOT NULL CHECK(remaining_quantity >= 0),
            unit TEXT NOT NULL,
            minimum_quantity REAL NOT NULL CHECK(minimum_quantity > 0),
            priority INTEGER NOT NULL DEFAULT 0,
            request_deadline TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            response_policy TEXT NOT NULL DEFAULT 'accept_if_eligible',
            synthetic INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(ngo_id) REFERENCES ngo_profiles(id),
            CHECK(remaining_quantity <= requested_quantity)
        );
        CREATE INDEX IF NOT EXISTS idx_ngo_demands_ngo ON ngo_demands(ngo_id, active);
        CREATE TABLE IF NOT EXISTS safety_assessments (
            id TEXT PRIMARY KEY, donation_id TEXT NOT NULL, verdict TEXT NOT NULL,
            rule_version TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS llm_usage (
            id TEXT PRIMARY KEY,
            donation_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            workflow_stage TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            outcome TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            error_category TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS llm_budget_extensions (
            donation_id TEXT PRIMARY KEY,
            extension INTEGER NOT NULL CHECK(extension > 0),
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_llm_usage_donation ON llm_usage(donation_id, created_at);
        CREATE TABLE IF NOT EXISTS match_runs (
            id TEXT PRIMARY KEY, donation_id TEXT NOT NULL, rule_version TEXT NOT NULL,
            details_json TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS approval_decisions (
            id TEXT PRIMARY KEY, donation_id TEXT NOT NULL, decision TEXT NOT NULL,
            actor TEXT NOT NULL, reason TEXT, created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS dispatch_attempts (
            id TEXT PRIMARY KEY, donation_id TEXT NOT NULL, ngo_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL, status TEXT NOT NULL, requested_at TEXT NOT NULL,
            responded_at TEXT, response_reason TEXT,
            UNIQUE(donation_id, attempt_number),
            FOREIGN KEY(donation_id) REFERENCES donations(id),
            FOREIGN KEY(ngo_id) REFERENCES ngo_profiles(id)
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY, donation_id TEXT NOT NULL, event_type TEXT NOT NULL,
            created_at TEXT NOT NULL, previous_status TEXT, resulting_status TEXT,
            details_json TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            donation_id TEXT NOT NULL, action TEXT NOT NULL, key TEXT NOT NULL,
            response_json TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(donation_id, action, key)
        );
        CREATE TABLE IF NOT EXISTS creation_requests (
            key TEXT PRIMARY KEY,
            message_hash TEXT NOT NULL,
            donation_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE TABLE IF NOT EXISTS agent_executions (
            id TEXT PRIMARY KEY,
            donation_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            workflow_stage TEXT NOT NULL,
            objective TEXT NOT NULL,
            execution_mode TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            input_summary_json TEXT NOT NULL,
            structured_output_json TEXT NOT NULL,
            tools_json TEXT NOT NULL,
            selected_action TEXT NOT NULL,
            resulting_status TEXT,
            error_json TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_donations_status ON donations(status);
        CREATE INDEX IF NOT EXISTS idx_audit_donation ON audit_events(donation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_execution_donation ON agent_executions(donation_id, created_at);
        """
        with self.transaction() as connection:
            connection.executescript(schema)

    def create_donation(self, message: str, status: str, synthetic: bool = True) -> dict[str, Any]:
        donation_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO donations (id, original_message, status, synthetic, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (donation_id, message, status, int(synthetic), now, now),
            )
            self._audit(connection, donation_id, "DONATION_CREATED", None, status, {"message_length": len(message)}, "donor", "New offer submitted")
        return self.get_donation(donation_id)

    def create_donation_once(
        self, message: str, status: str, idempotency_key: str | None, synthetic: bool = True
    ) -> tuple[dict[str, Any], bool]:
        """Atomically reserve a creation key and create at most one donation for it."""
        if not idempotency_key:
            return self.create_donation(message, status, synthetic), True
        message_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        donation_id: str
        created = False
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT message_hash, donation_id FROM creation_requests WHERE key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                if existing["message_hash"] != message_hash:
                    raise ValueError("Creation idempotency key was already used for a different donor message")
                donation_id = existing["donation_id"]
            else:
                donation_id = str(uuid.uuid4())
                now = utc_now()
                connection.execute(
                    "INSERT INTO donations (id, original_message, status, synthetic, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (donation_id, message, status, int(synthetic), now, now),
                )
                connection.execute(
                    "INSERT INTO creation_requests (key, message_hash, donation_id, created_at) VALUES (?, ?, ?, ?)",
                    (idempotency_key, message_hash, donation_id, now),
                )
                self._audit(
                    connection, donation_id, "DONATION_CREATED", None, status,
                    {"message_length": len(message), "idempotent_creation": True},
                    "donor", "New offer submitted",
                )
                created = True
        return self.get_donation(donation_id), created

    def get_donation(self, donation_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM donations WHERE id = ?", (donation_id,)).fetchone()
            if row is None:
                raise KeyError(donation_id)
            return self._decode_donation(row, connection)

    def list_donations(self, status: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        clauses, parameters = [], []
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        if search:
            clauses.append("(id LIKE ? OR original_message LIKE ?)")
            parameters.extend([f"%{search}%", f"%{search}%"])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as connection:
            rows = connection.execute(f"SELECT * FROM donations{where} ORDER BY created_at DESC", parameters).fetchall()
            return [self._decode_donation(row, connection, include_details=False) for row in rows]

    def update_donation(self, donation_id: str, previous_status: str, new_status: str, event_type: str,
                        actor: str, reason: str, details: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        allowed = {
            "parsed_json", "safety_json", "match_json", "selected_ngo_id", "donor_approved",
            "attempt_count", "attempted_ngo_ids_json", "delivery_confirmed_at", "final_outcome_json", "error_message"
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Unsupported donation fields: {sorted(invalid)}")
        assignments = ["status = ?", "updated_at = ?", "version = version + 1"]
        values: list[Any] = [new_status, utc_now()]
        for name, value in fields.items():
            assignments.append(f"{name} = ?")
            if name.endswith("_json") and not isinstance(value, str):
                value = json.dumps(value, default=str)
            if isinstance(value, bool):
                value = int(value)
            values.append(value)
        values.extend([donation_id, previous_status])
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE donations SET {', '.join(assignments)} WHERE id = ? AND status = ?", values
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Donation changed concurrently or is in an invalid state")
            self._audit(connection, donation_id, event_type, previous_status, new_status, details or {}, actor, reason)
        return self.get_donation(donation_id)

    def add_safety_assessment(self, donation_id: str, assessment: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO safety_assessments VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), donation_id, assessment["verdict"], assessment["rule_version"], json.dumps(assessment, default=str), utc_now()),
            )

    def add_match_run(self, donation_id: str, result: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO match_runs VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), donation_id, result["rule_version"], json.dumps(result, default=str), utc_now()),
            )

    def record_agent_execution(
        self,
        donation_id: str,
        *,
        agent_type: str,
        workflow_stage: str,
        objective: str,
        execution_mode: str,
        input_summary: dict[str, Any],
        structured_output: dict[str, Any],
        tools: list[dict[str, Any]],
        selected_action: str,
        resulting_status: str | None,
        duration_ms: int,
        provider: str | None = None,
        model: str | None = None,
        error: dict[str, Any] | None = None,
        retry_count: int = 0,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        execution_id = str(uuid.uuid4())
        created_at = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO agent_executions
                (id, donation_id, agent_type, workflow_stage, objective, execution_mode, provider, model,
                 input_summary_json, structured_output_json, tools_json, selected_action, resulting_status,
                 error_json, retry_count, started_at, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    execution_id, donation_id, agent_type, workflow_stage, objective, execution_mode,
                    provider, model, json.dumps(input_summary, default=str),
                    json.dumps(structured_output, default=str), json.dumps(tools, default=str),
                    selected_action, resulting_status, json.dumps(error, default=str) if error else None,
                    retry_count, started_at or created_at, max(0, int(duration_ms)), created_at,
                ),
            )
        return self.list_agent_executions(donation_id)[-1]

    def reserve_llm_request(
        self, donation_id: str, agent_type: str, workflow_stage: str, budget: int,
        provider: str | None, model: str | None
    ) -> str:
        request_id = str(uuid.uuid4())
        with self.transaction() as connection:
            used = connection.execute(
                "SELECT COUNT(*) AS count FROM llm_usage WHERE donation_id = ?",
                (donation_id,),
            ).fetchone()["count"]
            extension_row = connection.execute(
                "SELECT COALESCE(extension, 0) AS extension FROM llm_budget_extensions WHERE donation_id = ?",
                (donation_id,),
            ).fetchone()
            extension = extension_row["extension"] if extension_row else 0
            if used >= budget + extension:
                provider_label = {"groq": "Groq", "gemini": "Gemini"}.get(
                    provider or "", provider or "Configured LLM"
                )
                raise BudgetExceeded(
                    f"{provider_label} request budget exhausted for donation {donation_id}: "
                    f"{used}/{budget} requests used"
                )
            connection.execute(
                """INSERT INTO llm_usage
                (id, donation_id, agent_type, workflow_stage, provider, model, outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'started', ?)""",
                (request_id, donation_id, agent_type, workflow_stage, provider, model, utc_now()),
            )
        return request_id

    def complete_llm_request(
        self, request_id: str, *, outcome: str, retry_count: int = 0,
        input_tokens: int | None = None, output_tokens: int | None = None,
        latency_ms: int = 0, error_category: str | None = None
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE llm_usage SET outcome = ?, retry_count = ?, input_tokens = ?,
                output_tokens = ?, latency_ms = ?, error_category = ? WHERE id = ?""",
                (outcome, retry_count, input_tokens, output_tokens, max(0, latency_ms),
                 error_category, request_id),
            )

    def extend_llm_budget(self, donation_id: str, actor: str, reason: str, extension: int) -> dict[str, int]:
        if extension < 1:
            raise ValueError("Budget extension must be positive")
        with self.transaction() as connection:
            self._audit(
                connection, donation_id, "LLM_BUDGET_EXTENDED", None, None,
                {"extension": extension}, actor, reason,
            )
            connection.execute(
                """INSERT INTO llm_budget_extensions (donation_id, extension) VALUES (?, ?)
                ON CONFLICT(donation_id) DO UPDATE SET extension = extension + excluded.extension""",
                (donation_id, extension),
            )
            used = connection.execute(
                "SELECT COUNT(*) AS count FROM llm_usage WHERE donation_id = ?", (donation_id,)
            ).fetchone()["count"]
        return {"used": used, "extension": extension}

    def llm_usage(self, donation_id: str, budget: int = 0) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_usage WHERE donation_id = ? ORDER BY created_at, rowid",
                (donation_id,),
            ).fetchall()
            items = [dict(row) for row in rows]
        with self.connection() as connection:
            extension_row = connection.execute(
                "SELECT COALESCE(extension, 0) AS extension FROM llm_budget_extensions WHERE donation_id = ?",
                (donation_id,),
            ).fetchone()
            extension = extension_row["extension"] if extension_row else 0
        return {
            "request_count": len(items),
            "budget": budget + extension,
            "budget_remaining": max(0, budget + extension - len(items)),
            "successful_requests": sum(item["outcome"] == "succeeded" for item in items),
            "failed_requests": sum(item["outcome"] == "failed" for item in items),
            "rate_limit_errors": sum(item["error_category"] == "rate_limit" for item in items),
            "retry_count": sum(item["outcome"] == "retrying" for item in items),
            "input_tokens": sum(item["input_tokens"] or 0 for item in items),
            "output_tokens": sum(item["output_tokens"] or 0 for item in items),
            "token_metrics_available": any(
                item["input_tokens"] is not None or item["output_tokens"] is not None for item in items
            ),
            "requests": items,
        }

    def list_agent_executions(self, donation_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_executions WHERE donation_id = ? ORDER BY created_at, rowid",
                (donation_id,),
            ).fetchall()
            executions: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                for source, target in (
                    ("input_summary_json", "input_summary"),
                    ("structured_output_json", "structured_output"),
                    ("tools_json", "tools"),
                    ("error_json", "error"),
                ):
                    raw = item.pop(source)
                    item[target] = json.loads(raw) if raw else (None if target == "error" else {})
                executions.append(item)
            return executions

    def add_approval(self, donation_id: str, decision: str, actor: str, reason: str | None) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO approval_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), donation_id, decision, actor, reason, utc_now()),
            )

    def create_dispatch(self, donation_id: str, ngo_id: str, attempt_number: int) -> dict[str, Any]:
        dispatch_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO dispatch_attempts (id, donation_id, ngo_id, attempt_number, status, requested_at) VALUES (?, ?, ?, ?, 'AWAITING_RESPONSE', ?)",
                (dispatch_id, donation_id, ngo_id, attempt_number, now),
            )
        return {"id": dispatch_id, "donation_id": donation_id, "ngo_id": ngo_id, "attempt_number": attempt_number, "status": "AWAITING_RESPONSE", "requested_at": now}

    def resolve_dispatch(self, donation_id: str, response: str, reason: str | None) -> dict[str, Any]:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM dispatch_attempts WHERE donation_id = ? AND status = 'AWAITING_RESPONSE' ORDER BY attempt_number DESC LIMIT 1",
                (donation_id,),
            ).fetchone()
            if not row:
                raise RuntimeError("No dispatch is awaiting a response")
            connection.execute(
                "UPDATE dispatch_attempts SET status = ?, responded_at = ?, response_reason = ? WHERE id = ?",
                (response.upper(), utc_now(), reason, row["id"]),
            )
            result = dict(row)
            result.update(status=response.upper(), responded_at=utc_now(), response_reason=reason)
            return result

    def pending_dispatch(self, donation_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM dispatch_attempts WHERE donation_id = ? AND status = 'AWAITING_RESPONSE' "
                "ORDER BY attempt_number DESC LIMIT 1", (donation_id,)
            ).fetchone()
            if not row:
                raise RuntimeError("No dispatch is awaiting a response")
            return dict(row)

    def audit(self, donation_id: str, event_type: str, previous_status: str | None, resulting_status: str | None,
              details: dict[str, Any], actor: str, reason: str | None) -> None:
        with self.transaction() as connection:
            self._audit(connection, donation_id, event_type, previous_status, resulting_status, details, actor, reason)

    def _audit(self, connection: sqlite3.Connection, donation_id: str, event_type: str,
               previous_status: str | None, resulting_status: str | None, details: dict[str, Any],
               actor: str, reason: str | None) -> None:
        connection.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), donation_id, event_type, utc_now(), previous_status, resulting_status,
             json.dumps(details, default=str), actor, reason),
        )

    def idempotent_response(self, donation_id: str, action: str, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        with self.connection() as connection:
            row = connection.execute(
                "SELECT response_json FROM idempotency_keys WHERE donation_id = ? AND action = ? AND key = ?",
                (donation_id, action, key),
            ).fetchone()
            return json.loads(row["response_json"]) if row else None

    def save_idempotent_response(self, donation_id: str, action: str, key: str | None, response: dict[str, Any]) -> None:
        if not key:
            return
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO idempotency_keys VALUES (?, ?, ?, ?, ?)",
                (donation_id, action, key, json.dumps(response, default=str), utc_now()),
            )

    def seed_ngos(self, profiles: list[dict[str, Any]]) -> int:
        with self.transaction() as connection:
            for profile in profiles:
                connection.execute(
                    """INSERT INTO ngo_profiles
                    (id, name, service_area, latitude, longitude, accepted_categories_json, max_capacity,
                     available_capacity, opens_at, closes_at, available, minimum_quantity, synthetic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(id) DO UPDATE SET name=excluded.name, service_area=excluded.service_area,
                    latitude=excluded.latitude, longitude=excluded.longitude,
                    accepted_categories_json=excluded.accepted_categories_json,
                    max_capacity=excluded.max_capacity,
                    opens_at=excluded.opens_at, closes_at=excluded.closes_at,
                    available=excluded.available, minimum_quantity=excluded.minimum_quantity""",
                    (profile["id"], profile["name"], profile["service_area"], profile["latitude"], profile["longitude"],
                     json.dumps(profile["accepted_categories"]), profile["max_capacity"], profile["available_capacity"],
                     profile["opens_at"], profile["closes_at"], int(profile["available"]), profile["minimum_quantity"]),
                )
                for demand in profile.get("demands", []):
                    connection.execute(
                        """INSERT INTO ngo_demands
                        (id, ngo_id, item_name, requested_quantity, remaining_quantity, unit,
                         minimum_quantity, priority, request_deadline, active, response_policy, synthetic)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(id) DO UPDATE SET ngo_id=excluded.ngo_id,
                        item_name=excluded.item_name, requested_quantity=excluded.requested_quantity,
                        unit=excluded.unit, minimum_quantity=excluded.minimum_quantity,
                        priority=excluded.priority, request_deadline=excluded.request_deadline,
                        active=excluded.active, response_policy=excluded.response_policy""",
                        (
                            demand["id"], profile["id"], demand["item_name"],
                            demand["requested_quantity"], demand.get("remaining_quantity", demand["requested_quantity"]),
                            demand["unit"], demand.get("minimum_quantity", demand["requested_quantity"]),
                            demand.get("priority", 0), demand.get("request_deadline"),
                            int(demand.get("active", True)), demand.get("response_policy", "accept_if_eligible"),
                        ),
                    )
        return len(profiles)

    def list_ngos(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM ngo_profiles ORDER BY name").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["accepted_categories"] = json.loads(item.pop("accepted_categories_json"))
                item["available"] = bool(item["available"])
                item["synthetic"] = bool(item["synthetic"])
                item["demands"] = [
                    dict(demand)
                    for demand in connection.execute(
                        "SELECT id, item_name, requested_quantity, remaining_quantity, unit, minimum_quantity, "
                        "priority, request_deadline, active, response_policy, synthetic "
                        "FROM ngo_demands WHERE ngo_id = ? ORDER BY priority DESC, item_name",
                        (item["id"],),
                    ).fetchall()
                ]
                for demand in item["demands"]:
                    demand["active"] = bool(demand["active"])
                    demand["synthetic"] = bool(demand["synthetic"])
                result.append(item)
            return result

    def reserve_ngo_demand(self, ngo_id: str, offer_items: list[dict[str, Any]], donation_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            profile = connection.execute(
                "SELECT available, available_capacity FROM ngo_profiles WHERE id = ?", (ngo_id,)
            ).fetchone()
            if profile is None:
                raise ValueError(f"Unknown synthetic receiver: {ngo_id}")
            if not profile["available"]:
                raise ValueError("Selected receiver is unavailable")
            total = sum(float(item["quantity"]) for item in offer_items)
            if profile["available_capacity"] < total:
                raise ValueError("Selected receiver no longer has sufficient capacity")
            demands = connection.execute(
                "SELECT * FROM ngo_demands WHERE ngo_id = ? AND active = 1 ORDER BY priority DESC",
                (ngo_id,),
            ).fetchall()
            matched: list[sqlite3.Row] = []
            for item in offer_items:
                name = str(item["name"]).strip().lower()
                unit = str(item["unit"]).strip().lower()
                match = next(
                    (
                        demand for demand in demands
                        if demand["remaining_quantity"] >= item["quantity"]
                        and demand["minimum_quantity"] <= item["quantity"]
                        and demand["unit"].lower() == unit
                        and (
                            demand["item_name"].lower() == name
                            or (demand["item_name"].lower() == "vegetarian food" and item.get("category") == "vegetarian")
                        )
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"Receiver demand no longer matches item '{name}' in {unit}")
                matched.append(match)
            for demand, item in zip(matched, offer_items):
                remaining = demand["remaining_quantity"] - float(item["quantity"])
                connection.execute(
                    "UPDATE ngo_demands SET remaining_quantity = ?, active = ? WHERE id = ?",
                    (remaining, int(remaining > 0), demand["id"]),
                )
            connection.execute(
                "UPDATE ngo_profiles SET available_capacity = available_capacity - ? WHERE id = ?",
                (total, ngo_id),
            )
            return {
                "ngo_id": ngo_id,
                "donation_id": donation_id,
                "reserved_quantity": total,
                "demands": [{"id": demand["id"], "remaining_quantity": demand["remaining_quantity"] - float(item["quantity"])}
                            for demand, item in zip(matched, offer_items)],
            }

    def metrics(self) -> dict[str, Any]:
        with self.connection() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM donations GROUP BY status").fetchall()
            by_status = {row["status"]: row["count"] for row in rows}
            by_status["TOTAL"] = sum(by_status.values())
            by_status["AWAITING_CLARIFICATION"] = by_status.get("NEEDS_CLARIFICATION", 0)
            by_status["BLOCKED_OR_REVIEW"] = by_status.get("BLOCKED", 0) + by_status.get("REQUIRES_HUMAN_REVIEW", 0)
            by_status["ELIGIBLE_AWAITING_APPROVAL"] = by_status.get("AWAITING_DONOR_APPROVAL", 0)
            by_status["PENDING_NGO_RESPONSE"] = by_status.get("AWAITING_NGO_RESPONSE", 0)
            by_status["ACCEPTED_AWAITING_DELIVERY"] = by_status.get("ACCEPTED", 0)
            by_status["COMPLETED_DEMO_DELIVERIES"] = by_status.get("COMPLETED", 0)
            by_status["ESCALATED_CASES"] = by_status.get("ESCALATED", 0)
            usage = connection.execute(
                """SELECT COUNT(*) AS requests,                 SUM(CASE WHEN outcome = 'retrying' THEN 1 ELSE 0 END) AS retries,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN error_category = 'rate_limit' THEN 1 ELSE 0 END) AS rate_limits,
                SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens
                FROM llm_usage"""
            ).fetchone()
            by_status["GEMINI_REQUESTS"] = usage["requests"] or 0
            by_status["GEMINI_RETRIES"] = usage["retries"] or 0
            by_status["GEMINI_FAILED_REQUESTS"] = usage["failed"] or 0
            by_status["GEMINI_RATE_LIMIT_ERRORS"] = usage["rate_limits"] or 0
            by_status["GEMINI_INPUT_TOKENS"] = usage["input_tokens"]
            by_status["GEMINI_OUTPUT_TOKENS"] = usage["output_tokens"]
            return by_status

    def _decode_donation(self, row: sqlite3.Row, connection: sqlite3.Connection, include_details: bool = True) -> dict[str, Any]:
        result = dict(row)
        for field in ("parsed_json", "safety_json", "match_json", "attempted_ngo_ids_json", "final_outcome_json"):
            key = field.removesuffix("_json")
            value = result.pop(field)
            result[key] = json.loads(value) if value else ([] if field == "attempted_ngo_ids_json" else None)
        result["donor_approved"] = bool(result["donor_approved"])
        result["synthetic"] = bool(result["synthetic"])
        if include_details:
            result["dispatch_attempts"] = [dict(item) for item in connection.execute(
                "SELECT * FROM dispatch_attempts WHERE donation_id = ? ORDER BY attempt_number", (result["id"],)
            ).fetchall()]
            result["approval_decisions"] = [dict(item) for item in connection.execute(
                "SELECT * FROM approval_decisions WHERE donation_id = ? ORDER BY created_at", (result["id"],)
            ).fetchall()]
            events = []
            for item in connection.execute(
                "SELECT * FROM audit_events WHERE donation_id = ? ORDER BY created_at", (result["id"],)
            ).fetchall():
                event = dict(item)
                event["details"] = json.loads(event.pop("details_json"))
                events.append(event)
            result["audit_events"] = events
            result["agent_executions"] = self.list_agent_executions(result["id"])
            result["llm_usage"] = self.llm_usage(result["id"], self.config.llm_budget_per_donation)
        return result


database = Database()
