const state = { donations: [], selectedId: null, createRequestKey: null, creating: false, scenarioKeys: {}, actionKeys: {} };
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const pretty = (value) => String(value ?? "").replaceAll("_", " ").toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
const fmt = (value) => value ? new Date(value).toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : "—";

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {"Content-Type": "application/json", ...(options.headers || {})}, ...options});
  const payload = await response.json().catch(() => ({detail: response.statusText}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map(item => item.msg || String(item)).join("; ")
      : (payload.detail || payload.error || "Request failed");
    throw new Error(detail);
  }
  return payload;
}

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.hidden = false;
  clearTimeout(window.toastTimer); window.toastTimer = setTimeout(() => node.hidden = true, 3500);
}

async function loadHealth() {
  const health = await api("/api/health");
  $("#llmMode").textContent = health.llm === "configured" ? `Gemini 3.6 Flash (Mandatory)` : "Not configured (Startup Failed)";
  $("#routingMode").textContent = health.routing === "configured_osrm" ? "Configured OSRM" : "Straight-line fallback";
}

async function loadMetrics() {
  const data = await api("/api/metrics");
  const groups = [
    ["TOTAL", "Total offers"], ["AWAITING_CLARIFICATION", "Awaiting clarification"],
    ["BLOCKED_OR_REVIEW", "Blocked / review"], ["ELIGIBLE_AWAITING_APPROVAL", "Awaiting approval"],
    ["PENDING_NGO_RESPONSE", "NGO pending"], ["ACCEPTED_AWAITING_DELIVERY", "Awaiting delivery"],
    ["COMPLETED_DEMO_DELIVERIES", "Completed demo"], ["ESCALATED_CASES", "Escalated"],
  ];
  $("#metrics").innerHTML = groups.map(([key, label]) => `<div class="metric"><strong>${data[key] || 0}</strong><span>${label}</span></div>`).join("");
}

async function loadDonations() {
  const params = new URLSearchParams();
  if ($("#statusFilter").value) params.set("status", $("#statusFilter").value);
  if ($("#searchInput").value.trim()) params.set("search", $("#searchInput").value.trim());
  const records = await api(`/api/donations?${params}`);
  state.donations = [...new Map(records.map(item => [item.id, item])).values()];
  const list = $("#donationList");
  if (!state.donations.length) { list.innerHTML = '<div class="empty">No offers match this view.</div>'; return; }
  list.innerHTML = state.donations.map(item => `
    <button class="donation-row ${state.selectedId === item.id ? "selected" : ""}" data-id="${item.id}">
      <strong>${esc(item.original_message)}</strong><span class="status ${item.status}">${pretty(item.status)}</span>
      <time>${fmt(item.updated_at)}</time><span class="id">${item.id}</span>
    </button>`).join("");
  list.querySelectorAll("[data-id]").forEach(button => button.addEventListener("click", () => openDetail(button.dataset.id)));
}

function populateStatuses() {
  const values = ["NEEDS_CLARIFICATION","REQUIRES_HUMAN_REVIEW","BLOCKED","AWAITING_DONOR_APPROVAL","AWAITING_NGO_RESPONSE","ACCEPTED","ESCALATED","COMPLETED","DONOR_REJECTED","ERROR"];
  $("#statusFilter").insertAdjacentHTML("beforeend", values.map(value => `<option value="${value}">${pretty(value)}</option>`).join(""));
}

function nowExamples(kind) {
  const now = new Date(), prepared = new Date(now.getTime() - 20 * 60000), deadline = new Date(now.getTime() + 3 * 3600000);
  const stamp = d => d.toISOString();
  if (kind === "complete") return `40 portions vegetarian biryani prepared at ${stamp(prepared)}, kept hot, pickup by ${stamp(deadline)} from Demo Campus Jaipur, coordinates 26.9124, 75.7873`;
  if (kind === "review") return `24 portions vegetarian rice prepared at ${stamp(prepared)}, room temperature, pickup by ${stamp(deadline)} from Demo Hall Jaipur, coordinates 26.9050, 75.7750`;
  return "We have leftover food after our college event.";
}

async function submitDonation(event) {
  event.preventDefault();
  if (state.creating) return;
  const button = $("#submitButton"); state.creating = true; button.disabled = true; button.firstElementChild.textContent = "Coordinating…";
  state.createRequestKey ||= crypto.randomUUID();
  try {
    const result = await api("/api/donations", {method: "POST", body: JSON.stringify({message: $("#message").value, idempotency_key: state.createRequestKey})});
    state.createRequestKey = null; $("#message").value = ""; state.selectedId = result.id;
    await refreshAll(); await openDetail(result.id); toast("One offer created and persisted.");
  } catch (error) { toast(error.message); }
  finally { state.creating = false; button.disabled = false; button.firstElementChild.textContent = "Run coordinated intake"; }
}

async function openDetail(id) {
  state.selectedId = id;
  const [donation, workflow] = await Promise.all([api(`/api/donations/${id}`), api(`/api/donations/${id}/workflow`)]);
  $("#detailTitle").textContent = `Offer ${id.slice(0, 8)}`;
  $("#detailContent").innerHTML = renderDetail(donation, workflow);
  bindDetailActions(donation);
  renderWorkflowSpotlight(donation, workflow);
  $("#drawerBackdrop").hidden = false; $("#detailDrawer").classList.add("open"); $("#detailDrawer").setAttribute("aria-hidden", "false");
}

function renderDetail(d, workflow) {
  const p = d.parsed || {}, safety = d.safety || {}, match = d.match || {};
  const foodItems = (p.food_items || []).map(item => `
    <div class="food-item"><strong>${esc(item.name)}</strong><span>${esc(item.quantity)} ${esc(item.unit)} · ${pretty(item.category)}</span></div>`).join("");
  const candidates = (match.candidates || []).slice(0, 5).map(c => `
    <div class="candidate"><div class="candidate-top"><strong>#${c.rank} ${esc(c.ngo_name)}</strong><small>${c.score} points</small></div>
    <p>${c.route.distance_km == null ? "Route unavailable" : `${c.route.distance_km} km`} · ${pretty(c.route.method)}</p>
    <ul class="reason-list">${(c.reasons || []).map(reason => `<li>${esc(reason)}</li>`).join("")}</ul>
    <p class="route-note">${esc(c.route.note)}</p></div>`).join("");
  const exclusions = Object.entries(match.exclusions || {}).map(([ngo, reasons]) => `
    <div class="exclusion"><strong>${esc(ngo)}</strong><p>${reasons.map(esc).join(" · ")}</p></div>`).join("");
  const safetyReasons = (safety.reasons || []).map(r => `<li>${esc(r)}</li>`).join("");
  const clarification = renderClarification(d);
  const actions = renderActions(d);
  const events = (d.audit_events || []).slice().reverse().map(e => `<div class="event"><strong>${pretty(e.event_type)}</strong><span>${fmt(e.created_at)} · ${esc(e.actor)}</span><p>${esc(e.reason || "")}</p>${Object.keys(e.details || {}).length ? `<p class="route-note">${esc(JSON.stringify(e.details))}</p>` : ""}</div>`).join("");
  const approvals = (d.approval_decisions || []).map(item => `<div class="history-row"><strong>${pretty(item.decision)}</strong><span>${esc(item.actor)} · ${fmt(item.created_at)}</span></div>`).join("");
  const dispatches = (d.dispatch_attempts || []).map(a => `<div class="history-row"><strong>Attempt ${a.attempt_number} · ${esc(a.ngo_id)}</strong><span>${pretty(a.status)} · ${fmt(a.responded_at || a.requested_at)}</span></div>`).join("");
  const executions = renderAgentExecutions(d.agent_executions || []);
  return `
    <div class="detail-status"><span class="status ${d.status}">${pretty(d.status)}</span><small>Updated ${fmt(d.updated_at)}</small></div>
    <div class="demo-banner">Synthetic demonstration record. No real receiver or delivery service is connected.</div>
    ${actions}${clarification}
    <section class="detail-card agent-observatory"><div class="observatory-heading"><div><h3>Agent execution observatory</h3><p class="observatory-note">Validated Gemini 3.6 Flash decisions and real tool results, shown newest first. Expand any run for evidence.</p></div><span>${(d.agent_executions || []).length} runs</span></div>${executions || '<div class="empty compact">No agent executions were recorded for this case.</div>'}</section>
    <section class="detail-card"><h3>Actual agent progression</h3><div class="detail-workflow">${renderWorkflowDetail(workflow)}</div></section>
    <section class="detail-card"><h3>Original donor message</h3><p class="quote">“${esc(d.original_message)}”</p></section>
    <section class="detail-card"><h3>Structured offer</h3>
      ${foodItems ? `<div class="food-items"><span class="section-label">Itemized source facts</span>${foodItems}</div>` : ""}
      <div class="facts">
      ${fact("Food", p.food_type)}${fact("Category", pretty(p.food_category))}${fact("Aggregate quantity", p.quantity ? `${p.quantity} ${p.unit || ""}` : (foodItems ? "Itemized above" : null))}
      ${fact("Prepared", fmt(p.prepared_at))}${fact("Pickup deadline", fmt(p.pickup_deadline))}${fact("Storage", p.storage_conditions)}
      ${fact("Location", p.donor_location)}${fact("Extraction", pretty(p.extraction_source))}
    </div></section>
    <section class="detail-card"><h3>Field evidence</h3><div class="evidence-grid">
      ${evidence("Explicit fields", p.explicit_fields)}${evidence("Inferred fields", p.inferred_fields)}
      ${evidence("Missing fields", p.missing_fields)}${evidence("Conflicts", p.contradictions)}
    </div></section>
    ${safety.verdict ? `<section class="detail-card"><h3>Safety assessment · ${pretty(safety.verdict)}</h3><ul class="reason-list">${safetyReasons}</ul><p>${esc(safety.disclaimer)}</p></section>` : ""}
    ${candidates ? `<section class="detail-card"><h3>Ranked synthetic receivers</h3>${candidates}</section>` : ""}
    ${exclusions ? `<section class="detail-card"><h3>Receiver exclusions · actual rule output</h3>${exclusions}</section>` : ""}
    ${approvals ? `<section class="detail-card"><h3>Donor approval history</h3>${approvals}</section>` : ""}
    ${dispatches ? `<section class="detail-card"><h3>Simulated dispatch & response history</h3>${dispatches}<p>No real NGO communication occurred.</p></section>` : ""}
    <section class="detail-card"><h3>Audit trail</h3><div class="timeline">${events}</div><a class="download" href="/api/donations/${d.id}/record.csv">Download synthetic CSV record ↗</a></section>`;
}

function renderAgentExecutions(executions) {
  return executions.slice().reverse().map((execution, index) => {
    const output = execution.structured_output || {};
    const mode = execution.execution_mode || "deterministic";
    const isError = Boolean(execution.error) || mode === "failed";
    const category = execution.error?.category || "";
    const isRetryable = category === "retryable_provider_error";

    let modeBadgeClass = "deterministic";
    let modeLabel = "Deterministic Rule / Gate";
    if (mode === "llm" && !isError) {
      modeBadgeClass = "llm-success";
      modeLabel = `Gemini 3.6 Flash · Succeeded`;
    } else if (isRetryable) {
      modeBadgeClass = "provider-retryable";
      modeLabel = `Retryable Provider Failure`;
    } else if (mode === "failed" || isError) {
      modeBadgeClass = "llm-failed";
      modeLabel = `LLM Execution Failed`;
    } else if (mode === "test_mock") {
      modeBadgeClass = "test-mock";
      modeLabel = `Test Mock · ${execution.model || "gemini-3.6-flash"}`;
    }

    const providerModel = execution.provider ? `${execution.provider} · ${execution.model || "gemini-3.6-flash"}` : (execution.model || "Authoritative Backend");

    const toolRows = (execution.tools || []).map(tool => `<div class="tool-event">
      <div><span class="tool-index">${tool.sequence || "•"}</span><strong>${esc(tool.name)}</strong></div>
      <p>${esc(tool.purpose || "Validated backend tool")}</p>
      <details><summary>Validated arguments and result</summary><pre>${esc(JSON.stringify({arguments: tool.arguments || {}, result: tool.result || {}}, null, 2))}</pre></details>
    </div>`).join("");

    const evidenceList = (output.evidence || []).map(item => `<li>${esc(item)}</li>`).join("");

    const retryGuidance = category === "configuration_error"
      ? "Verify that ANNAOPS_GEMINI_API_KEY is configured with access to gemini-3.6-flash."
      : category === "retryable_provider_error"
      ? "The provider is temporarily rate-limited or unreachable. Safe to retry when service recovers."
      : category === "structured_output_validation_failure"
      ? "The model output failed schema validation. Safe to retry."
      : category === "agent_protocol_failure"
      ? "Agent protocol violation: model proposed an invalid action or unknown tool. Safe to retry."
      : category === "agent_execution_limit_failure"
      ? "Maximum agent iterations reached without a terminal decision. Safe to retry."
      : (execution.error ? "Execution failed safely without state corruption. Safe to retry." : "");

    return `<details class="agent-execution ${modeBadgeClass}" ${index === 0 ? "open" : ""}>
      <summary class="agent-execution-summary">
        <span class="summary-chevron" aria-hidden="true">›</span>
        <span class="agent-identity"><strong>${pretty(execution.agent_type)}</strong><small>${pretty(execution.workflow_stage)} · ${fmt(execution.started_at)}</small></span>
        <span class="summary-transition"><small>Decision</small><strong>${pretty(execution.selected_action)}</strong></span>
        <span class="summary-arrow" aria-hidden="true">→</span>
        <span class="summary-transition"><small>State</small><strong>${pretty(execution.resulting_status)}</strong></span>
        <span class="mode-badge ${modeBadgeClass}">${esc(modeLabel)}</span>
      </summary>
      <div class="agent-execution-body">
        <p class="agent-objective">${esc(execution.objective)}</p>
        <div class="execution-meta">
          <span><small>Provider & Model</small><strong>${esc(providerModel)}</strong></span>
          <span><small>Duration</small><strong>${duration(execution.duration_ms)}</strong></span>
          <span><small>Extra model turns</small><strong>${execution.retry_count || 0}</strong></span>
          <span><small>Tools used</small><strong>${(execution.tools || []).length}</strong></span>
        </div>
        ${execution.input_summary && Object.keys(execution.input_summary).length ? `
          <div class="execution-section">
            <h4>Relevant input summary</h4>
            <pre class="input-summary-box">${esc(JSON.stringify(execution.input_summary, null, 2))}</pre>
          </div>
        ` : ""}
        ${evidenceList ? `<div class="execution-section"><h4>Decision evidence</h4><ul class="reason-list execution-evidence">${evidenceList}</ul></div>` : ""}
        ${execution.error ? `
          <div class="execution-error-box">
            <p class="execution-error"><strong>Failure Category:</strong> ${esc(execution.error.category || execution.error.type)}<br>
            <strong>Diagnostic Message:</strong> ${esc(execution.error.message)}</p>
            ${retryGuidance ? `<p class="retry-guidance"><strong>Safe Retry Guidance:</strong> ${esc(retryGuidance)}</p>` : ""}
          </div>
        ` : ""}
        ${toolRows ? `<div class="tool-sequence"><span class="section-label">Actual tool sequence</span>${toolRows}</div>` : '<p class="no-tools">No backend tool was needed for this decision.</p>'}
      </div>
    </details>`;
  }).join("");
}

function fact(label, value) { return `<div class="fact"><span>${label}</span><strong>${esc(value || "Not supplied")}</strong></div>`; }
function evidence(label, values) { const list = values || []; return `<div class="evidence-box"><span>${label}</span><p>${list.length ? list.map(esc).join(" · ") : "None"}</p></div>`; }

function duration(value) {
  if (value == null) return "";
  return value < 1000 ? `${value} ms` : `${(value / 1000).toFixed(1)} s`;
}

function renderWorkflowDetail(workflow) {
  return (workflow.stages || []).map(stage => `<div class="detail-stage ${stage.status}">
    <span class="stage-rail"></span><div><strong>${esc(stage.label)}</strong><span class="component">${esc(stage.component)} · ${pretty(stage.status)}</span></div>
    <div><p>${esc(stage.explanation)}</p>${stage.fallback ? `<span class="fallback-note">${esc(stage.fallback)}</span>` : ""}</div>
    <time>${stage.timestamp ? fmt(stage.timestamp) : "Not run"}${stage.duration_ms != null ? ` · ${duration(stage.duration_ms)}` : ""}</time>
  </div>`).join("");
}

function renderWorkflowTrack(workflow) {
  return `<div class="workflow-track">${(workflow.stages || []).map(stage => `<article class="workflow-node ${stage.status}">
    <div class="node-meta"><span>${pretty(stage.status)}</span><span>${stage.duration_ms != null ? duration(stage.duration_ms) : ""}</span></div>
    <strong>${esc(stage.label)}</strong><p>${esc(stage.explanation)}</p>${stage.fallback ? `<span class="fallback-note">${esc(stage.fallback)}</span>` : ""}
  </article>`).join("")}</div>`;
}

function renderWorkflowSpotlight(donation, workflow) {
  $("#monitorTitle").textContent = `${donation.id.slice(0, 8)} · ${pretty(donation.status)}`;
  $("#workflowSpotlight").innerHTML = renderWorkflowTrack(workflow);
}

async function loadWorkflowSpotlight() {
  const id = state.selectedId || state.donations[0]?.id;
  if (!id) return;
  try {
    const [donation, workflow] = await Promise.all([api(`/api/donations/${id}`), api(`/api/donations/${id}/workflow`)]);
    renderWorkflowSpotlight(donation, workflow);
  } catch (error) { $("#workflowSpotlight").innerHTML = `<div class="empty">${esc(error.message)}</div>`; }
}

function renderClarification(d) {
  if (!["NEEDS_CLARIFICATION", "REQUIRES_HUMAN_REVIEW"].includes(d.status)) return "";
  const fields = d.parsed?.missing_fields || [];
  if (!fields.length && d.status === "REQUIRES_HUMAN_REVIEW") return "";
  const placeholders = {prepared_at: "e.g. 6 pm or 2026-10-23T18:00:00+05:30", pickup_deadline: "e.g. 8 pm or a full ISO timestamp", storage_conditions: "e.g. refrigerated, kept hot, cold", donor_location: "Pickup location"};
  return `<section class="detail-card"><h3>Supply missing facts</h3><form id="clarifyForm" class="clarify-grid">${fields.map(field => `<label>${pretty(field)}<input name="${field}" placeholder="${esc(placeholders[field] || `Enter ${pretty(field).toLowerCase()}`)}" required></label>`).join("")}<p class="form-error" id="clarifyError" role="alert" hidden></p><button class="action" id="clarifySubmit" type="submit">Submit clarification</button></form></section>`;
}

function renderActions(d) {
  if (d.status === "AWAITING_DONOR_APPROVAL") return `<div class="actions"><button class="action" data-action="approve">Approve proposed receiver</button><button class="action danger" data-action="reject">Reject proposal</button></div>`;
  if (d.status === "AWAITING_NGO_RESPONSE") return `<div class="actions"><button class="action wide" data-action="accept">Simulate NGO acceptance</button><button class="action danger" data-action="decline">Simulate decline</button><button class="action amber" data-action="timeout">Simulate timeout</button></div>`;
  if (d.status === "ACCEPTED") return `<div class="actions"><button class="action wide" data-action="delivery">Record separate demo delivery confirmation</button></div>`;
  if (d.status === "ERROR") return `<div class="actions"><button class="action wide" data-action="retry">Retry workflow with Gemini 3.6 Flash</button><button class="action amber" data-action="escalate">Escalate to coordinator</button></div>`;
  if (["NEEDS_CLARIFICATION","REQUIRES_HUMAN_REVIEW","AWAITING_DONOR_APPROVAL","AWAITING_NGO_RESPONSE"].includes(d.status)) return `<div class="actions"><button class="action amber wide" data-action="escalate">Escalate to coordinator</button></div>`;
  return "";
}

function bindDetailActions(d) {
  document.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => runAction(d.id, button.dataset.action, button)));
  const form = $("#clarifyForm"); if (form) form.addEventListener("submit", event => submitClarification(event, d.id));
}

async function runAction(id, action, button) {
  let path, body;
  const actionKey = `${id}:${action}`;
  state.actionKeys[actionKey] ||= crypto.randomUUID();
  const key = state.actionKeys[actionKey];
  if (["approve","reject"].includes(action)) { path = action; body = {actor: "demo_donor", idempotency_key: key}; }
  else if (["accept","decline","timeout"].includes(action)) { path = "ngo-response"; body = {actor: "demo_ngo", response: action, idempotency_key: key}; }
  else if (action === "delivery") { path = "delivery-confirmation"; body = {actor: "demo_coordinator", verification_note: "Delivery confirmation recorded by the hackathon demo operator.", idempotency_key: key}; }
  else if (action === "retry") { path = "retry"; body = {actor: "demo_coordinator", reason: "Retrying after provider recovery", idempotency_key: key}; }
  else { path = "escalate"; body = {actor: "demo_coordinator", reason: "Manual coordinator review requested in demo.", idempotency_key: key}; }
  button.disabled = true;
  try {
    await api(`/api/donations/${id}/${path}`, {method: "POST", body: JSON.stringify(body)});
    delete state.actionKeys[actionKey]; await refreshAll(); await openDetail(id); toast(`${pretty(action)} recorded.`);
  } catch (error) { button.disabled = false; toast(error.message); }
}

async function submitClarification(event, id) {
  event.preventDefault();
  const element = event.currentTarget, form = new FormData(element), fields = {};
  const button = element.querySelector("#clarifySubmit"), errorNode = element.querySelector("#clarifyError");
  for (const [key, value] of form.entries()) fields[key] = ["quantity","donor_latitude","donor_longitude"].includes(key) ? Number(value) : String(value).trim();
  button.disabled = true; button.textContent = "Validating and saving…"; errorNode.hidden = true;
  try {
    state.actionKeys[`${id}:clarify`] ||= crypto.randomUUID();
    await api(`/api/donations/${id}/clarify`, {method: "POST", body: JSON.stringify({fields, idempotency_key: state.actionKeys[`${id}:clarify`]})});
    delete state.actionKeys[`${id}:clarify`];
    await refreshAll(); await openDetail(id); toast("Clarification saved and workflow re-evaluated.");
  } catch (error) {
    errorNode.textContent = error.message; errorNode.hidden = false; toast(error.message);
    button.disabled = false; button.textContent = "Submit clarification";
  }
}

async function runScenario(scenario, button) {
  state.scenarioKeys[scenario] ||= crypto.randomUUID();
  const original = button.innerHTML; button.disabled = true;
  button.insertAdjacentHTML("beforeend", '<small class="scenario-running">Running Gemini 3.6 Flash workflow…</small>');
  try {
    const result = await api(`/api/demo/scenario/${scenario}`, {
      method: "POST", body: JSON.stringify({idempotency_key: state.scenarioKeys[scenario]})
    });
    delete state.scenarioKeys[scenario]; button.innerHTML = original; button.disabled = false; state.selectedId = result.id;
    await refreshAll(); await openDetail(result.id); toast(`${pretty(scenario)} scenario created.`);
  } catch (error) {
    button.innerHTML = original; button.disabled = false; toast(error.message);
  }
}

async function refreshAll() { await Promise.all([loadMetrics(), loadDonations()]); await loadWorkflowSpotlight(); }
function closeDrawer() { $("#detailDrawer").classList.remove("open"); $("#detailDrawer").setAttribute("aria-hidden", "true"); $("#drawerBackdrop").hidden = true; }

document.addEventListener("DOMContentLoaded", async () => {
  populateStatuses();
  $("#donationForm").addEventListener("submit", submitDonation);
  $("#refreshButton").addEventListener("click", refreshAll);
  $("#statusFilter").addEventListener("change", loadDonations);
  let searchTimer; $("#searchInput").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadDonations, 250); });
  document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => $("#message").value = nowExamples(button.dataset.example)));
  document.querySelectorAll("[data-scenario]").forEach(button => button.addEventListener("click", () => runScenario(button.dataset.scenario, button)));
  $("#closeDrawer").addEventListener("click", closeDrawer); $("#drawerBackdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", event => { if (event.key === "Escape") closeDrawer(); });
  try { await Promise.all([loadHealth(), refreshAll()]); } catch (error) { toast(error.message); }
});
