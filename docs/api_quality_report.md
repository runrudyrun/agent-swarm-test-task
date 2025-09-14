# API Quality Report — Agent Swarm

Timestamp: 2025-09-14T16:25 (local)

This report evaluates eight end-to-end API queries against the running service, focusing on routing, language consistency, usefulness, accuracy, source handling, and UX. References to implementation files are given for traceability.

Implementation references:
- Router: `agents/router_agent.py` (intent classification, escalation, deterministic ticket override)
- Knowledge: `agents/knowledge_agent.py` (RAG prompt, sources, language)
- Support: `agents/support_agent.py` (diagnostics, LLM triage for tickets, localized replies)
- User store: `tools/user_store.py` (mock persistence)

Service health at test time:
- GET /health → status: healthy
- Agents: { router: healthy, knowledge: healthy, support: healthy, personality: healthy }

---

## Query 1
Input: "What are the fees of the Maquininha Smart" (user_id: client789)
- Agent: knowledge
- Language: English (matches input)
- Sources: present and localized as "Sources"
- Leakage: no "CONTEXT" found
- Usefulness: High — clear fees (debit 1.37%, credit 3.15%, 12x 12.40%, Pix 0%), device cost, context notes
- Accuracy: Good for provided material; mentions tier dependency
- Verdict: SUCCESS

## Query 2
Input: "What is the cost of the Maquininha Smart?" (user_id: client789)
- Agent: knowledge
- Language: English
- Sources: present
- Usefulness: High — direct price (12 × R$16,58 = R$198,96)
- Accuracy: High for product page reference
- Verdict: SUCCESS

## Query 3
Input: "What are the rates for debit and credit card transactions?" (user_id: client789)
- Agent: knowledge
- Language: English
- Sources: present
- Usefulness: High — splits in-person vs. online, gives ranges and guidance to personalize
- Accuracy: Good with ranges and caveats
- Verdict: SUCCESS

## Query 4
Input: "How can I use my phone as a card machine?" (user_id: client789)
- Agent: knowledge
- Language: English
- Sources: present
- Usefulness: High — step-by-step InfiniteTap/Tap to Pay, requirements, security, fees
- Verdict: SUCCESS

## Query 5
Input: "Quando foi o último jogo do Palmeiras?" (user_id: client789)
- Agent: knowledge (off-topic deflection)
- Language: Portuguese (matches input)
- Sources: PT label ("Fontes")
- Usefulness: Appropriate refusal; offers InfinitePay topics
- Verdict: SUCCESS

## Query 6
Input: "Quais as principais notícias de São Paulo hoje?" (user_id: client789)
- Agent: knowledge (off-topic deflection)
- Language: Portuguese
- Sources: PT label
- Usefulness: Appropriate refusal + relevant InfinitePay areas
- Verdict: SUCCESS

## Query 7
Input: "Why I am not able to make transfers?" (user_id: user789)
- Agent: support
- Language: English
- Personalization: Yes — uses mock data (status: suspended; balance R$ 890,25; 0 pending; 1 failed)
- Usefulness: High — explains reason, next steps, offers ticket
- Verdict: SUCCESS

## Query 8
Input: "I can't sign in to my account." (user_id: user123)
- Agent: support
- Language: English
- Personalization: Yes — active account; practical steps; offer to escalate
- Usefulness: High
- Verdict: SUCCESS

---

# Findings
- **Routing quality**: Correct across all eight cases. Deterministic override in `RouterAgent.route_query()` ensures explicit ticket requests route to Support.
- **Language consistency**: Correct; EN inputs → EN outputs; PT inputs → PT outputs.
- **RAG scaffolding**: No leakage of literal "CONTEXT"; sources label localized (EN: "Sources", PT: "Fontes").
- **Formatting**: Clean bullets, no mid-word breaks (router splitter fixed).
- **Support UX**: Data-driven summaries with clear next steps, handoff option. Ticket confirmations localized; priority hidden from user messages.
- **Ticket triage**: LLM-based triage converts free-form problem statements into structured subject/description; logs include triage summary at INFO level.

# Minor UX Observations
- **Inline "Source:" phrases in body**: Occasionally present in knowledge answers even though sources are listed in the footer. Cosmetic; can be removed with prompt guidance/sanitizer.
- **Occasional PT terms in EN answers**: Words like "faturamento" can slip into EN responses when quoting materials; generally understandable but could be normalized.

# Recommendations
- **Forbid inline "Source:" in body**: Adjust `agents/knowledge_agent.py` prompt to request no inline "Source:" lines; keep only the footer citations. Optional sanitizer to strip residual patterns.
- **Optional language polish**: In EN answers, translate PT terms unless they are brand/product names.
- **Expose debug fields behind flag**: Optionally include `lang` and `triage` in API responses when a `debug=1` query param is present (kept out of default user responses).
- **Tests**:
  - Router splitter regression tests to avoid mid-word splits.
  - E2E assertions that `answer` contains no literal "CONTEXT" and that source label matches language.

# Overall Assessment
- **Quality**: High. All 8 test queries returned contextually correct, readable, and language-consistent answers with proper routing and helpful UX.
- **Production readiness**: Good for a mock-backed stack. Observability improved via triage logging and health checks; further polish is optional.
