# The AI security stack

Securing an AI system is not one control; it is six layers, and the failures usually happen in the seams
between them rather than inside any single layer.

## Identity and Access

Manage who is allowed to use AI systems, models, and data. Controls: role-based and attribute-based access
(RBAC / ABAC), zero-trust, API authentication. This protects model APIs, internal AI assistants, and agent
access rights.

## Data Protection

Protect sensitive company data before it reaches a model. Controls: data masking, tokenization, and encryption
in transit and at rest. This protects PII, financial data, and internal files from leaking into prompts.

## Prompt and Input Security

Protect models from harmful or altered inputs. Risks addressed: prompt injection, data-extraction prompts, and
jailbreak attempts. Controls: input checks, prompt filtering, and policy enforcement.

## Governance and Compliance

Keep AI systems inside regulations and company policy. Framework coverage: GDPR, the EU AI Act, and ISO 42001.
Controls: audit records, risk categorization, and decision tracking.

## Output Validation

Check AI responses before they become actions or output. Risks addressed: hallucinated answers, compliance
issues, and unsafe responses. Controls: fact verification, policy validation, and output moderation.

## Monitoring and Observability

Track AI system behaviour in production. What it monitors: usage irregularities, response accuracy, and model
drift. Controls: behaviour tracking, audit logging, and performance monitoring.

## The seams

A prompt injection that Input Security misses becomes a data leak that Output Validation was never told to look
for. Each layer can do its job while the breach happens in the space between two of them. The tools are the
easy part; owning the transitions — who owns each handoff, what signal says it failed, and where the evidence
lives — is where AI security is won or lost.
