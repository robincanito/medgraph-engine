# Security Policy

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting.** Open
[the Security tab](https://github.com/robincanito/medgraph-engine/security/advisories/new) and file
a draft advisory. That channel is private between you and the maintainers until a fix is published,
and it is the only one we monitor for security issues — please do **not** open a public issue, a
pull request, or a discussion for something exploitable.

If the button is not available to you, open a regular issue saying only *"I would like to report a
security issue privately"* — no details — and a maintainer will enable the report for you.

### What to include

- what the problem is, and what an attacker gets out of it;
- the smallest way to reproduce it (a snippet, a request, a malformed document);
- the commit or tag you tested;
- how you would fix it, if you have an opinion.

### What to expect

- **acknowledgement:** within 7 days;
- **assessment:** we say whether we can reproduce it, and what we think the impact is;
- **fix:** in the advisory thread, with a patched release and credit to you unless you ask us not
  to. Coordinated disclosure: we publish the advisory when the fix is out.

This is a small project maintained in someone's spare time. Those are honest intentions, not an SLA.

## Supported versions

The **default branch** is what gets fixed. Older tags are not patched — if you run one, upgrade.

## Scope

In scope: anything in this repository — the `pipeline/` package, the root scripts, the `api/`
service (including its MCP endpoint, `POST /mcp`), the Docker Compose setup, the profiles.

Worth knowing before you report:

- **`api/` is a pre-unification snapshot and is not a running service here** (see the README,
  "Known gaps"). Findings in it are still welcome — the code is published, so it can be copied —
  but say so, because the fix may be "delete it" rather than "patch it".
- **Out of the box this is a local-development setup**, and the README's "Security notes" section
  says so: single bearer token, `localhost` CORS, a default database password in
  `docker-compose.yml`. Those are documented defaults, not vulnerabilities. A way to *bypass* the
  token, or a default that is unsafe even when the documented steps are followed, is.
- **You provide the documents and the credentials.** A report that depends on running untrusted
  input through the pipeline is in scope (that is the normal use); one that depends on us holding
  data we do not ship is not — this repository contains no content and no database.

## Secrets

No credential belongs in this tree. `.env` is gitignored, `.env.example` carries placeholders, and
`tests/test_repo.py` fails the build if a credential-shaped string appears in any tracked file. If
you find one anyway, that is a vulnerability: report it through the channel above and say which
file and which commit, so it can be rotated first and removed second.
