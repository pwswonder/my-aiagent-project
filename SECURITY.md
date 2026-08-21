# Security policy

Do not report credentials in a public issue. Rotate the credential first, then use GitHub's
private vulnerability reporting or contact the repository owner directly.

## Required incident response for the former `backend/.env`

The historical PostgreSQL credential must be treated as compromised even after the file is
deleted. Revoke it at the database provider, create a new least-privilege credential, store it in
the deployment secret manager, and review provider access logs. Never reuse the exposed password.

After rotation, run `scripts/purge_sensitive_history.sh` from a fresh mirror clone and coordinate
the force-push with every contributor. Existing clones and forks can still contain the old data.

## Generated-code boundary

Generated code runs only in the sandbox image with no network, no secrets, a read-only root,
dropped capabilities, `no-new-privileges`, and CPU/RAM/PID/time limits. LLM-written code is limited
to one AST-validated `torch.nn.Module`; it cannot freely rewrite a project.
