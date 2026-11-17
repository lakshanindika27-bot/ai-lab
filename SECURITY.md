# 🔒 Security Policy - 100% Local & Private

## Why This RAG is Secure

✅ **No Cloud APIs** - Runs on Ollama llama3.2:3b locally
✅ **No Data Leakage** - Your PDFs never leave your server
✅ **Zero API Cost** - No OpenAI bills
✅ **Private by Design** - ChromaDB local, data/ is gitignored

## Protected by GitHub Advanced Security

- Secret scanning enabled
- Dependabot alerts enabled
- Malware alerts enabled
- Private vulnerability reporting enabled

## Reporting a Vulnerability

Use GitHub Security tab > "Report a vulnerability" - privately report.

## Known issues

### chromadb 1.5.9: PYSEC-2026-311, 3813, 3814, 3815

- **Status:** No patched version available (1.5.9 is the latest release).
- **Scope:** All four advisories affect the ChromaDB *server* HTTP API
  (pre-auth/authenticated code injection via `trust_remote_code`,
  multi-tenant RBAC bypass).
- **Our usage:** The project uses ChromaDB only in embedded mode via
  `chromadb.PersistentClient(path="chroma_db")`. No Chroma server is
  started (no `HttpClient`, no `chroma run`), so the vulnerable
  endpoints are not exposed.
- **Mitigations:** Do not run or expose a Chroma server; no multi-tenant
  or RBAC configuration is used; `trust_remote_code` is never enabled.
- **Review:** Re-run `pip-audit` and re-assess whenever a new chromadb
  release is published.
