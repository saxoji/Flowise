# Deployment & Build Troubleshooting (Render)

Operational notes for building and deploying this fork on Render. Keep this file in
sync across the source repo and the deploy forks.

## Deploy topology

```
github.com/saxoji/Flowise                         ← source (develop here)
  └─ fork → Linkbricks-Horizon-AI/Flowise          ← Render service: MAIN web server
              └─ fork → Linkbricks-Horizon-AI/Flowise-Worker ← Render service: WORKER
```

-   Runtime is **QUEUE mode**: web + worker + PostgreSQL + Redis/Valkey (Singapore region).
-   **Render builds the two Linkbricks forks, not `saxoji/Flowise`.** A fix must be merged/pushed
    into the deploy forks to actually deploy; pushing to `saxoji` alone does nothing for Render.
-   A **failed Docker build does not take down the running service** — Render keeps the previous
    version live until a new build succeeds. So a broken build is safe to iterate on.

---

## Issue: `pnpm install` fails — `ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED` (flowise-embed)

### Symptom (Docker build, step `RUN pnpm install && pnpm build:docker`)

```
ERR_PNPM_GIT_DEP_PREPARE_NOT_ALLOWED  Failed to prepare git-hosted package fetched from
"https://codeload.github.com/saxoji/FlowiseChatEmbed/tar.gz/<hash>": The git-hosted package
"flowise-embed@3.0.5" needs to execute build scripts but is not in the
"onlyBuiltDependencies" allowlist.
```

### Root cause

-   **pnpm 10.26+** introduced a strict allowlist gate for **git-hosted dependencies** that run
    `prepare`/build scripts.
-   The Dockerfile installed pnpm **unpinned** (`npm install -g pnpm`), so the build pulled the
    latest 10.x (10.34.3) and the gate became active. This is **not** an upstream FlowiseAI code
    change — upstream uses the npm-published `flowise-embed` (prebuilt, no `prepare`), so it never
    hits the gate. This fork uses `flowise-embed: github:saxoji/FlowiseChatEmbed` (git-hosted →
    must be built from source → gated).
-   Only reproduces on a **cold pnpm store** (a fresh Docker build). A warm local store skips the
    `prepare` step and hides the failure.

### Key findings (validated with a cold-store reproduction)

1. A **bare package name** (`flowise-embed`) does **not** satisfy the gate for git deps — the
   resolved **tarball URL** form is required.
2. `package.json`'s `pnpm.onlyBuiltDependencies` **shadows** `pnpm-workspace.yaml`'s
   `onlyBuiltDependencies` (precedence). Adding the URL form only to the workspace file is
   ignored while the package.json name-form entry remains.
3. pnpm **self-enforces `engines.pnpm` regardless of `engine-strict`**
   (`ERR_PNPM_UNSUPPORTED_ENGINE`). Pinning below `engines.pnpm` requires relaxing that field.

### Fix applied (Option A — pin to a pre-gate pnpm; current state)

1. **Dockerfile:** `npm install -g pnpm@10.25.0` (last release before the 10.26 gate).
2. **package.json:** `engines.pnpm` `^10.26.0` → `>=10.21.0` (so the 10.25.0 pin is accepted).

Why this option: hash-free (no per-commit maintenance), restores the pre-incident behavior, and
changing the Dockerfile line busts the Docker layer cache to force a clean rebuild. Applied to
all three repos.

### Alternative (Option B — stay on modern pnpm ≥ 10.26)

1. **Remove** `onlyBuiltDependencies` from `package.json` (keep `overrides`).
2. In `pnpm-workspace.yaml`, allowlist the git dep by its **resolved tarball URL**:
    ```yaml
    onlyBuiltDependencies:
        - faiss-node
        - sqlite3
        - 'flowise-embed@https://codeload.github.com/saxoji/FlowiseChatEmbed/tar.gz/<hash>'
    ```
3. Keep the Dockerfile pinned to a `>=10.26 <11` version (e.g. `pnpm@10.34.3`) so `engines.pnpm`
   stays satisfied and the version is deterministic.

> ⚠️ `<hash>` must match `pnpm-lock.yaml`. When `saxoji/FlowiseChatEmbed` is bumped and the
> lockfile re-resolves, this hash changes — update the allowlist or the build breaks again.

### Reproduce / validate locally

The gate only fires on a cold store, so force one:

```bash
mkdir -p /tmp/embed-repro && cd /tmp/embed-repro
printf '{"name":"r","version":"1.0.0","dependencies":{"flowise-embed":"github:saxoji/FlowiseChatEmbed"}}' > package.json
printf 'packages: []\n' > pnpm-workspace.yaml
npx -y pnpm@10.34.3 install --no-frozen-lockfile --store-dir /tmp/coldstore   # reproduces the gate
npx -y pnpm@10.25.0 install --no-frozen-lockfile --store-dir /tmp/coldstore2  # passes (no gate)
```

---

## Maintenance notes

-   **Always pin pnpm** in the Dockerfile (never unpinned `npm install -g pnpm`) to avoid silent
    version drift that re-triggers gates like the one above.
-   If you intentionally upgrade pnpm past 10.26 later, switch to **Option B** and remember the
    tarball-URL hash maintenance.
-   This troubleshooting file should be kept identical in `saxoji/Flowise`,
    `Linkbricks-Horizon-AI/Flowise`, and `Linkbricks-Horizon-AI/Flowise-Worker`.
