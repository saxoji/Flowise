/**
 * Database connection-pool sizing and (opt-in) timeout helpers.
 *
 * Kept in a dependency-light module (no entities/migrations/typeorm imports) so the pure
 * env-derivation logic is unit-testable in isolation; consumed by DataSource.ts.
 *
 * Why this exists
 * ---------------
 * In QUEUE mode a worker process runs up to WORKER_CONCURRENCY jobs concurrently and each job
 * may briefly check out a DB connection for its queries. The historical pg-pool default of 10
 * starves a worker whose concurrency is higher (e.g. WORKER_CONCURRENCY=20), so jobs queue on
 * connection acquisition. We size the pool to the process role.
 *
 * Side-effect safety
 * ------------------
 * - Web processes (no WORKER_CONCURRENCY) keep the historical default of 10 — unchanged.
 * - connectionTimeoutMillis and statement_timeout are OPT-IN (return undefined unless their env
 *   var is set), so default behavior is byte-for-byte the previous behavior. connectionTimeout
 *   only bounds how long acquiring a pooled connection waits; it NEVER interrupts a running
 *   query, custom-tool HTTP call, or LLM completion (those do not hold a DB connection while
 *   they run). statement_timeout is left off by default because pg sends it via the startup
 *   `options` parameter, which PgBouncer in transaction pooling rejects unless it is added to
 *   `ignore_startup_parameters`.
 */

const DEFAULT_POOL_SIZE = 10 // historical pg-pool default; keeps web processes unchanged
const MAX_AUTO_POOL_SIZE = 50 // ceiling for concurrency-derived auto sizing
const POOL_HEADROOM = 5 // spare connections beyond worker concurrency (migrations, misc queries)
const MAX_AUTO_SIZE_CONCURRENCY = 100 // above this WORKER_CONCURRENCY is treated as unbounded sentinel

const parsePositiveInt = (value: string | undefined): number | undefined => {
    if (value === undefined) return undefined
    const parsed = parseInt(value, 10)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

/**
 * Pool size (pg-pool `max`). Precedence:
 *  1. explicit DATABASE_POOL_SIZE
 *  2. WORKER_CONCURRENCY + headroom (workers only), capped — ignores the unbounded 100000 sentinel
 *  3. historical default of 10 (web processes)
 */
export const getDatabasePoolSize = (env: NodeJS.ProcessEnv = process.env): number => {
    const explicit = parsePositiveInt(env.DATABASE_POOL_SIZE)
    if (explicit !== undefined) return explicit

    const concurrency = parsePositiveInt(env.WORKER_CONCURRENCY)
    if (concurrency !== undefined && concurrency <= MAX_AUTO_SIZE_CONCURRENCY) {
        return Math.min(concurrency + POOL_HEADROOM, MAX_AUTO_POOL_SIZE)
    }
    return DEFAULT_POOL_SIZE
}

/**
 * pg-pool `connectionTimeoutMillis`. Opt-in via DATABASE_CONNECTION_TIMEOUT (0 = wait forever).
 * Returns undefined when unset so DataSource omits the option entirely (unchanged behavior).
 */
export const getDatabaseConnectionTimeoutMillis = (env: NodeJS.ProcessEnv = process.env): number | undefined => {
    const value = env.DATABASE_CONNECTION_TIMEOUT
    if (value === undefined) return undefined
    const parsed = parseInt(value, 10)
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined
}

/**
 * Postgres `statement_timeout` (ms). Opt-in via DATABASE_STATEMENT_TIMEOUT. Off by default to
 * avoid aborting legitimately long operations and to avoid PgBouncer startup-parameter issues.
 */
export const getDatabaseStatementTimeoutMillis = (env: NodeJS.ProcessEnv = process.env): number | undefined => {
    return parsePositiveInt(env.DATABASE_STATEMENT_TIMEOUT)
}
