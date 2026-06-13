import { getDatabasePoolSize, getDatabaseConnectionTimeoutMillis, getDatabaseStatementTimeoutMillis } from './databasePoolConfig'

describe('getDatabasePoolSize', () => {
    it('uses an explicit DATABASE_POOL_SIZE when provided', () => {
        expect(getDatabasePoolSize({ DATABASE_POOL_SIZE: '30' })).toBe(30)
    })

    it('auto-sizes a worker to its WORKER_CONCURRENCY plus headroom', () => {
        // worker deploy sets WORKER_CONCURRENCY=20 -> pool 25 so jobs do not starve the pool (B4)
        expect(getDatabasePoolSize({ WORKER_CONCURRENCY: '20' })).toBe(25)
    })

    it('ignores the unbounded WORKER_CONCURRENCY sentinel (100000) and keeps the default', () => {
        expect(getDatabasePoolSize({ WORKER_CONCURRENCY: '100000' })).toBe(10)
    })

    it('defaults to 10 when nothing is set (web process stays unchanged)', () => {
        expect(getDatabasePoolSize({})).toBe(10)
    })

    it('caps the concurrency-derived pool at 50', () => {
        expect(getDatabasePoolSize({ WORKER_CONCURRENCY: '80' })).toBe(50)
    })

    it('prefers an explicit DATABASE_POOL_SIZE over WORKER_CONCURRENCY', () => {
        expect(getDatabasePoolSize({ DATABASE_POOL_SIZE: '15', WORKER_CONCURRENCY: '20' })).toBe(15)
    })

    it('ignores an invalid DATABASE_POOL_SIZE', () => {
        expect(getDatabasePoolSize({ DATABASE_POOL_SIZE: 'abc' })).toBe(10)
    })
})

describe('getDatabaseConnectionTimeoutMillis', () => {
    // Opt-in only: default undefined => DataSource does not set connectionTimeoutMillis, so the
    // historical pg-pool behavior (wait indefinitely for a connection) is preserved unchanged.
    // This timeout only bounds how long acquiring a pooled connection waits; it never interrupts
    // a running query, tool call, or LLM completion.
    it('is undefined by default so connection-acquisition behavior is unchanged', () => {
        expect(getDatabaseConnectionTimeoutMillis({})).toBeUndefined()
    })

    it('returns the configured value when DATABASE_CONNECTION_TIMEOUT is set', () => {
        expect(getDatabaseConnectionTimeoutMillis({ DATABASE_CONNECTION_TIMEOUT: '30000' })).toBe(30000)
    })

    it('allows an explicit 0 (wait indefinitely)', () => {
        expect(getDatabaseConnectionTimeoutMillis({ DATABASE_CONNECTION_TIMEOUT: '0' })).toBe(0)
    })

    it('stays undefined on an invalid value', () => {
        expect(getDatabaseConnectionTimeoutMillis({ DATABASE_CONNECTION_TIMEOUT: 'x' })).toBeUndefined()
    })
})

describe('getDatabaseStatementTimeoutMillis', () => {
    it('is undefined by default (opt-in; no behavior change behind PgBouncer)', () => {
        expect(getDatabaseStatementTimeoutMillis({})).toBeUndefined()
    })

    it('returns the configured value when DATABASE_STATEMENT_TIMEOUT is set', () => {
        expect(getDatabaseStatementTimeoutMillis({ DATABASE_STATEMENT_TIMEOUT: '120000' })).toBe(120000)
    })

    it('ignores non-positive or invalid values', () => {
        expect(getDatabaseStatementTimeoutMillis({ DATABASE_STATEMENT_TIMEOUT: '0' })).toBeUndefined()
        expect(getDatabaseStatementTimeoutMillis({ DATABASE_STATEMENT_TIMEOUT: 'nope' })).toBeUndefined()
    })
})
