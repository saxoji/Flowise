import logger from '../utils/logger'

/**
 * Thrown when a job can no longer be found in the queue (e.g. it was removed by
 * `removeOnComplete` age/count limits before we could recover its result). Callers
 * may catch this to fall back to another source of truth (e.g. the persisted DB row).
 */
export class JobNotFoundError extends Error {
    constructor(public readonly jobId: string) {
        super(`Job ${jobId} not found in queue`)
        this.name = 'JobNotFoundError'
    }
}

export interface ResilientWaitOptions {
    /** Per-attempt wait before re-checking the job's actual state. Defaults to env or 30s. */
    pollTtlMs?: number
    /** Hard cap on total wait for a still-running job before giving up. Defaults to env or 25min. */
    maxTotalMs?: number
    /** Injectable clock for testing. */
    nowFn?: () => number
    /** Optional label for log lines. */
    label?: string
}

// In-code defaults so the behaviour works with NO new environment variables.
// Optional env overrides are read only if present.
const DEFAULT_POLL_TTL_MS = parseInt(process.env.QUEUE_WAIT_POLL_TTL_MS || '', 10) || 30_000
// 25 min: comfortably below the default REMOVE_ON_AGE window so the job (and its
// returnvalue) is still present when we recover, yet bounded so we never hang forever.
const DEFAULT_MAX_TOTAL_MS = parseInt(process.env.QUEUE_WAIT_MAX_TOTAL_MS || '', 10) || 1_500_000

// States that mean the job is still in flight — keep waiting, exactly as the original
// `waitUntilFinished` would have.
const IN_FLIGHT_STATES = new Set(['active', 'waiting', 'waiting-children', 'delayed', 'prioritized', 'paused'])

interface MinimalJob {
    id?: string
    waitUntilFinished: (queueEvents: any, ttl?: number) => Promise<any>
}

interface MinimalQueue {
    getJob: (id: string) => Promise<any>
}

/**
 * Wait for a BullMQ job to finish, resilient to missed completion events.
 *
 * On the happy path this resolves the moment the completion event arrives (identical to
 * `job.waitUntilFinished`). If the event is missed (e.g. the QueueEvents Redis connection
 * dropped during a restart), the per-attempt TTL expires and we inspect the job's real
 * state instead of hanging:
 *   - completed  → recover the stored return value
 *   - failed     → throw the failure reason (same as the original behaviour)
 *   - in-flight  → keep waiting (up to maxTotalMs)
 *   - gone       → throw JobNotFoundError so the caller can fall back to the DB
 */
export async function resilientWaitUntilFinished(
    queue: MinimalQueue,
    job: MinimalJob,
    queueEvents: any,
    options: ResilientWaitOptions = {}
): Promise<any> {
    const pollTtl = options.pollTtlMs ?? DEFAULT_POLL_TTL_MS
    const maxTotal = options.maxTotalMs ?? DEFAULT_MAX_TOTAL_MS
    const now = options.nowFn ?? Date.now
    const label = options.label ?? `job ${job.id}`
    const startedAt = now()

    // eslint-disable-next-line no-constant-condition
    while (true) {
        try {
            return await job.waitUntilFinished(queueEvents, pollTtl)
        } catch (err) {
            // The rejection is either a TTL timeout or a real job failure. Disambiguate by
            // reading the job's actual state rather than trusting the (event-driven) wait.
            const fresh = await queue.getJob(job.id as string).catch(() => null)
            if (!fresh) {
                throw new JobNotFoundError(job.id as string)
            }
            const state = await fresh.getState().catch(() => 'unknown')

            if (state === 'completed') {
                logger.warn(`[resilientWait] ${label}: completion event missed; recovered from job.returnvalue`)
                return fresh.returnvalue
            }
            if (state === 'failed') {
                throw new Error(fresh.failedReason || (err instanceof Error ? err.message : 'Job failed'))
            }
            if (!IN_FLIGHT_STATES.has(state)) {
                // 'unknown' or any unexpected terminal-without-result state.
                throw new JobNotFoundError(job.id as string)
            }

            // Still running — keep waiting, bounded by the hard cap.
            if (now() - startedAt >= maxTotal) {
                throw new Error(`[resilientWait] ${label}: exceeded max wait (${maxTotal}ms) while state=${state}`)
            }
            logger.debug(`[resilientWait] ${label}: still ${state}, re-waiting (poll ${pollTtl}ms)`)
        }
    }
}
