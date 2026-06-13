import { Queue, Worker, Job, QueueEvents, RedisOptions, KeepJobs } from 'bullmq'
import { v4 as uuidv4 } from 'uuid'
import logger from '../utils/logger'

const QUEUE_REDIS_EVENT_STREAM_MAX_LEN = process.env.QUEUE_REDIS_EVENT_STREAM_MAX_LEN
    ? parseInt(process.env.QUEUE_REDIS_EVENT_STREAM_MAX_LEN)
    : 10000
const WORKER_CONCURRENCY = process.env.WORKER_CONCURRENCY ? parseInt(process.env.WORKER_CONCURRENCY) : 100000
const REMOVE_ON_AGE = process.env.REMOVE_ON_AGE ? parseInt(process.env.REMOVE_ON_AGE) : -1
const REMOVE_ON_COUNT = process.env.REMOVE_ON_COUNT ? parseInt(process.env.REMOVE_ON_COUNT) : -1

const stringifyError = (error: unknown): string => {
    if (error instanceof Error) {
        const errorRecord = error as Error & Record<string, unknown>
        const details = [
            `name=${error.name}`,
            `message=${error.message}`,
            'code' in errorRecord ? `code=${String(errorRecord.code)}` : undefined,
            'errno' in errorRecord ? `errno=${String(errorRecord.errno)}` : undefined,
            'syscall' in errorRecord ? `syscall=${String(errorRecord.syscall)}` : undefined,
            'address' in errorRecord ? `address=${String(errorRecord.address)}` : undefined,
            'port' in errorRecord ? `port=${String(errorRecord.port)}` : undefined,
            'command' in errorRecord ? `command=${String(errorRecord.command)}` : undefined,
            'status' in errorRecord ? `status=${String(errorRecord.status)}` : undefined,
            'statusCode' in errorRecord ? `statusCode=${String(errorRecord.statusCode)}` : undefined
        ].filter(Boolean)

        return `${details.join(' ')}${error.stack ? `\n${error.stack}` : ''}`
    }

    if (typeof error === 'string') return error

    try {
        return JSON.stringify(error)
    } catch {
        return String(error)
    }
}

export abstract class BaseQueue {
    protected queue: Queue
    protected queueEvents: QueueEvents
    protected connection: RedisOptions
    private worker: Worker

    constructor(queueName: string, connection: RedisOptions) {
        this.connection = connection
        this.queue = new Queue(queueName, {
            connection: this.connection,
            streams: { events: { maxLen: QUEUE_REDIS_EVENT_STREAM_MAX_LEN } }
        })
        this.queueEvents = new QueueEvents(queueName, { connection: this.connection })

        // The QueueEvents stream connection (used by job.waitUntilFinished) can drop when Redis
        // restarts. Without an 'error' listener these surface as unhandled errors; logging them
        // also gives visibility into the windows where completion events may be missed (the
        // resilient wait in buildChatflow recovers from those by polling job state).
        this.queueEvents.on('error', (err) => {
            logger.error(`[BaseQueue] QueueEvents error for queue "${queueName}": ${stringifyError(err)}`)
        })
    }

    abstract processJob(data: any): Promise<any>

    abstract getQueueName(): string

    abstract getQueue(): Queue

    public getWorker(): Worker {
        return this.worker
    }

    public async addJob(jobData: any): Promise<Job> {
        const jobId = jobData.id || uuidv4()

        let removeOnFail: number | boolean | KeepJobs | undefined = true
        let removeOnComplete: number | boolean | KeepJobs | undefined = undefined

        // Only override removal options if age or count is specified
        if (REMOVE_ON_AGE !== -1 || REMOVE_ON_COUNT !== -1) {
            const keepJobObj: KeepJobs = {}
            if (REMOVE_ON_AGE !== -1) {
                keepJobObj.age = REMOVE_ON_AGE
            }
            if (REMOVE_ON_COUNT !== -1) {
                keepJobObj.count = REMOVE_ON_COUNT
            }
            removeOnFail = keepJobObj
            removeOnComplete = keepJobObj
        }

        return await this.queue.add(jobId, jobData, { removeOnFail, removeOnComplete })
    }

    public createWorker(concurrency: number = WORKER_CONCURRENCY): Worker {
        try {
            this.worker = new Worker(
                this.queue.name,
                async (job: Job) => {
                    const start = new Date().getTime()
                    logger.info(`[BaseQueue] Processing job ${job.id} in ${this.queue.name} at ${new Date().toISOString()}`)
                    try {
                        const result = await this.processJob(job.data)
                        const end = new Date().getTime()
                        logger.info(
                            `[BaseQueue] Completed job ${job.id} in ${this.queue.name} at ${new Date().toISOString()} (${end - start}ms)`
                        )
                        return result
                    } catch (error) {
                        const end = new Date().getTime()
                        logger.error(
                            `[BaseQueue] Job ${job.id} failed in ${this.queue.name} at ${new Date().toISOString()} (${
                                end - start
                            }ms): ${stringifyError(error)}`
                        )
                        throw error
                    }
                },
                {
                    connection: this.connection,
                    concurrency
                }
            )

            // Add error listeners to the worker
            this.worker.on('error', (err) => {
                logger.error(`[BaseQueue] Worker error for queue "${this.queue.name}": ${stringifyError(err)}`)
            })

            this.worker.on('closed', () => {
                logger.info(`[BaseQueue] Worker closed for queue "${this.queue.name}"`)
            })

            this.worker.on('failed', (job, err) => {
                logger.error(`[BaseQueue] Worker job ${job?.id} failed in queue "${this.queue.name}": ${stringifyError(err)}`)
            })

            this.worker.on('stalled', (jobId) => {
                logger.warn(`[BaseQueue] Worker job ${jobId} stalled in queue "${this.queue.name}"`)
            })

            this.worker.on('ioredis:close', () => {
                logger.warn(`[BaseQueue] Worker Redis connection closed for queue "${this.queue.name}"`)
            })

            logger.info(`[BaseQueue] Worker created successfully for queue "${this.queue.name}"`)
            return this.worker
        } catch (error) {
            logger.error(`[BaseQueue] Failed to create worker for queue "${this.queue.name}":`, { error })
            throw error
        }
    }

    public async getJobs(): Promise<Job[]> {
        return await this.queue.getJobs()
    }

    public async getJobCounts(): Promise<{ [index: string]: number }> {
        return await this.queue.getJobCounts()
    }

    public async getJobByName(jobName: string): Promise<Job> {
        const jobs = await this.queue.getJobs()
        const job = jobs.find((job) => job.name === jobName)
        if (!job) throw new Error(`Job name ${jobName} not found`)
        return job
    }

    public getQueueEvents(): QueueEvents {
        return this.queueEvents
    }

    public async clearQueue(): Promise<void> {
        await this.queue.obliterate({ force: true })
    }
}
