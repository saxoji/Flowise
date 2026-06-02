import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { BaseMessage } from '@langchain/core/messages'
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { IMultiModalOption, IVisionChatModal } from '../../../src'

type OpenRouterFields = ChatOpenAIFields & {
    model?: string
    openAIApiKey?: unknown
    roundRobinScope?: string
    roundRobinSessionId?: string
}

type OpenRouterAttempt = {
    modelName: string
    apiKey?: unknown
    apiKeyIndex: number
}

type AttemptChatModel = Pick<LangchainChatOpenAI, '_generate' | '_streamResponseChunks'>

type RoundRobinState = {
    nextIndex: number
    lastUsedAt: number
}

type SessionAttemptState = {
    attemptIndex: number
    lastUsedAt: number
}

type ProviderErrorFacts = {
    statuses: number[]
    codes: string[]
    types: string[]
    names: string[]
    messages: string[]
}

const ROUND_ROBIN_STATE_TTL_MS = 24 * 60 * 60 * 1000
const ROUND_ROBIN_STATE_PRUNE_INTERVAL_MS = 60 * 60 * 1000
const roundRobinAssignmentStates = new Map<string, RoundRobinState>()
const roundRobinSessionStates = new Map<string, SessionAttemptState>()
let lastRoundRobinStatePrune = 0
const TRANSIENT_PROVIDER_STATUS_CODES = new Set([408, 429, 500, 502, 503, 504, 529])
const TRANSIENT_PROVIDER_ERROR_CODES = new Set([
    'ECONNABORTED',
    'ECONNREFUSED',
    'ECONNRESET',
    'EAI_AGAIN',
    'ENOTFOUND',
    'ETIMEDOUT',
    'UND_ERR_CONNECT_TIMEOUT',
    'UND_ERR_HEADERS_TIMEOUT',
    'UND_ERR_SOCKET'
])
const TRANSIENT_PROVIDER_ERROR_TYPES = new Set([
    'api_connection_error',
    'overloaded_error',
    'rate_limit_error',
    'server_error',
    'timeout_error'
])
const TRANSIENT_PROVIDER_ERROR_NAMES = new Set(['APIConnectionError', 'APIConnectionTimeoutError', 'TimeoutError'])

const toStatusCode = (value: unknown): number | undefined => {
    if (typeof value === 'number' && Number.isInteger(value)) return value
    if (typeof value === 'string' && /^\d{3}$/.test(value.trim())) return parseInt(value.trim(), 10)
    return undefined
}

const addStringFact = (target: string[], value: unknown): void => {
    if (typeof value === 'string' && value.trim()) target.push(value.trim())
}

const isRecord = (value: unknown): value is Record<string, any> => typeof value === 'object' && value !== null

const collectProviderErrorFacts = (value: unknown, facts: ProviderErrorFacts, seen = new Set<unknown>(), depth = 0): ProviderErrorFacts => {
    if (!value || seen.has(value) || depth > 5) return facts

    if (value instanceof Error) {
        addStringFact(facts.names, value.name)
        addStringFact(facts.messages, value.message)
    }

    if (!isRecord(value)) return facts
    seen.add(value)

    for (const statusField of ['status', 'statusCode']) {
        const status = toStatusCode(value[statusField])
        if (status !== undefined) facts.statuses.push(status)
    }

    for (const codeField of ['code', 'errno']) {
        addStringFact(facts.codes, value[codeField])
    }

    addStringFact(facts.types, value.type)
    addStringFact(facts.names, value.name)
    addStringFact(facts.messages, value.message)

    for (const childField of ['cause', 'error', 'response', 'data', 'body']) {
        collectProviderErrorFacts(value[childField], facts, seen, depth + 1)
    }

    return facts
}

const isTransientProviderErrorMessage = (message: string): boolean => {
    const normalized = message.toLowerCase()

    return (
        /\b(408|429|500|502|503|504|529)\b/.test(normalized) ||
        normalized.includes('bad gateway') ||
        normalized.includes('gateway timeout') ||
        normalized.includes('overloaded') ||
        normalized.includes('rate limit') ||
        normalized.includes('service unavailable') ||
        normalized.includes('too many requests')
    )
}

export const isTransientProviderFailure = (error: unknown): boolean => {
    const facts = collectProviderErrorFacts(error, {
        statuses: [],
        codes: [],
        types: [],
        names: [],
        messages: []
    })

    if (facts.statuses.some((status) => status >= 400 && status < 500 && !TRANSIENT_PROVIDER_STATUS_CODES.has(status))) return false
    if (facts.statuses.length) return facts.statuses.some((status) => TRANSIENT_PROVIDER_STATUS_CODES.has(status) || status >= 500)
    if (facts.codes.some((code) => TRANSIENT_PROVIDER_ERROR_CODES.has(code))) return true
    if (facts.types.some((type) => TRANSIENT_PROVIDER_ERROR_TYPES.has(type))) return true
    if (facts.names.some((name) => TRANSIENT_PROVIDER_ERROR_NAMES.has(name))) return true

    return facts.messages.some(isTransientProviderErrorMessage)
}

const isForbiddenProviderFailure = (error: unknown): boolean => {
    const facts = collectProviderErrorFacts(error, {
        statuses: [],
        codes: [],
        types: [],
        names: [],
        messages: []
    })

    if (facts.statuses.some((status) => status === 403)) return true

    return facts.messages.some((message) => {
        const normalized = message.trim().toLowerCase()
        return normalized === 'forbidden' || normalized.includes('403 forbidden')
    })
}

export const isProviderFallbackEligibleFailure = (error: unknown): boolean =>
    isTransientProviderFailure(error) || isForbiddenProviderFailure(error)

const splitCommaSeparatedValues = (value: unknown): string[] => {
    if (typeof value !== 'string') return []

    return Array.from(
        new Set(
            value
                .split(',')
                .map((item) => item.trim())
                .filter(Boolean)
        )
    )
}

const getConfiguredModelName = (fields?: OpenRouterFields): string => {
    if (!fields) return ''
    return fields.modelName ?? fields.model ?? ''
}

const getConfiguredApiKey = (fields?: OpenRouterFields): unknown => {
    if (!fields) return undefined
    return fields.apiKey ?? fields.openAIApiKey ?? fields.configuration?.apiKey
}

const getFieldsForAttempt = (fields: OpenRouterFields | undefined, modelName: string, apiKey: unknown): OpenRouterFields => {
    const nextFields: OpenRouterFields = { ...(fields ?? {}) }

    if (modelName) {
        nextFields.modelName = modelName
        nextFields.model = modelName
    }

    if (apiKey != null) {
        nextFields.apiKey = apiKey as ChatOpenAIFields['apiKey']
        nextFields.openAIApiKey = apiKey as OpenRouterFields['openAIApiKey']
    }

    if (fields?.configuration) {
        nextFields.configuration = { ...fields.configuration }
        if (apiKey != null && Object.prototype.hasOwnProperty.call(fields.configuration, 'apiKey')) {
            nextFields.configuration.apiKey = apiKey as any
        }
    }

    return nextFields
}

const cloneOpenRouterFields = (fields?: OpenRouterFields): OpenRouterFields | undefined => {
    if (!fields) return undefined

    return {
        ...fields,
        configuration: fields.configuration ? { ...fields.configuration } : undefined
    }
}

const createNormalizedFields = (fields?: OpenRouterFields) => {
    const { roundRobinScope, roundRobinSessionId, ...chatFields } = fields ?? {}
    const sourceFields = chatFields as OpenRouterFields
    const modelCandidates = splitCommaSeparatedValues(getConfiguredModelName(sourceFields))
    const cacheModelName = modelCandidates[0] ?? getConfiguredModelName(sourceFields)
    const configuredApiKey = getConfiguredApiKey(sourceFields)
    const apiKeyCandidates = splitCommaSeparatedValues(configuredApiKey)
    const firstApiKey = apiKeyCandidates[0] ?? configuredApiKey

    return {
        fields: getFieldsForAttempt(sourceFields, cacheModelName, firstApiKey),
        modelCandidates: modelCandidates.length ? modelCandidates : cacheModelName ? [cacheModelName] : [],
        apiKeyCandidates,
        configuredApiKey,
        cacheModelName,
        roundRobinScope,
        roundRobinSessionId
    }
}

export class ChatOpenRouter extends LangchainChatOpenAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    id: string
    private readonly originalFields?: OpenRouterFields
    private readonly baseFields: OpenRouterFields
    private readonly modelCandidates: string[]
    private readonly apiKeyCandidates: string[]
    private readonly configuredApiKey: unknown
    private readonly cacheModelName: string
    private readonly roundRobinScope?: string
    private readonly roundRobinSessionId?: string

    constructor(id: string, fields?: OpenRouterFields) {
        const originalFieldsSnapshot = cloneOpenRouterFields(fields)
        const normalized = createNormalizedFields(fields)
        super(normalized.fields)
        this.id = id
        this.originalFields = originalFieldsSnapshot
        this.baseFields = normalized.fields
        this.modelCandidates = normalized.modelCandidates
        this.apiKeyCandidates = normalized.apiKeyCandidates
        this.configuredApiKey = normalized.configuredApiKey
        this.cacheModelName = normalized.cacheModelName
        this.roundRobinScope = normalized.roundRobinScope
        this.roundRobinSessionId = normalized.roundRobinSessionId
        this.configuredModel = this.cacheModelName
        this.configuredMaxToken = fields?.maxTokens
    }

    setMultiModalOption(multiModalOption: IMultiModalOption): void {
        this.multiModalOption = multiModalOption
    }

    withConfig(config: Parameters<LangchainChatOpenAI['withConfig']>[0]): ReturnType<LangchainChatOpenAI['withConfig']> {
        const model = new ChatOpenRouter(this.id, this.originalFields)
        ;(model as any).defaultOptions = {
            ...((this as any).defaultOptions ?? {}),
            ...(config ?? {})
        }

        if (this.multiModalOption) model.setMultiModalOption(this.multiModalOption)

        return model as ReturnType<LangchainChatOpenAI['withConfig']>
    }

    _identifyingParams(): ReturnType<LangchainChatOpenAI['_identifyingParams']> {
        const params = { ...super._identifyingParams() } as ReturnType<LangchainChatOpenAI['_identifyingParams']> & Record<string, any>

        if (this.cacheModelName) {
            params.model_name = this.cacheModelName
            params.model = this.cacheModelName
        }

        if (this.apiKeyCandidates.length > 1) {
            delete params.apiKey
            delete params.openAIApiKey
            delete params.openai_api_key
        }

        return params
    }

    async _generate(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        const attempts = this.getAttemptSequence()
        if (attempts.length === 1) return super._generate(messages, options, runManager)

        let lastError: unknown
        const attemptedAttemptIndexes = new Set<number>()
        const failedApiKeyIndexes = new Set<number>()
        const failedModelNames = new Set<string>()

        while (attemptedAttemptIndexes.size < attempts.length) {
            const nextAttempt = this.getNextAttempt(attempts, attemptedAttemptIndexes, failedApiKeyIndexes, failedModelNames)
            if (!nextAttempt) break

            const { attempt, index } = nextAttempt
            attemptedAttemptIndexes.add(index)

            try {
                const result = await this.createAttemptModel(attempt, attempts.length > 1)._generate(messages, options, runManager)
                this.annotateResultWithSelectedModel(result, attempt.modelName)
                return result
            } catch (error) {
                if (this.isAbortError(error, options)) throw error
                if (!isProviderFallbackEligibleFailure(error)) throw error
                lastError = error
                this.trackFailedAttempt(attempt, failedApiKeyIndexes, failedModelNames)
            }
        }

        throw this.createAllAttemptsFailedError(lastError, attempts)
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        const attempts = this.getAttemptSequence()
        if (attempts.length === 1) {
            yield* super._streamResponseChunks(messages, options, runManager)
            return
        }

        let lastError: unknown
        const attemptedAttemptIndexes = new Set<number>()
        const failedApiKeyIndexes = new Set<number>()
        const failedModelNames = new Set<string>()

        while (attemptedAttemptIndexes.size < attempts.length) {
            const nextAttempt = this.getNextAttempt(attempts, attemptedAttemptIndexes, failedApiKeyIndexes, failedModelNames)
            if (!nextAttempt) break

            const { attempt, index } = nextAttempt
            attemptedAttemptIndexes.add(index)

            let hasYieldedChunk = false

            try {
                const stream = this.createAttemptModel(attempt, attempts.length > 1)._streamResponseChunks(messages, options, runManager)

                for await (const chunk of stream) {
                    hasYieldedChunk = true
                    this.annotateChunkWithSelectedModel(chunk, attempt.modelName)
                    yield chunk
                }

                return
            } catch (error) {
                if (hasYieldedChunk || this.isAbortError(error, options)) throw error
                if (!isProviderFallbackEligibleFailure(error)) throw error
                lastError = error
                this.trackFailedAttempt(attempt, failedApiKeyIndexes, failedModelNames)
            }
        }

        throw this.createAllAttemptsFailedError(lastError, attempts)
    }

    protected createAttemptModel(attempt: OpenRouterAttempt, disableInternalRetries: boolean): AttemptChatModel {
        const fields = getFieldsForAttempt(this.baseFields, attempt.modelName, attempt.apiKey)
        if (disableInternalRetries && fields.maxRetries === undefined) fields.maxRetries = 0

        const model = new LangchainChatOpenAI(fields)
        ;(model as any).defaultOptions = (this as any).defaultOptions
        return model
    }

    private getAttemptSequence(): OpenRouterAttempt[] {
        const attempts = this.getAllAttempts()
        if (attempts.length <= 1) return attempts

        const assignmentKey = this.getRoundRobinAssignmentStateKey(attempts)
        const sessionKey = this.getRoundRobinSessionStateKey(assignmentKey)
        this.pruneRoundRobinStates()

        const now = Date.now()
        const sessionState = roundRobinSessionStates.get(sessionKey)
        let startIndex = sessionState ? this.normalizeAttemptIndex(sessionState.attemptIndex, attempts.length) : undefined

        if (startIndex === undefined) {
            const assignmentState = roundRobinAssignmentStates.get(assignmentKey)
            startIndex = assignmentState ? this.normalizeAttemptIndex(assignmentState.nextIndex, attempts.length) : 0

            roundRobinAssignmentStates.set(assignmentKey, {
                nextIndex: this.normalizeAttemptIndex(startIndex + 1, attempts.length),
                lastUsedAt: now
            })
        }

        roundRobinSessionStates.set(sessionKey, {
            attemptIndex: startIndex,
            lastUsedAt: now
        })

        return [...attempts.slice(startIndex), ...attempts.slice(0, startIndex)]
    }

    private getAllAttempts(): OpenRouterAttempt[] {
        const models = this.modelCandidates.length ? this.modelCandidates : this.cacheModelName ? [this.cacheModelName] : []
        const apiKeys = this.apiKeyCandidates.length ? this.apiKeyCandidates : [this.configuredApiKey]
        const attempts: OpenRouterAttempt[] = []

        for (const modelName of models) {
            apiKeys.forEach((apiKey, apiKeyIndex) => {
                attempts.push({ modelName, apiKey, apiKeyIndex })
            })
        }

        return attempts.length ? attempts : [{ modelName: this.cacheModelName, apiKey: this.configuredApiKey, apiKeyIndex: 0 }]
    }

    private getRoundRobinAssignmentStateKey(attempts: OpenRouterAttempt[]): string {
        const models = this.modelCandidates.length ? this.modelCandidates : this.cacheModelName ? [this.cacheModelName] : []
        const apiKeys = this.apiKeyCandidates.length ? this.apiKeyCandidates : [this.configuredApiKey]
        return [this.roundRobinScope || this.id, models.join(','), `keys:${apiKeys.length}`, `attempts:${attempts.length}`].join('|')
    }

    private getRoundRobinSessionStateKey(assignmentKey: string): string {
        return `${assignmentKey}|session:${this.roundRobinSessionId || 'default'}`
    }

    private normalizeAttemptIndex(index: number, attemptsLength: number): number {
        if (attemptsLength <= 0) return 0
        return ((index % attemptsLength) + attemptsLength) % attemptsLength
    }

    private pruneRoundRobinStates(): void {
        const now = Date.now()
        if (now - lastRoundRobinStatePrune < ROUND_ROBIN_STATE_PRUNE_INTERVAL_MS) return

        lastRoundRobinStatePrune = now

        for (const [key, state] of roundRobinAssignmentStates) {
            if (now - state.lastUsedAt > ROUND_ROBIN_STATE_TTL_MS) roundRobinAssignmentStates.delete(key)
        }

        for (const [key, state] of roundRobinSessionStates) {
            if (now - state.lastUsedAt > ROUND_ROBIN_STATE_TTL_MS) roundRobinSessionStates.delete(key)
        }
    }

    private getNextAttempt(
        attempts: OpenRouterAttempt[],
        attemptedAttemptIndexes: Set<number>,
        failedApiKeyIndexes: Set<number>,
        failedModelNames: Set<string>
    ): { attempt: OpenRouterAttempt; index: number } | undefined {
        const hasMultipleModels = this.modelCandidates.length > 1
        const hasMultipleApiKeys = this.apiKeyCandidates.length > 1
        const predicates: Array<(_attempt: OpenRouterAttempt) => boolean> = []

        if (hasMultipleModels || hasMultipleApiKeys) {
            predicates.push(
                (attempt) =>
                    (!hasMultipleModels || !failedModelNames.has(attempt.modelName)) &&
                    (!hasMultipleApiKeys || !failedApiKeyIndexes.has(attempt.apiKeyIndex))
            )
        }

        if (hasMultipleApiKeys) {
            predicates.push((attempt) => !failedApiKeyIndexes.has(attempt.apiKeyIndex))
        }

        if (hasMultipleModels) {
            predicates.push((attempt) => !failedModelNames.has(attempt.modelName))
        }

        predicates.push(() => true)

        for (const predicate of predicates) {
            for (let index = 0; index < attempts.length; index += 1) {
                if (attemptedAttemptIndexes.has(index)) continue

                const attempt = attempts[index]
                if (predicate(attempt)) return { attempt, index }
            }
        }

        return undefined
    }

    private trackFailedAttempt(attempt: OpenRouterAttempt, failedApiKeyIndexes: Set<number>, failedModelNames: Set<string>): void {
        if (this.apiKeyCandidates.length > 1) failedApiKeyIndexes.add(attempt.apiKeyIndex)
        if (this.modelCandidates.length > 1) failedModelNames.add(attempt.modelName)
    }

    private annotateResultWithSelectedModel(result: ChatResult, modelName: string): void {
        for (const generation of result.generations) {
            generation.generationInfo = {
                model_name: modelName,
                ...generation.generationInfo
            }

            generation.message.response_metadata = {
                model: modelName,
                model_name: modelName,
                ...generation.message.response_metadata
            }
        }
    }

    private annotateChunkWithSelectedModel(chunk: ChatGenerationChunk, modelName: string): void {
        chunk.generationInfo = {
            model_name: modelName,
            ...chunk.generationInfo
        }

        chunk.message.response_metadata = {
            model: modelName,
            model_name: modelName,
            ...chunk.message.response_metadata
        }
    }

    private isAbortError(error: unknown, options?: this['ParsedCallOptions']): boolean {
        if (options?.signal?.aborted) return true
        if (!(error instanceof Error)) return false
        return error.name === 'AbortError' || error.name === 'ModelAbortError' || error.message === 'AbortError'
    }

    private createAllAttemptsFailedError(lastError: unknown, attempts: OpenRouterAttempt[]): Error {
        const attemptedModels = Array.from(new Set(attempts.map((attempt) => attempt.modelName).filter(Boolean))).join(', ')
        const lastMessage = lastError instanceof Error ? lastError.message : String(lastError)
        const error = new Error(
            `ChatOpenRouter failed for all fallback attempts${attemptedModels ? ` (${attemptedModels})` : ''}: ${lastMessage}`
        )
        ;(error as any).cause = lastError
        return error
    }
}
