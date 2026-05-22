import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { BaseMessage } from '@langchain/core/messages'
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { IMultiModalOption, IVisionChatModal } from '../../../src'

type OpenRouterFields = ChatOpenAIFields & {
    model?: string
    openAIApiKey?: unknown
}

type OpenRouterAttempt = {
    modelName: string
    apiKey?: unknown
}

type AttemptChatModel = Pick<LangchainChatOpenAI, '_generate' | '_streamResponseChunks'>

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

const createNormalizedFields = (fields?: OpenRouterFields) => {
    const modelCandidates = splitCommaSeparatedValues(getConfiguredModelName(fields))
    const cacheModelName = modelCandidates[0] ?? getConfiguredModelName(fields)
    const configuredApiKey = getConfiguredApiKey(fields)
    const apiKeyCandidates = splitCommaSeparatedValues(configuredApiKey)
    const firstApiKey = apiKeyCandidates[0] ?? configuredApiKey

    return {
        fields: getFieldsForAttempt(fields, cacheModelName, firstApiKey),
        modelCandidates: modelCandidates.length ? modelCandidates : cacheModelName ? [cacheModelName] : [],
        apiKeyCandidates,
        configuredApiKey,
        cacheModelName
    }
}

export class ChatOpenRouter extends LangchainChatOpenAI implements IVisionChatModal {
    configuredModel: string
    configuredMaxToken?: number
    multiModalOption: IMultiModalOption
    id: string
    private readonly baseFields: OpenRouterFields
    private readonly modelCandidates: string[]
    private readonly apiKeyCandidates: string[]
    private readonly configuredApiKey: unknown
    private readonly cacheModelName: string

    constructor(id: string, fields?: ChatOpenAIFields) {
        const normalized = createNormalizedFields(fields as OpenRouterFields | undefined)
        super(normalized.fields)
        this.id = id
        this.baseFields = normalized.fields
        this.modelCandidates = normalized.modelCandidates
        this.apiKeyCandidates = normalized.apiKeyCandidates
        this.configuredApiKey = normalized.configuredApiKey
        this.cacheModelName = normalized.cacheModelName
        this.configuredModel = this.cacheModelName
        this.configuredMaxToken = fields?.maxTokens
    }

    setMultiModalOption(multiModalOption: IMultiModalOption): void {
        this.multiModalOption = multiModalOption
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

        for (const attempt of attempts) {
            try {
                const result = await this.createAttemptModel(attempt, attempts.length > 1)._generate(messages, options, runManager)
                this.annotateResultWithSelectedModel(result, attempt.modelName)
                return result
            } catch (error) {
                if (this.isAbortError(error, options)) throw error
                lastError = error
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

        for (const attempt of attempts) {
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
                lastError = error
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

    protected shuffleAttempts<T>(attempts: T[]): T[] {
        const shuffled = [...attempts]

        for (let index = shuffled.length - 1; index > 0; index -= 1) {
            const randomIndex = Math.floor(Math.random() * (index + 1))
            ;[shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]]
        }

        return shuffled
    }

    private getAttemptSequence(): OpenRouterAttempt[] {
        const models = this.modelCandidates.length ? this.modelCandidates : this.cacheModelName ? [this.cacheModelName] : []
        const apiKeys = this.apiKeyCandidates.length ? this.apiKeyCandidates : [this.configuredApiKey]
        const attempts: OpenRouterAttempt[] = []

        for (const modelName of models) {
            for (const apiKey of apiKeys) {
                attempts.push({ modelName, apiKey })
            }
        }

        return this.shuffleAttempts(attempts.length ? attempts : [{ modelName: this.cacheModelName, apiKey: this.configuredApiKey }])
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
