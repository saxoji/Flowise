import { AnthropicInput } from '@langchain/anthropic'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { BaseMessage } from '@langchain/core/messages'
import { BaseChatModelParams } from '@langchain/core/language_models/chat_models'
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs'
import { GoogleGenerativeAIChatInput } from '@langchain/google-genai'
import { ChatOpenAIFields } from '@langchain/openai'
import { ChatXAIInput } from '@langchain/xai'
import { IMultiModalOption } from '../../../src'
import { supportsSamplingParams } from '../../../src/anthropicUtils'
import { ChatAnthropic } from '../ChatAnthropic/FlowiseChatAnthropic'
import { ChatGoogleGenerativeAI } from '../ChatGoogleGenerativeAI/FlowiseChatGoogleGenerativeAI'
import { ChatOpenAI } from '../ChatOpenAI/FlowiseChatOpenAI'
import { ChatOpenRouter, isProviderFallbackEligibleFailure } from '../ChatOpenRouter/FlowiseChatOpenRouter'
import { ChatXAI } from '../ChatXAI/FlowiseChatXAI'

export type VoraRouter2FallbackProvider = 'openai' | 'xai' | 'anthropic' | 'google'

export type VoraRouter2FallbackConfig = {
    provider: VoraRouter2FallbackProvider
    providerLabel: string
    modelName: string
    apiKey: unknown
    order: number
}

type AttemptChatModel = {
    _generate: (messages: BaseMessage[], options: any, runManager?: CallbackManagerForLLMRun) => Promise<ChatResult>
    _streamResponseChunks: (
        messages: BaseMessage[],
        options: any,
        runManager?: CallbackManagerForLLMRun
    ) => AsyncGenerator<ChatGenerationChunk>
}

type VoraRouter2Fields = ChatOpenAIFields & {
    roundRobinScope?: string
    roundRobinSessionId?: string
}

type CommonFallbackFields = {
    cache?: ChatOpenAIFields['cache']
    frequencyPenalty?: ChatOpenAIFields['frequencyPenalty']
    maxTokens?: number
    presencePenalty?: ChatOpenAIFields['presencePenalty']
    streaming?: boolean
    temperature?: number
    timeout?: number
    topP?: number
}

const cloneVoraRouter2Fields = (fields?: VoraRouter2Fields): VoraRouter2Fields | undefined => {
    if (!fields) return undefined

    return {
        ...fields,
        configuration: fields.configuration ? { ...fields.configuration } : undefined
    }
}

const cloneFallbackConfigs = (fallbackConfigs: VoraRouter2FallbackConfig[] = []): VoraRouter2FallbackConfig[] =>
    fallbackConfigs.map((fallback) => ({ ...fallback })).sort((a, b) => a.order - b.order)

const getCommonFallbackFields = (fields?: VoraRouter2Fields): CommonFallbackFields => ({
    cache: fields?.cache,
    frequencyPenalty: fields?.frequencyPenalty,
    maxTokens: fields?.maxTokens,
    presencePenalty: fields?.presencePenalty,
    streaming: fields?.streaming,
    temperature: fields?.temperature,
    timeout: fields?.timeout,
    topP: fields?.topP
})

export class ChatVoraRouter2 extends ChatOpenRouter {
    private readonly primaryFields?: VoraRouter2Fields
    private readonly fallbackConfigs: VoraRouter2FallbackConfig[]
    private readonly commonFallbackFields: CommonFallbackFields

    constructor(id: string, fields?: VoraRouter2Fields, fallbackConfigs: VoraRouter2FallbackConfig[] = []) {
        super(id, fields)
        this.primaryFields = cloneVoraRouter2Fields(fields)
        this.fallbackConfigs = cloneFallbackConfigs(fallbackConfigs)
        this.commonFallbackFields = getCommonFallbackFields(fields)
    }

    withConfig(config: Parameters<ChatOpenRouter['withConfig']>[0]): ReturnType<ChatOpenRouter['withConfig']> {
        const model = new ChatVoraRouter2(this.id, this.primaryFields, this.fallbackConfigs)
        ;(model as any).defaultOptions = {
            ...((this as any).defaultOptions ?? {}),
            ...(config ?? {})
        }

        if (this.multiModalOption) model.setMultiModalOption(this.multiModalOption)

        return model as ReturnType<ChatOpenRouter['withConfig']>
    }

    _identifyingParams(): ReturnType<ChatOpenRouter['_identifyingParams']> {
        const params = { ...super._identifyingParams() } as ReturnType<ChatOpenRouter['_identifyingParams']> & Record<string, any>

        if (this.fallbackConfigs.length) {
            params.vora_router2_fallbacks = this.fallbackConfigs
                .map((fallback) => `${fallback.order}:${fallback.provider}:${fallback.modelName}`)
                .join(',')
        }

        return params
    }

    async _generate(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
        try {
            return await super._generate(messages, options, runManager)
        } catch (primaryError) {
            if (
                !this.fallbackConfigs.length ||
                this.isAbortErrorForRouter2(primaryError, options) ||
                !isProviderFallbackEligibleFailure(primaryError)
            ) {
                throw primaryError
            }

            return await this.generateWithProviderFallbacks(messages, options, runManager, primaryError)
        }
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        let primaryHasYieldedChunk = false

        try {
            const stream = super._streamResponseChunks(messages, options, runManager)
            for await (const chunk of stream) {
                primaryHasYieldedChunk = true
                yield chunk
            }
            return
        } catch (primaryError) {
            if (
                primaryHasYieldedChunk ||
                !this.fallbackConfigs.length ||
                this.isAbortErrorForRouter2(primaryError, options) ||
                !isProviderFallbackEligibleFailure(primaryError)
            ) {
                throw primaryError
            }

            yield* this.streamWithProviderFallbacks(messages, options, runManager, primaryError)
        }
    }

    protected createFallbackModel(fallback: VoraRouter2FallbackConfig): AttemptChatModel {
        const common = this.commonFallbackFields
        let model: AttemptChatModel

        if (fallback.provider === 'openai') {
            const fields: ChatOpenAIFields = {
                apiKey: fallback.apiKey as ChatOpenAIFields['apiKey'],
                openAIApiKey: fallback.apiKey as ChatOpenAIFields['openAIApiKey'],
                modelName: fallback.modelName,
                streaming: common.streaming ?? true
            }
            if (common.cache) fields.cache = common.cache
            if (common.frequencyPenalty) fields.frequencyPenalty = common.frequencyPenalty
            if (common.maxTokens) fields.maxCompletionTokens = common.maxTokens
            if (common.presencePenalty) fields.presencePenalty = common.presencePenalty
            if (common.temperature !== undefined) fields.temperature = common.temperature
            if (common.timeout) fields.timeout = common.timeout
            if (common.topP) fields.topP = common.topP
            model = new ChatOpenAI(this.id, fields)
        } else if (fallback.provider === 'xai') {
            const fields: ChatXAIInput = {
                apiKey: fallback.apiKey as ChatXAIInput['apiKey'],
                model: fallback.modelName,
                streaming: common.streaming ?? true
            }
            if (common.cache) fields.cache = common.cache
            if (common.maxTokens) fields.maxTokens = common.maxTokens
            if (common.temperature !== undefined) fields.temperature = common.temperature
            model = new ChatXAI(this.id, fields)
        } else if (fallback.provider === 'anthropic') {
            const fields: Partial<AnthropicInput> & BaseChatModelParams & { anthropicApiKey?: string } = {
                anthropicApiKey: fallback.apiKey as string,
                modelName: fallback.modelName,
                streaming: common.streaming ?? true
            }
            if (common.cache) fields.cache = common.cache
            if (common.maxTokens) fields.maxTokens = common.maxTokens
            if (common.temperature !== undefined && supportsSamplingParams(fallback.modelName)) fields.temperature = common.temperature
            if (common.topP && supportsSamplingParams(fallback.modelName)) fields.topP = common.topP
            model = new ChatAnthropic(this.id, fields)
        } else {
            const fields: GoogleGenerativeAIChatInput = {
                apiKey: fallback.apiKey as string,
                model: fallback.modelName,
                streaming: common.streaming ?? true
            }
            if (common.cache) fields.cache = common.cache
            if (common.maxTokens) fields.maxOutputTokens = common.maxTokens
            if (common.temperature !== undefined) fields.temperature = common.temperature
            if (common.topP) fields.topP = common.topP
            model = new ChatGoogleGenerativeAI(this.id, fields)
        }

        ;(model as any).defaultOptions = (this as any).defaultOptions
        if (this.multiModalOption && 'setMultiModalOption' in model) {
            ;(model as any).setMultiModalOption(this.multiModalOption as IMultiModalOption)
        }

        return model
    }

    private async generateWithProviderFallbacks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager: CallbackManagerForLLMRun | undefined,
        primaryError: unknown
    ): Promise<ChatResult> {
        let lastError: unknown = primaryError
        const attemptedFallbacks: VoraRouter2FallbackConfig[] = []

        for (const fallback of this.fallbackConfigs) {
            attemptedFallbacks.push(fallback)
            try {
                const result = await this.createFallbackModel(fallback)._generate(messages, options, runManager)
                this.annotateResultWithSelectedFallback(result, fallback)
                return result
            } catch (error) {
                if (this.isAbortErrorForRouter2(error, options)) throw error
                if (!isProviderFallbackEligibleFailure(error)) throw error
                lastError = error
            }
        }

        throw this.createAllProviderFallbacksFailedError(primaryError, lastError, attemptedFallbacks)
    }

    private async *streamWithProviderFallbacks(
        messages: BaseMessage[],
        options: this['ParsedCallOptions'],
        runManager: CallbackManagerForLLMRun | undefined,
        primaryError: unknown
    ): AsyncGenerator<ChatGenerationChunk> {
        let lastError: unknown = primaryError
        const attemptedFallbacks: VoraRouter2FallbackConfig[] = []

        for (const fallback of this.fallbackConfigs) {
            attemptedFallbacks.push(fallback)
            let hasYieldedChunk = false

            try {
                const stream = this.createFallbackModel(fallback)._streamResponseChunks(messages, options, runManager)
                for await (const chunk of stream) {
                    hasYieldedChunk = true
                    this.annotateChunkWithSelectedFallback(chunk, fallback)
                    yield chunk
                }
                return
            } catch (error) {
                if (hasYieldedChunk || this.isAbortErrorForRouter2(error, options)) throw error
                if (!isProviderFallbackEligibleFailure(error)) throw error
                lastError = error
            }
        }

        throw this.createAllProviderFallbacksFailedError(primaryError, lastError, attemptedFallbacks)
    }

    private annotateResultWithSelectedFallback(result: ChatResult, fallback: VoraRouter2FallbackConfig): void {
        for (const generation of result.generations) {
            generation.generationInfo = {
                model_name: fallback.modelName,
                vora_router2_fallback_provider: fallback.provider,
                ...generation.generationInfo
            }

            generation.message.response_metadata = {
                model: fallback.modelName,
                model_name: fallback.modelName,
                vora_router2_fallback_provider: fallback.provider,
                ...generation.message.response_metadata
            }
        }
    }

    private annotateChunkWithSelectedFallback(chunk: ChatGenerationChunk, fallback: VoraRouter2FallbackConfig): void {
        chunk.generationInfo = {
            model_name: fallback.modelName,
            vora_router2_fallback_provider: fallback.provider,
            ...chunk.generationInfo
        }

        chunk.message.response_metadata = {
            model: fallback.modelName,
            model_name: fallback.modelName,
            vora_router2_fallback_provider: fallback.provider,
            ...chunk.message.response_metadata
        }
    }

    private isAbortErrorForRouter2(error: unknown, options?: this['ParsedCallOptions']): boolean {
        if (options?.signal?.aborted) return true
        if (!(error instanceof Error)) return false
        return error.name === 'AbortError' || error.name === 'ModelAbortError' || error.message === 'AbortError'
    }

    private createAllProviderFallbacksFailedError(
        primaryError: unknown,
        lastError: unknown,
        attemptedFallbacks: VoraRouter2FallbackConfig[]
    ): Error {
        const attemptedModels = attemptedFallbacks
            .map((fallback) => `${fallback.order}:${fallback.providerLabel}:${fallback.modelName}`)
            .join(', ')
        const primaryMessage = primaryError instanceof Error ? primaryError.message : String(primaryError)
        const lastMessage = lastError instanceof Error ? lastError.message : String(lastError)
        const error = new Error(
            `ChatVoraRouter2 failed primary OpenRouter attempts and all provider fallbacks${
                attemptedModels ? ` (${attemptedModels})` : ''
            }: primary=${primaryMessage}; last=${lastMessage}`
        )
        ;(error as any).cause = lastError
        return error
    }
}
