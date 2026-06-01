import { AIMessage, AIMessageChunk } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import { ChatVoraRouter2, VoraRouter2FallbackConfig } from './FlowiseChatVoraRouter2'

class DeterministicChatVoraRouter2 extends ChatVoraRouter2 {
    readonly primaryAttempts: string[] = []
    readonly fallbackAttempts: string[] = []
    primaryMode: 'fail' | 'streamAfterTokenFailure' = 'fail'
    fallbackFailuresRemaining = 0

    protected createAttemptModel(attempt: any): any {
        const model = this
        return {
            async _generate() {
                model.primaryAttempts.push(`${attempt.modelName}:${attempt.apiKey}`)
                throw new Error('primary failed')
            },
            async *_streamResponseChunks() {
                model.primaryAttempts.push(`${attempt.modelName}:${attempt.apiKey}`)
                if (model.primaryMode === 'streamAfterTokenFailure') {
                    yield new ChatGenerationChunk({
                        text: 'primary',
                        message: new AIMessageChunk({ content: 'primary' })
                    })
                    throw new Error('primary stream failed after token')
                }
                throw new Error('primary stream failed before token')
            }
        }
    }

    protected createFallbackModel(fallback: VoraRouter2FallbackConfig): any {
        const model = this
        return {
            async _generate() {
                model.fallbackAttempts.push(`${fallback.provider}:${fallback.modelName}`)
                if (model.fallbackFailuresRemaining > 0) {
                    model.fallbackFailuresRemaining -= 1
                    throw new Error('fallback failed')
                }

                return {
                    generations: [
                        {
                            text: 'fallback ok',
                            message: new AIMessage('fallback ok')
                        }
                    ],
                    llmOutput: {}
                }
            },
            async *_streamResponseChunks() {
                model.fallbackAttempts.push(`${fallback.provider}:${fallback.modelName}`)
                yield new ChatGenerationChunk({
                    text: 'fallback ok',
                    message: new AIMessageChunk({ content: 'fallback ok' })
                })
            }
        }
    }
}

const fallbackConfigs: VoraRouter2FallbackConfig[] = [
    {
        provider: 'openai',
        providerLabel: 'OpenAI',
        modelName: 'gpt-fallback',
        apiKey: 'openai-key',
        order: 2
    },
    {
        provider: 'xai',
        providerLabel: 'xAI Grok',
        modelName: 'grok-fallback',
        apiKey: 'xai-key',
        order: 1
    }
]

describe('ChatVoraRouter2 provider fallbacks', () => {
    it('runs provider fallbacks by configured order after primary OpenRouter attempts fail', async () => {
        const model = new DeterministicChatVoraRouter2(
            'chatVoraRouter2_0',
            {
                modelName: 'openrouter-a, openrouter-b',
                apiKey: 'openrouter-key'
            },
            fallbackConfigs
        )
        model.fallbackFailuresRemaining = 1

        const result = await model._generate([], {} as any)

        expect(result.generations[0].text).toBe('fallback ok')
        expect(model.primaryAttempts).toEqual(['openrouter-a:openrouter-key', 'openrouter-b:openrouter-key'])
        expect(model.fallbackAttempts).toEqual(['xai:grok-fallback', 'openai:gpt-fallback'])
        expect(result.generations[0].generationInfo?.vora_router2_fallback_provider).toBe('openai')
    })

    it('falls back for streaming primary failures before the first token', async () => {
        const model = new DeterministicChatVoraRouter2(
            'chatVoraRouter2_0',
            {
                modelName: 'openrouter-a, openrouter-b',
                apiKey: 'openrouter-key'
            },
            fallbackConfigs
        )

        const chunks: string[] = []
        for await (const chunk of model._streamResponseChunks([], {} as any)) {
            chunks.push(chunk.text)
        }

        expect(chunks).toEqual(['fallback ok'])
        expect(model.fallbackAttempts).toEqual(['xai:grok-fallback'])
    })

    it('does not switch providers after a streaming primary attempt has yielded a token', async () => {
        const model = new DeterministicChatVoraRouter2(
            'chatVoraRouter2_0',
            {
                modelName: 'openrouter-a, openrouter-b',
                apiKey: 'openrouter-key'
            },
            fallbackConfigs
        )
        model.primaryMode = 'streamAfterTokenFailure'

        const chunks: string[] = []
        await expect(async () => {
            for await (const chunk of model._streamResponseChunks([], {} as any)) {
                chunks.push(chunk.text)
            }
        }).rejects.toThrow('primary stream failed after token')

        expect(chunks).toEqual(['primary'])
        expect(model.fallbackAttempts).toEqual([])
    })

    it('preserves VoraRouter2 provider fallback behavior through withConfig', async () => {
        const model = new DeterministicChatVoraRouter2(
            'chatVoraRouter2_0',
            {
                modelName: 'openrouter-a, openrouter-b',
                apiKey: 'openrouter-key'
            },
            fallbackConfigs
        )

        const configured = model.withConfig({ tags: ['test'] })

        expect(configured).toBeInstanceOf(ChatVoraRouter2)
        expect(((configured as ChatVoraRouter2)._identifyingParams() as any).vora_router2_fallbacks).toContain('xai:grok-fallback')
    })
})
