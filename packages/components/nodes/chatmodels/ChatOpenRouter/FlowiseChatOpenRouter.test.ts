import { AIMessage, AIMessageChunk } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import { ChatOpenRouter } from './FlowiseChatOpenRouter'

class DeterministicChatOpenRouter extends ChatOpenRouter {
    readonly attempts: string[] = []
    mode: 'generateFallback' | 'streamBeforeTokenFallback' | 'streamAfterTokenFailure' = 'generateFallback'

    protected shuffleAttempts<T>(attempts: T[]): T[] {
        return attempts
    }

    protected createAttemptModel(attempt: any): any {
        const attempts = this.attempts
        const mode = this.mode

        return {
            async _generate() {
                attempts.push(`${attempt.modelName}:${attempt.apiKey}`)
                if (attempts.length === 1) throw new Error('first attempt failed')

                return {
                    generations: [
                        {
                            text: 'ok',
                            message: new AIMessage('ok')
                        }
                    ],
                    llmOutput: {}
                }
            },
            async *_streamResponseChunks() {
                attempts.push(`${attempt.modelName}:${attempt.apiKey}`)

                if (mode === 'streamBeforeTokenFallback' && attempts.length === 1) {
                    throw new Error('stream failed before token')
                }

                yield new ChatGenerationChunk({
                    text: 'ok',
                    message: new AIMessageChunk({ content: 'ok' })
                })

                if (mode === 'streamAfterTokenFailure') {
                    throw new Error('stream failed after token')
                }
            }
        }
    }
}

describe('ChatOpenRouter fallback candidates', () => {
    it('uses the first comma-separated model as the cache model and removes multiple API keys from the cache key', () => {
        const model = new ChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            temperature: 0.7
        })

        const params = model._identifyingParams()
        const cacheKey = model._getSerializedCacheKeyParametersForCall({})

        expect(params.model_name).toBe('openai/gpt-5.4')
        expect(params.model).toBe('openai/gpt-5.4')
        expect(params.apiKey).toBeUndefined()
        expect(cacheKey).toContain('openai/gpt-5.4')
        expect(cacheKey).not.toContain('openai/gpt-5.5')
        expect(cacheKey).not.toContain('key-a')
        expect(cacheKey).not.toContain('key-b')
    })

    it('keeps single model and single key cache parameters unchanged', () => {
        const model = new ChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4',
            apiKey: 'key-a',
            temperature: 0.7
        })

        const params = model._identifyingParams()

        expect(params.model_name).toBe('openai/gpt-5.4')
        expect(params.model).toBe('openai/gpt-5.4')
        expect(params.apiKey).toBe('key-a')
    })

    it('falls back to the next model/key pair when a non-streaming call fails', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b'
        })

        const result = await model._generate([], {} as any)

        expect(result.generations[0].text).toBe('ok')
        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.4:key-b'])
    })

    it('falls back for streaming failures before the first token', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a'
        })
        model.mode = 'streamBeforeTokenFallback'

        const chunks: string[] = []
        for await (const chunk of model._streamResponseChunks([], {} as any)) {
            chunks.push(chunk.text)
        }

        expect(chunks).toEqual(['ok'])
        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.5:key-a'])
    })

    it('does not fall back for streaming failures after a token was yielded', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a'
        })
        model.mode = 'streamAfterTokenFailure'

        const chunks: string[] = []
        await expect(async () => {
            for await (const chunk of model._streamResponseChunks([], {} as any)) {
                chunks.push(chunk.text)
            }
        }).rejects.toThrow('stream failed after token')

        expect(chunks).toEqual(['ok'])
        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a'])
    })
})
