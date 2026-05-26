import { AIMessage, AIMessageChunk } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import { ChatOpenRouter } from './FlowiseChatOpenRouter'

class DeterministicChatOpenRouter extends ChatOpenRouter {
    readonly attempts: string[] = []
    mode: 'success' | 'generateFallback' | 'streamBeforeTokenFallback' | 'streamAfterTokenFailure' = 'generateFallback'
    failuresRemaining = 1

    protected createAttemptModel(attempt: any): any {
        const attempts = this.attempts
        const mode = this.mode
        const model = this

        return {
            async _generate() {
                attempts.push(`${attempt.modelName}:${attempt.apiKey}`)
                if (mode === 'generateFallback' && model.failuresRemaining > 0) {
                    model.failuresRemaining -= 1
                    throw new Error('first attempt failed')
                }

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

                if (mode === 'streamBeforeTokenFallback' && model.failuresRemaining > 0) {
                    model.failuresRemaining -= 1
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
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-generate-fallback',
            roundRobinSessionId: 'session-a'
        })

        const result = await model._generate([], {} as any)

        expect(result.generations[0].text).toBe('ok')
        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.5:key-b'])
    })

    it('falls back for streaming failures before the first token', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a',
            roundRobinScope: 'test-stream-fallback',
            roundRobinSessionId: 'session-a'
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
            apiKey: 'key-a',
            roundRobinScope: 'test-stream-after-token',
            roundRobinSessionId: 'session-a'
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

    it('keeps the assigned primary attempt across calls within the same session', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-same-session',
            roundRobinSessionId: 'session-a'
        })
        const rebuiltModelForSameSession = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-same-session',
            roundRobinSessionId: 'session-a'
        })
        model.mode = 'success'
        rebuiltModelForSameSession.mode = 'success'

        await model._generate([], {} as any)
        await model._generate([], {} as any)
        await model._generate([], {} as any)
        await rebuiltModelForSameSession._generate([], {} as any)

        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.4:key-a', 'openai/gpt-5.4:key-a'])
        expect(rebuiltModelForSameSession.attempts).toEqual(['openai/gpt-5.4:key-a'])
    })

    it('round robins the primary attempt when assigning new sessions', async () => {
        const sessionA = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-new-session-assignment',
            roundRobinSessionId: 'session-a'
        })
        const sessionB = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-new-session-assignment',
            roundRobinSessionId: 'session-b'
        })
        sessionA.mode = 'success'
        sessionB.mode = 'success'

        await sessionA._generate([], {} as any)
        await sessionA._generate([], {} as any)
        await sessionB._generate([], {} as any)

        expect(sessionA.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.4:key-a'])
        expect(sessionB.attempts).toEqual(['openai/gpt-5.4:key-b'])
    })

    it('tries remaining model/key pairs before failing when preferred fallbacks are exhausted', async () => {
        const model = new DeterministicChatOpenRouter('chatOpenRouter_0', {
            modelName: 'openai/gpt-5.4, openai/gpt-5.5',
            apiKey: 'key-a, key-b',
            roundRobinScope: 'test-exhausted-preferred-fallbacks',
            roundRobinSessionId: 'session-a'
        })
        model.failuresRemaining = 2

        const result = await model._generate([], {} as any)

        expect(result.generations[0].text).toBe('ok')
        expect(model.attempts).toEqual(['openai/gpt-5.4:key-a', 'openai/gpt-5.5:key-b', 'openai/gpt-5.4:key-b'])
    })
})
