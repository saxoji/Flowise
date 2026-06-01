jest.mock('../../../src/utils', () => ({
    getBaseClasses: jest.fn(() => ['BaseChatModel']),
    getCredentialData: jest.fn(),
    getCredentialParam: jest.fn()
}))

const { nodeClass: ChatVoraRouter_ChatModels } = require('../ChatVoraRouter/ChatVoraRouter')
const { nodeClass: ChatVoraRouter2_ChatModels } = require('./ChatVoraRouter2')

describe('ChatVoraRouter2 node', () => {
    it('registers as a new node without changing the existing VoraRouter node', () => {
        const voraRouterNode = new ChatVoraRouter_ChatModels()
        const voraRouter2Node = new ChatVoraRouter2_ChatModels()

        expect(voraRouterNode.name).toBe('chatVoraRouter')
        expect(voraRouterNode.type).toBe('ChatVoraRouter')

        expect(voraRouter2Node.label).toBe('VoraRouter2')
        expect(voraRouter2Node.name).toBe('chatVoraRouter2')
        expect(voraRouter2Node.type).toBe('ChatVoraRouter2')
        expect(voraRouter2Node.category).toBe('Chat Models')
        expect(voraRouter2Node.icon).toBe('voraRouter.png')
        expect(voraRouter2Node.inputs).toContainEqual(
            expect.objectContaining({
                label: 'Image Resolution',
                name: 'imageResolution',
                default: 'auto',
                show: { allowImageUploads: true }
            })
        )
        expect(voraRouter2Node.inputs).toContainEqual(
            expect.objectContaining({
                label: 'Fallback OpenAI Credential',
                name: 'fallbackOpenAICredential',
                type: 'asyncOptions',
                credentialNames: ['openAIApi'],
                additionalParams: true
            })
        )
        expect(voraRouter2Node.inputs).toContainEqual(
            expect.objectContaining({
                label: 'Fallback xAI Grok Order',
                name: 'fallbackXAIOrder',
                default: 'disabled',
                additionalParams: true
            })
        )
    })
})
