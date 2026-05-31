const { nodeClass: ChatOpenRouter_ChatModels } = require('../ChatOpenRouter/ChatOpenRouter')
const { nodeClass: ChatVoraRouter_ChatModels } = require('./ChatVoraRouter')

describe('ChatVoraRouter node', () => {
    it('registers as a separate node while reusing ChatOpenRouter behavior', () => {
        const openRouterNode = new ChatOpenRouter_ChatModels()
        const voraRouterNode = new ChatVoraRouter_ChatModels()

        expect(voraRouterNode.label).toBe('VoraRouter')
        expect(voraRouterNode.name).toBe('chatVoraRouter')
        expect(voraRouterNode.type).toBe('ChatVoraRouter')
        expect(voraRouterNode.category).toBe('Chat Models')
        expect(voraRouterNode.description).toBe('Vora LLM Router Interface API')
        expect(voraRouterNode.icon).toBe('voraRouter.png')
        expect(voraRouterNode.baseClasses[0]).toBe('ChatVoraRouter')
        expect(voraRouterNode.baseClasses).toContain('BaseChatModel')
        expect(voraRouterNode.credential).toEqual(openRouterNode.credential)
        expect(voraRouterNode.inputs).toEqual(openRouterNode.inputs)
        expect(voraRouterNode.init).toBe(openRouterNode.init)
    })
})
