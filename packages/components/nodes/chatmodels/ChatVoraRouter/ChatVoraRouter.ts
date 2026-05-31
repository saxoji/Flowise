import { ChatOpenAI as LangchainChatOpenAI } from '@langchain/openai'
import { INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'

const { nodeClass: ChatOpenRouter_ChatModels } = require('../ChatOpenRouter/ChatOpenRouter')

class ChatVoraRouter_ChatModels extends ChatOpenRouter_ChatModels {
    label: string
    name: string
    version: number
    type: string
    icon: string
    category: string
    description: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        super()
        this.label = 'VoraRouter'
        this.name = 'chatVoraRouter'
        this.type = 'ChatVoraRouter'
        this.icon = 'voraRouter.png'
        this.category = 'Chat Models'
        this.description = 'Vora LLM Router Interface API'
        this.baseClasses = [this.type, ...getBaseClasses(LangchainChatOpenAI)]
    }
}

module.exports = { nodeClass: ChatVoraRouter_ChatModels }
