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
        this.label = 'ChatVoraRouter'
        this.name = 'chatVoraRouter'
        this.type = 'ChatVoraRouter'
        this.icon = 'openRouter.svg'
        this.category = 'Chat Models'
        this.description = 'Wrapper around Open Router Inference API'
        this.baseClasses = [this.type, ...getBaseClasses(LangchainChatOpenAI)]
    }
}

module.exports = { nodeClass: ChatVoraRouter_ChatModels }
