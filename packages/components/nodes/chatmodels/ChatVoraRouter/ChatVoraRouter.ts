import { ChatOpenAI as LangchainChatOpenAI } from '@langchain/openai'
import { INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'

const { nodeClass: ChatOpenRouter_ChatModels } = require('../ChatOpenRouter/ChatOpenRouter')

const imageResolutionInput: INodeParams = {
    label: 'Image Resolution',
    description: 'This parameter controls the resolution in which the model views the image.',
    name: 'imageResolution',
    type: 'options',
    options: [
        {
            label: 'Low',
            name: 'low'
        },
        {
            label: 'High',
            name: 'high'
        },
        {
            label: 'Auto',
            name: 'auto'
        }
    ],
    default: 'auto',
    optional: false,
    show: {
        allowImageUploads: true
    }
}

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

        if (!this.inputs.some((input) => input.name === imageResolutionInput.name)) {
            const allowImageUploadsIndex = this.inputs.findIndex((input) => input.name === 'allowImageUploads')
            this.inputs.splice(allowImageUploadsIndex + 1, 0, imageResolutionInput)
        }
    }
}

module.exports = { nodeClass: ChatVoraRouter_ChatModels }
