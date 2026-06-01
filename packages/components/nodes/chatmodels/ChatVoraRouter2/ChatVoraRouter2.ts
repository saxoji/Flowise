import { ChatOpenAI as LangchainChatOpenAI, ChatOpenAIFields } from '@langchain/openai'
import { BaseCache } from '@langchain/core/caches'
import { ICommonObject, IMultiModalOption, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { ChatVoraRouter2, VoraRouter2FallbackConfig, VoraRouter2FallbackProvider } from './FlowiseChatVoraRouter2'

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

const fallbackOrderOptions = [
    {
        label: 'Disabled',
        name: 'disabled'
    },
    {
        label: '1',
        name: '1'
    },
    {
        label: '2',
        name: '2'
    },
    {
        label: '3',
        name: '3'
    },
    {
        label: '4',
        name: '4'
    }
]

type FallbackDefinition = {
    provider: VoraRouter2FallbackProvider
    providerLabel: string
    credentialInput: string
    credentialLabel: string
    credentialNames: string[]
    credentialParam: string
    modelInput: string
    modelLabel: string
    modelPlaceholder: string
    orderInput: string
    orderLabel: string
}

const fallbackDefinitions: FallbackDefinition[] = [
    {
        provider: 'openai',
        providerLabel: 'OpenAI',
        credentialInput: 'fallbackOpenAICredential',
        credentialLabel: 'Fallback OpenAI Credential',
        credentialNames: ['openAIApi'],
        credentialParam: 'openAIApiKey',
        modelInput: 'fallbackOpenAIModel',
        modelLabel: 'Fallback OpenAI Model Name',
        modelPlaceholder: 'gpt-4o-mini',
        orderInput: 'fallbackOpenAIOrder',
        orderLabel: 'Fallback OpenAI Order'
    },
    {
        provider: 'xai',
        providerLabel: 'xAI Grok',
        credentialInput: 'fallbackXAICredential',
        credentialLabel: 'Fallback xAI Grok Credential',
        credentialNames: ['xaiApi'],
        credentialParam: 'xaiApiKey',
        modelInput: 'fallbackXAIModel',
        modelLabel: 'Fallback xAI Grok Model Name',
        modelPlaceholder: 'grok-4',
        orderInput: 'fallbackXAIOrder',
        orderLabel: 'Fallback xAI Grok Order'
    },
    {
        provider: 'anthropic',
        providerLabel: 'Anthropic Claude',
        credentialInput: 'fallbackAnthropicCredential',
        credentialLabel: 'Fallback Anthropic Credential',
        credentialNames: ['anthropicApi'],
        credentialParam: 'anthropicApiKey',
        modelInput: 'fallbackAnthropicModel',
        modelLabel: 'Fallback Anthropic Model Name',
        modelPlaceholder: 'claude-sonnet-4-5',
        orderInput: 'fallbackAnthropicOrder',
        orderLabel: 'Fallback Anthropic Order'
    },
    {
        provider: 'google',
        providerLabel: 'Google Gemini',
        credentialInput: 'fallbackGoogleCredential',
        credentialLabel: 'Fallback Google Gemini Credential',
        credentialNames: ['googleGenerativeAI'],
        credentialParam: 'googleGenerativeAPIKey',
        modelInput: 'fallbackGoogleModel',
        modelLabel: 'Fallback Google Gemini Model Name',
        modelPlaceholder: 'gemini-2.5-pro',
        orderInput: 'fallbackGoogleOrder',
        orderLabel: 'Fallback Google Gemini Order'
    }
]

const createFallbackInputs = (): INodeParams[] =>
    fallbackDefinitions.flatMap((fallback) => [
        {
            label: fallback.credentialLabel,
            name: fallback.credentialInput,
            type: 'asyncOptions',
            credentialNames: fallback.credentialNames,
            optional: true,
            additionalParams: true
        },
        {
            label: fallback.modelLabel,
            name: fallback.modelInput,
            type: 'string',
            placeholder: fallback.modelPlaceholder,
            optional: true,
            additionalParams: true
        },
        {
            label: fallback.orderLabel,
            name: fallback.orderInput,
            type: 'options',
            options: fallbackOrderOptions,
            default: 'disabled',
            optional: true,
            additionalParams: true
        }
    ])

class ChatVoraRouter2_ChatModels extends ChatOpenRouter_ChatModels implements INode {
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
        this.label = 'VoraRouter2'
        this.name = 'chatVoraRouter2'
        this.version = 1.0
        this.type = 'ChatVoraRouter2'
        this.icon = 'voraRouter.png'
        this.category = 'Chat Models'
        this.description = 'Vora LLM Router with provider fallback chain'
        this.baseClasses = [this.type, ...getBaseClasses(LangchainChatOpenAI)]

        if (!this.inputs.some((input) => input.name === imageResolutionInput.name)) {
            const allowImageUploadsIndex = this.inputs.findIndex((input) => input.name === 'allowImageUploads')
            this.inputs.splice(allowImageUploadsIndex + 1, 0, imageResolutionInput)
        }

        this.inputs.push(...createFallbackInputs())
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const temperature = nodeData.inputs?.temperature as string
        const modelName = nodeData.inputs?.modelName as string
        const maxTokens = nodeData.inputs?.maxTokens as string
        const topP = nodeData.inputs?.topP as string
        const frequencyPenalty = nodeData.inputs?.frequencyPenalty as string
        const presencePenalty = nodeData.inputs?.presencePenalty as string
        const timeout = nodeData.inputs?.timeout as string
        const streaming = nodeData.inputs?.streaming as boolean
        const basePath = (nodeData.inputs?.basepath as string) || 'https://openrouter.ai/api/v1'
        const baseOptions = nodeData.inputs?.baseOptions
        const cache = nodeData.inputs?.cache as BaseCache
        const allowImageUploads = nodeData.inputs?.allowImageUploads as boolean
        const roundRobinSessionId = (options?.sessionId as string) || (options?.chatId as string)
        const roundRobinScope = [((options?.chatflowid as string) || (options?.chatflowId as string)), nodeData.id].filter(Boolean).join(':')

        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const openRouterApiKey = getCredentialParam('openRouterApiKey', credentialData, nodeData)
        const fallbackConfigs = await this.buildFallbackConfigs(nodeData, options)

        const obj: ChatOpenAIFields & { roundRobinScope?: string; roundRobinSessionId?: string } = {
            temperature: parseFloat(temperature),
            modelName,
            openAIApiKey: openRouterApiKey,
            apiKey: openRouterApiKey,
            streaming: streaming ?? true,
            roundRobinScope: roundRobinScope || nodeData.id,
            roundRobinSessionId
        }

        if (maxTokens) obj.maxTokens = parseInt(maxTokens, 10)
        if (topP) obj.topP = parseFloat(topP)
        if (frequencyPenalty) obj.frequencyPenalty = parseFloat(frequencyPenalty)
        if (presencePenalty) obj.presencePenalty = parseFloat(presencePenalty)
        if (timeout) obj.timeout = parseInt(timeout, 10)
        if (cache) obj.cache = cache

        let parsedBaseOptions: any | undefined = undefined

        if (baseOptions) {
            try {
                parsedBaseOptions = typeof baseOptions === 'object' ? baseOptions : JSON.parse(baseOptions)
            } catch (exception) {
                throw new Error("Invalid JSON in the ChatVoraRouter2's BaseOptions: " + exception)
            }
        }

        if (basePath || parsedBaseOptions) {
            obj.configuration = {
                baseURL: basePath,
                defaultHeaders: parsedBaseOptions
            }
        }

        const multiModalOption: IMultiModalOption = {
            image: {
                allowImageUploads: allowImageUploads ?? false
            }
        }

        const model = new ChatVoraRouter2(nodeData.id, obj, fallbackConfigs)
        model.setMultiModalOption(multiModalOption)
        return model
    }

    private async buildFallbackConfigs(nodeData: INodeData, options: ICommonObject): Promise<VoraRouter2FallbackConfig[]> {
        const fallbackConfigs: VoraRouter2FallbackConfig[] = []
        const usedOrders = new Map<number, string>()

        for (const fallback of fallbackDefinitions) {
            const orderValue = `${nodeData.inputs?.[fallback.orderInput] ?? 'disabled'}`.trim()
            const credentialId = `${nodeData.inputs?.[fallback.credentialInput] ?? ''}`.trim()
            const modelName = `${nodeData.inputs?.[fallback.modelInput] ?? ''}`.trim()

            if (!orderValue || orderValue === 'disabled') continue

            const order = parseInt(orderValue, 10)
            if (!Number.isInteger(order) || order < 1 || order > 4) {
                throw new Error(`${fallback.orderLabel} must be Disabled or a number from 1 to 4`)
            }

            if (usedOrders.has(order)) {
                throw new Error(`${fallback.orderLabel} duplicates fallback order ${order} already used by ${usedOrders.get(order)}`)
            }
            if (!credentialId || !modelName) {
                throw new Error(`${fallback.providerLabel} fallback requires credential, model name, and order`)
            }

            const credentialData = await getCredentialData(credentialId, options)
            const apiKey = getCredentialParam(fallback.credentialParam, credentialData, { ...nodeData, inputs: {} })
            if (!apiKey) throw new Error(`${fallback.providerLabel} fallback credential is missing ${fallback.credentialParam}`)

            usedOrders.set(order, fallback.providerLabel)
            fallbackConfigs.push({
                provider: fallback.provider,
                providerLabel: fallback.providerLabel,
                modelName,
                apiKey,
                order
            })
        }

        return fallbackConfigs.sort((a, b) => a.order - b.order)
    }
}

module.exports = { nodeClass: ChatVoraRouter2_ChatModels }
