import { syncNodeInputsWithEdges } from './index'

const makeToolAgentNode = (tools: string[]) =>
    ({
        id: 'toolAgent_0',
        data: {
            id: 'toolAgent_0',
            inputs: { tools },
            inputAnchors: [{ id: 'toolAgent_0-input-tools-Tool', name: 'tools', type: 'Tool', list: true }]
        }
    }) as any

describe('syncNodeInputsWithEdges', () => {
    it('rebuilds missing tool input refs from connected edges', () => {
        const nodes = [makeToolAgentNode(['{{customTool_0.data.instance}}'])]
        const edges = [
            { source: 'customTool_0', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-tools-Tool' },
            { source: 'customTool_45', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-tools-Tool' }
        ] as any

        const result = syncNodeInputsWithEdges(nodes, edges)

        expect(result[0].data.inputs.tools).toEqual(['{{customTool_0.data.instance}}', '{{customTool_45.data.instance}}'])
    })

    it('removes stale tool refs that no longer have edges', () => {
        const nodes = [makeToolAgentNode(['{{customTool_0.data.instance}}', '{{customTool_45.data.instance}}'])]
        const edges = [{ source: 'customTool_0', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-tools-Tool' }] as any

        const result = syncNodeInputsWithEdges(nodes, edges)

        expect(result[0].data.inputs.tools).toEqual(['{{customTool_0.data.instance}}'])
    })

    it('deduplicates duplicate edges to the same list anchor', () => {
        const nodes = [makeToolAgentNode([])]
        const edges = [
            { source: 'customTool_45', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-tools-Tool' },
            { source: 'customTool_45', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-tools-Tool' }
        ] as any

        const result = syncNodeInputsWithEdges(nodes, edges)

        expect(result[0].data.inputs.tools).toEqual(['{{customTool_45.data.instance}}'])
    })

    it('syncs single connected input anchors without touching input params', () => {
        const nodes = [
            {
                id: 'toolAgent_0',
                data: {
                    id: 'toolAgent_0',
                    inputs: { model: '', systemMessage: 'keep me' },
                    inputAnchors: [{ id: 'toolAgent_0-input-model-BaseChatModel', name: 'model', type: 'BaseChatModel' }],
                    inputParams: [{ name: 'systemMessage', type: 'string' }]
                }
            }
        ] as any
        const edges = [{ source: 'chatOpenRouter_0', target: 'toolAgent_0', targetHandle: 'toolAgent_0-input-model-BaseChatModel' }] as any

        const result = syncNodeInputsWithEdges(nodes, edges)

        expect(result[0].data.inputs).toEqual({
            model: '{{chatOpenRouter_0.data.instance}}',
            systemMessage: 'keep me'
        })
    })

    it('leaves nodes without input anchors unchanged', () => {
        const node = {
            id: 'stickyNote_0',
            data: {
                id: 'stickyNote_0',
                inputs: { note: 'keep me' },
                inputAnchors: []
            }
        } as any

        const result = syncNodeInputsWithEdges([node], [])

        expect(result[0]).toBe(node)
    })
})
