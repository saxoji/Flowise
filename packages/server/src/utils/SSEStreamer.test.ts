import { SSEStreamer } from './SSEStreamer'

// Minimal Express Response stub capturing SSE writes and socket close.
const mockRes = () => {
    const state = {
        writes: [] as string[],
        ended: false,
        write(s: string) {
            state.writes.push(s)
            return true
        },
        end() {
            state.ended = true
        }
    }
    return state as any
}

describe('SSEStreamer.closeAllClients (graceful shutdown drain)', () => {
    it('emits a terminal error event and closes every open client, then empties the map', () => {
        const s = new SSEStreamer()
        const a = mockRes()
        const b = mockRes()
        s.addClient('chatA', a)
        s.addExternalClient('chatB', b)

        const drained = s.closeAllClients()

        expect(drained).toBe(2)
        expect(s.hasClient('chatA')).toBe(false)
        expect(s.hasClient('chatB')).toBe(false)
        expect(a.writes.join('')).toContain('"event":"error"')
        expect(a.ended).toBe(true)
        expect(b.ended).toBe(true)
    })

    it('does NOT emit an "end" event (an interrupted stream must not look like a successful completion)', () => {
        const s = new SSEStreamer()
        const a = mockRes()
        s.addClient('chatA', a)

        s.closeAllClients()

        expect(a.writes.join('')).not.toContain('"event":"end"')
    })

    it('uses the provided message in the terminal event', () => {
        const s = new SSEStreamer()
        const a = mockRes()
        s.addClient('c', a)

        s.closeAllClients('server restarting')

        expect(a.writes.join('')).toContain('server restarting')
    })

    it('is a no-op that returns 0 when there are no clients', () => {
        const s = new SSEStreamer()
        expect(s.closeAllClients()).toBe(0)
    })

    it('does not throw when a client response is already disconnected, and still removes it', () => {
        const s = new SSEStreamer()
        const bad = {
            write() {
                throw new Error('EPIPE')
            },
            end() {
                throw new Error('socket closed')
            }
        } as any
        s.addClient('c', bad)

        expect(() => s.closeAllClients()).not.toThrow()
        expect(s.hasClient('c')).toBe(false)
    })

    it('also clears observers so no stale references remain after shutdown', () => {
        const s = new SSEStreamer()
        const a = mockRes()
        s.addClient('src', a)
        s.addObserver('src', 'obs1')

        s.closeAllClients()

        expect(s.hasClientOrObserver('src')).toBe(false)
    })
})
