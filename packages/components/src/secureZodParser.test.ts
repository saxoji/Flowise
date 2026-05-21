import { SecureZodSchemaParser } from './secureZodParser'

describe('SecureZodSchemaParser', () => {
    it('parses description strings that contain URL slashes and dots', () => {
        const schema = SecureZodSchemaParser.parseZodSchema(`
            z.object({
                imageUrl: z.string().describe("Default value is http://wwww.xxx.com/ddd.jpg")
            })
        `) as any

        expect(schema.shape.imageUrl.description).toBe('Default value is http://wwww.xxx.com/ddd.jpg')
        expect(schema.parse({ imageUrl: 'https://example.com/image.jpg' })).toEqual({ imageUrl: 'https://example.com/image.jpg' })
    })

    it('keeps URLs inside strings while removing actual comments', () => {
        const schema = SecureZodSchemaParser.parseZodSchema(`
            z.object({
                // This comment should be ignored.
                imageUrl: z.string().describe("Use https://example.com/assets/a.b.jpg") // trailing comment
            })
        `) as any

        expect(schema.shape.imageUrl.description).toBe('Use https://example.com/assets/a.b.jpg')
    })

    it('parses default string values that contain URL punctuation', () => {
        const schema = SecureZodSchemaParser.parseZodSchema(`
            z.object({
                imageUrl: z.string().default("http://wwww.xxx.com/ddd.jpg")
            })
        `) as any

        expect(schema.parse({})).toEqual({ imageUrl: 'http://wwww.xxx.com/ddd.jpg' })
    })
})
