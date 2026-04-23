#!/usr/bin/env node
import {CozoDb} from 'cozo-node'
import { pipeline } from "@huggingface/transformers";
import csv from 'csv-parser'
import fs from 'fs'
import { resolve, join, dirname, sep } from 'path'
import { fileURLToPath } from 'url'

import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// import {sanitizeValue} from './utils/sanitize.js'

const pathToThisFile = resolve(fileURLToPath(import.meta.url))
const pathPassedToNode = resolve(process.argv[1])
const isThisFileBeingRunViaCLI = pathToThisFile.includes(pathPassedToNode) || pathPassedToNode.includes('.npm-global')
const PATH = dirname(pathToThisFile)

export const config = {
    VECTOR_SIZE: 384, // number of dimensions (must match the models output)
    MODEL_NAME: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', // model to use (passed to Transformers.js)
    SHOW_ERRORS: true, // Show errors
    DB_FILE: join(PATH, 'vector.db'), // Path to the datbase file (SQLite file used by CozoDB)
    PATH: PATH, // directory of recall.js
    DEVICE: undefined, // Transformers.js device
    DTYPE: undefined, // Transformers.js dtype
    PROGRESS_CALLBACK: undefined // Transformers.js progress_callback
}

var db = null, initDone = false

export const getDb = () => {
    if(!db) {
        db = new CozoDb('sqlite', config.DB_FILE)
    }
    return db
}

async function printQuery(query, params = {}) {
    try{
        if(!initDone) {
            initDone = true
            await createTable()
        }
    }catch(err) {
        //console.log('CREATE TABLE ERROR', err)
    }
    try {
        let data = getDb().run(query, params)
        return data
    }catch(err){
        if(config.SHOW_ERRORS) console.error(err.display || err.message)
    }
}

export const getEmbeddings = async (text) => {
    let pipe = config._pipe
    if(!pipe) {
        config._pipe = await pipeline("feature-extraction", config.MODEL_NAME, {
            progress_callback:(progress) => {
                if(config.PROGRESS_CALLBACK) return config.PROGRESS_CALLBACK();
                if(progress.status === "progress_total"){
                    process.stdout.write(`\r\x1b[K✅ Loaded ${ Math.round(progress.progress)}% ${progress.name || "model"}`)
                }
            },
            device: config.DEVICE,
            dtype: config.DTYPE
        });
        pipe = config._pipe
    }
    const embedding = await pipe(text, { pooling: "mean", normalize: true });
    return Array.from(embedding.data)
}

export const createTable = async () => {
    // create table (id, v, input, result, data)
    let tableCreated = await printQuery(`:create embeddings {id: String, category: String => v: <F32; ${config.VECTOR_SIZE}>, input: String, result: String, data: Json}`)
    if(tableCreated){
        // create index
        let indexCreated = await printQuery(`::hnsw create embeddings:index_name {
            dim: ${config.VECTOR_SIZE},
            m: 50,
            dtype: F32,
            fields: [v],
            distance: L2, # Cosine, IP
            ef_construction:50, # number of nearest neighbors
            extend_candidates: false, # include nearest neighbors of the nearest neighbors
            keep_pruned_connections: false,
        }`)
        return tableCreated && indexCreated
    }
    return false
}

export const add = async (input, result, data={}, category="") => {
    if(!input || !result) return
    input = sanitizeString(input)
    result = sanitizeString(result)
    const embedding = await getEmbeddings(input)
    let id = data.id || Math.random().toString().substring(2)
    return await printQuery(`?[id, v, input, result, data, category] <- [["${id}", ${JSON.stringify(embedding)}, ${JSON.stringify(input.replaceAll('"', "'"))}, ${JSON.stringify(result.replaceAll('"', "'"))}, ${JSON.stringify(data)}, ${JSON.stringify(category.replaceAll('"', "'"))} ]]
        :put embeddings {id, category => v, input, result, data}
    `)
}

/**
 * 
 * Batch array:
 * [{input:"", result:"", data:{}}]
 * 
 * @param {Array} batch 
 * @returns 
 */
export const addBatch = async (batch, opts={onProgress:null}) => {
    if(!batch || !Array.isArray(batch)) return
    let vectorBatch = []
    for(let i=0;i<batch.length; i++){
        let {input, result, data, category} = batch[i]

        if(!input || !result) continue
        if(!data) data = {}
        if(!category) category = ''
        const embedding = await getEmbeddings(input)
        batch[i].embedding = embedding
        let item = ''
        if(i == 0) {
            item += `?[id, v, input, result, data, category] <- [`
        }

        input = sanitizeString(input)
        result = sanitizeString(result)

        let id = data?.id ? data.id : Math.random().toString().substring(2)
        item += `["${id}", ${JSON.stringify(embedding)}, ${JSON.stringify(input)}, ${JSON.stringify(result)}, ${JSON.stringify(data)}, ${JSON.stringify(category)} ],`
        if(i == batch.length-1) {
            item += `]
            :put embeddings {id, category => v, input, result, data}`
        }
        vectorBatch.push(item)

        if(opts.onProgress && typeof opts.onProgress == 'function') {
            await opts.onProgress({index: i+1, total:batch.length, item: batch[i], embedding, percent: Math.round((i+1) / batch.length * 100)})
        }
    }
    return await printQuery(vectorBatch.join("\n"))
}

const sanitizeString = (str)=>{
    return str.replace(/[\/#$%\^&\*{}=_`~()\"]/g," ").replace(/\s{2,}/g, " ").trim()
}

export const remove = async (id, category="") => {
    if(!id || typeof id != 'string') return 
    id = id.replace(/[^a-zA-Z0-9]/g, '')
    category = sanitizeString(category)
    if(!id || !category) return
    let results = await printQuery(
        `?[id, category] <- [['${id}', '${category}']]
        ::rm embeddings {id, category}`)
    return results
}

export const removeAllByCategory = async (category="") => {
    category = sanitizeString(category)
    if(!category) return
    let results
    try {
        results = await printQuery(
            `?[id, category] := *embeddings{id, category}, category = "${category}"
            :rm embeddings {id, category}`)
    }catch(err){
        console.error(err)
    }
    return results
}

export const searchText = async (text, category="", numResults = 5, includeInput=false) => {
    const embedding = await getEmbeddings(text)
    let results = await printQuery(`?[dist, result, id, data, category${includeInput? ', input' : ''}] := ~embeddings:index_name { id, v, input, result, data, category${includeInput? ', input' : ''} |
        query: q,
        k: ${numResults}, # number of results
        ef: 50, # number of neighbours to consider 
        bind_distance: dist,
        filter: category==${JSON.stringify(category)},
        radius: 10.0
    }, q = vec(${JSON.stringify(embedding)})
    :sort -dist`)
    return results
}

export const vectorSearch = async (query, category='', numResults=5) => {
    let result = undefined
    try{
        result = await searchText(query, category, numResults)
    }catch(err){
        if(config.SHOW_ERRORS) console.error(err.display || err.message)
    }
    return result
}

const cmdArgs = (list = []) => {
    let args = {}, current = null
    for(let i=0; i<process.argv.length; i++){
        let val = process.argv[i]
        if(current && !list.includes(val)){
            args[current] = val
            current = null
        } 
        if(list.includes(val)) {
            current = val
            args[current] = ''
        }
    }
    args._cmd = process.argv[1].split(sep).pop()
    return args
}

export const nuke = () => {
    return fs.unlinkSync(config.DB_FILE)
}

export const importFromJSONStream = async (fileName) => {
    async function jsonStream(readable, callback = async function(){}) {
        readable.setEncoding('utf8');
        let data = '';
        for await (const chunk of readable) {
            if(chunk.indexOf("\n")) {
                pts = chunk.split("\n")
                for(let i=0;i<pts.length; i++){
                    data += pts[i]
                    try {
                        let json = JSON.parse(data)
                        await callback(json)
                        json = null
                        data = ''
                    }catch(err) {
                        //console.error(err)
                    }
                }
            }else{
                data += chunk;
            }
        }
    }
    let batchSize = 40, batch = [], i=0, currentBatch = 0
    let stream = typeof fileName == 'string' ? fs.createReadStream(fileName) : fileName
    await jsonStream(stream, async (json) => {
        if(json.input && json.result){
            if(!json.data) json.data = {}
            if(i % batchSize === 0){
                if(batch.length) {
                    currentBatch = currentBatch + 1
                    console.log(`Adding batch ${currentBatch} (${batch.length} items)`)
                    await addBatch(batch)
                    batch = []
                }
            }
            batch.push(json)
            i=i+1
        }
    })
    if(batch.length) {
        console.log(`Adding batch ${currentBatch + 1} (${batch.length} items)`)
        await addBatch(batch)
    }
}

export const importFromCSVorTSV = async (fileName, inputHeader, resultHeader) => {
    if(!fileName || !fileName.includes('.')) return
    let ext = fileName.split('.').pop()
    ext = ext.toLowerCase()
    if(ext != 'csv' && ext != 'tsv') return console.log('File must have csv or tsv extension')
    let parseOpts = { 
        separator: ext == 'tsv' ? '\t' : ',', 
        mapHeaders: ({ header, index }) => {
            if(inputHeader) {
                if(inputHeader == header){
                    return 'input'
                }
            }else if(index === 0){
                return 'input'
            }
            if(resultHeader){
                if(resultHeader == header){
                    return 'result'
                }
            }else if(index === 1){
                return 'result'
            }
            return header.replaceAll(/\W/gi, '_').replaceAll(/[^a-zA-Z0-9\_]/g, '').toLowerCase()
        }
    }
    let fetchFromFile = async (fileName) => {
        return new Promise(async (resolve, reject)=>{
            let results = []
            fs.createReadStream(fileName)
            .pipe(csv(parseOpts))
            .on('data', async (data) => {
                results.push(data)
            })
            .on('end', () => {
                console.log(`${fileName} loaded.`);
                resolve(results)
            }).on('error', (err) => {
                console.error(err);
            })
        })
    }

    
    let results = await fetchFromFile(fileName)

    let batchSize = 40, batch = [], currentBatch = 0, totalBatches = Math.ceil(results.length / batchSize), dataHeaders = Object.keys(results[results.length-1]).filter(k => k != 'input' && k != 'result'), data
    for(let i=0; i<results.length; i++){
        if(i % batchSize === 0){
            if(batch.length) {
                currentBatch = currentBatch + 1
                console.log(`Adding batch ${currentBatch} of ${totalBatches} (${batch.length} items)`)
                await addBatch(batch)
                batch = []
            }
        }
        data = {}
        dataHeaders.forEach(k => k && results[i][k] ? data[k] = results[i][k] : null)
        batch.push({
            input: results[i].input,
            result: results[i].result,
            data
        })
    }
    if(batch.length) {
        console.log(`Adding batch ${currentBatch + 1} of ${totalBatches} (${batch.length} items)`)
        await addBatch(batch)
    }
}

const mcp = async () => {

    // Create an MCP server
    // const server = new McpServer({
    //     name: "Demo",
    //     version: "1.0.0"
    // });
    
    // // Add an addition tool
    // server.tool("add",
    //     { a: z.number(), b: z.number() },
    //     async ({ a, b }) => ({
    //         content: [{ type: "text", text: String(a + b) }]
    //     })
    // );
    
    // // Add a dynamic greeting resource
    // server.resource(
    //     "greeting",
    //     new ResourceTemplate("greeting://{name}", { list: undefined }),
    //     async (uri, { name }) => ({
    //     contents: [{
    //         uri: uri.href,
    //         text: `Hello, ${name}!`
    //     }]
    //     })
    // );


    const server = new McpServer({
        name: "Recall",
        description: "Recall provides semantic search on the local vector database.",
        version: "1.0.0"
    });
    
    // server.resource(
    //     "echo",
    //     new ResourceTemplate("echo://{message}", { list: undefined }),
    //     async (uri, { message }) => ({
    //     contents: [{
    //         uri: uri.href,
    //         text: `Resource echo: ${message}`
    //     }]
    //     })
    // );
    
    server.tool(
        "search",
        { 
            text: z.string(),
            //numberOfResults: z.number()
        },
        async ({ text, numberOfResults }) => {
            if(numberOfResults && numberOfResults > 50) numberOfResults = 50

            let startTime = performance.now()
            let results = await searchText(text, numberOfResults)
            var timeDiff = ((performance.now() - startTime) / 1000).toFixed(2)
            let content = [
                { 
                    type: "text", 
                    text: `Sorry. Recal search didn't find anything.` 
                }
            ]
            if(results && results.rows && results.rows.length) {
                // content = results.rows.map(r => {
                //     return { 
                //         type: "text", 
                //         text: r[1]
                //     }
                // })
                content = [{ 
                    type: "text", 
                    text: `Recal search found the following results in ${timeDiff}s:` 
                }]
                for(let i=0; i<results.rows.length; i++){
                    let row = results.rows[i]
                    content.push({
                        type: "text", 
                        text: row[1]
                    })
                    // if(results.rows[2] && Object.keys(results.rows[2])){
                    //     content.push({
                    //         type: "json", 
                    //         text: row[2]
                    //     })
                    // }
                }
            }

            return {
                content
            }
        }
    );
    
    // server.prompt(
    //     "echo",
    //     { message: z.string() },
    //     ({ message }) => ({
    //     messages: [{
    //         role: "user",
    //         content: {
    //         type: "text",
    //         text: `Please process this message: ${message}`
    //         }
    //     }]
    //     })
    // );
    
    // Start receiving messages on stdin and sending messages on stdout
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

const splitSentences = (text) => {
    return text.replace(/([.?!])\s*(?=[A-Z])/g, "$1|").split("|")
}

const runCLI = async () => {
    let args = cmdArgs(['--query', '-q', '--add', '--db', '--import', '--json', '--mcp', '--nuke', '--input-header', '--result-header', '--test', '--limit', '--category'])
    let query = args['--query'] || args['-q']
    if(args['--db']){
        config.DB_FILE = args['--db']
    }
    let category = ''
    if(args['--category']) {
        category = args['--category']
    }
    if(query){
        let numResults = 5
        if(args['--limit'] && parseInt(args['--limit'])) {
            numResults = parseInt(args['--limit'])
        }
        console.time('Search time')
        let result = await vectorSearch(query, category, numResults)
        console.timeEnd('Search time')
        console.log('Results:')
        console.log(JSON.stringify(result, null, 2))
    }else if(args['--add']){
        let [input, result, dataString] = args['--add'].split('|')
        if(!input || !result) {
            console.log('Usage:')
            return console.log(args._cmd + `--add 'input|result|{"foo":"bar"}'`)
        }
        let data = {}
        if(dataString) {
            try {data = JSON.parse(dataString)}catch(err) {}
        }
        let resp = await add(input, result, data, category)
        console.log(JSON.stringify(resp, null, 2))
    }else if(args['--remove']){
        let id = args['--remove']
        if(!id) return console.log('Please specify ID to remove')
        let resp = await remove(id, category)
        console.log(JSON.stringify(resp, null, 2))
    }else if(args['--nuke'] != undefined){
        nuke()
        console.log('Nuked.')
    }else if(args['--import']){
        await importFromCSVorTSV(args['--import'], args['--input-header'], args['--result-header'])
        console.log('Imported.')
    }else if(args['--json']){
        await importFromJSONStream(args['--json'])
        console.log('Imported.')
    }else if(args['--mcp'] != undefined){
        await mcp()
        console.log('MCP server running.')
    }else if(args['--test'] != undefined){
        console.log('Test: ', await test())
    }else{
        console.log('Usage:')
        console.log(args._cmd + ' --query "Foo Bar"')
        console.log("\n" + 'Options:')
        console.log('--query "SEARCH_STRING"                    - search')
        console.log('--limit 2                                  - limit number of results (used with --query)')
        console.log(`--add 'input|result|{"foo":"bar"}|categ'   - add data`)
        console.log(`--remove 'id'                              - remove data`)
        console.log(`--nuke                                     - destroy database`)
        console.log(`--mcp                                      - run as MCP server (experimental)`)
        console.log(`--db "FILE_NAME"                           - database file (SQLite)`)
        console.log(`--import "file.csv | file.tsv"             - import from CSV or TSV w/ columns: 1. input 2. result 3. and remaining columns are additional data`)
        console.log('--input-header "foo"                       - when used with --import designates specific header column as input')
        console.log('--result-header "bar"                      - when used with --import designates specific header column as result')
        console.log(`--json "FILE_NAME"                         - import from file which has one json object per line: {input:"", result:"", data:{}}`)
        console.log(`--category "CATEGORY"                      - specify category when adding data and to filter by when querying (defaults to empty string)`)
    }
}

if(isThisFileBeingRunViaCLI){
    runCLI()
}

// recall --nuke
// recall --import "test.tsv"
// recall --add 'The quick brown fox jumps over the lazy dog|Fox|{"foo":"bar"}'
// recall --query "Un animal saute par-dessus un autre animal"