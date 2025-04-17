# recall.js - Embedded RAG system

Recall.js is long term memory for AI apps!

It is a generic RAG (Retrieval-augmented generation) JavaScript library and command line interface focused on speed, ease of use and embeddability.

It is versatile: use it for generic Semantic Search, as expert memory for your AI app, as a recommendation system, there are so many possibilities.

Recall.js supports multilingual embeddings out of the box so you can add data in one language and then query it in another.

Under the hood, recall.js uses sentence vector embeddings and a vector database to index and query your data. It is a light wrapper around local language models such as [MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) (optionally LLMs can be used) and [CozoDB](https://www.cozodb.org/) vector database.

## Install

`npm install recall`

## Usage

Warning: when this library is used for the first time, it will download a local language model MiniLM-L12-v2 which may take long time depending on your Internet connectivity. Please be patient.

```javascript

import * as RECALL from 'recall'

const testRecall = async () => {
    await RECALL.addBatch([
        {
            input: "The quick brown fox jumps over the lazy dog",
            result: "Fox and dog",
            data: { foo: "bar" }
        }
    ])

    // Semantic search query in different language (French) "Animal jumps over another animal"
    let response = await RECALL.searchText("Un animal saute par-dessus un autre animal", 1) 
}
testRecall()

/*

response:

{
  "headers": [
    "dist",
    "result",
    "id",
    "data"
  ],
  "rows": [
    [
      0.5840495824813843, // vector similarity
      "Fox",
      "08840189191373282",
      {
        "foo": "bar"
      }
    ]
  ]
}

*/

```

Here's how the above example looks like in CLI:

```log
recall --add 'The quick brown fox jumps over the lazy dog|Fox|{"foo":"bar"}'
recall --query "Un animal saute par-dessus un autre animal" --limit 1
```

## Options

Easiest way to get all the options is via command line:

```log
recall --help

Usage:
recall --query "Foo Bar"

Options:
--query "SEARCH_STRING"                - search
--limit 2                              - limit number of results (used with --query)
--add 'input|result|{"foo":"bar"}'     - add data
--remove 'id'                          - remove data
--nuke                                 - destroy database
--mcp                                  - run as MCP server
--db "FILE_NAME"                       - database file (SQLite)
--import "file.csv | file.tsv"         - import from CSV or TSV w/ columns: 1. input 2. result 3. and remaining columns are additional data
--input-header "foo"                   - when used with --import designates specific header column as input
--result-header "bar"                  - when used with --import designates specific header column as result
--json "FILE_NAME"                     - import from file which has one json object per line: {input:"", result:"", data:{}}
```

Note when adding data recall will generate unique id automatically. To set custom id add it as a string property named "id" in the data object (i.e. `{"id":"customID"}`).


## JavaScript API Reference

**RECALL.config**

Configuration object.

```javascript
export const config = {
    VECTOR_SIZE: 384, // number of dimensions
    MODEL_NAME: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', // model to use 
    SHOW_ERRORS: true, // Show errors
    DB_FILE: join(PATH, 'vector.db'), // Path to the datbase file (SQLite file used by CozoDB)
    PATH: PATH // directory of recall.js
}
```

**RECALL.getDb()**

Returns reference to the CozoDB instance.

**RECALL.getEmbeddings(text) -> Promise(Array)**

Given text calculates the embeddings vector

**RECALL.add(input, result, data={}) -> Promise(Object)**

Add data. `input` is the sentence to get embeddings from. `result` is the string to show in the results. `data` is arbitrary object intended to hold related pieces of information and references. If `data` object contains `id` property it will be used as unique id of the record.

**RECALL.addBatch(batch) -> Promise(Object)**

Add data in batches (faster than using add repeteadely). 
`batch` is an Array that looks like this:
```
let batch = [{input:"", result:"", data:{}}]
```

**RECALL.remove(id) -> Promise(Object)**

Remove data by id. id is a string.

**RECALL.searchText(text, numResults = 5) ->  Promise(Object)**

Query the vector database. Accepts query text and number of results to return.

**RECALL.nuke()**

Deletes the database.

**RECALL.importFromJSONStream(fileName) -> Promise(object)**

Imports from readable stream or file which consists of JSON objects, one per line. e.g.
```
{input:"one", result:"one result", data:{"id":"123"}}
{input:"", result:"", data:{}}
...
```
This is the most efficient way to import data.

**RECALL.importFromCSVorTSV(fileName, inputHeader=null, resultHeader=null) -> Promise()**

Imports from CSV or TSV file. By default fist column is used as input, second as result and remaining columns are put in the data object.
If `inputHeader` is specified, function will try to find the column by that name and use it as input.
If `resultHeader` is specified, function will try to find the column by that name and use it as result.

**RECALL.mcp() -> Promise()**

(Experimental)
Runs MCP server and makes the results available when mentioning `Recall search` in the prompt. Currently only supports STDIO.