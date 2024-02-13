from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import StructuredTool
import base64
import json
import random

import requests
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_json_chat_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langserve import add_routes
from pydantic import BaseModel
from langchain.adapters import openai as lc_openai
from starlette.responses import StreamingResponse

# Start it with -rpc
zoekt_addr = "http://localhost:6070"
openai_api_address = "http://192.168.0.169:5000/v1"
openai_api_key = "none"

query = "Get the code for ClusterController and make sure the 500 error codes are reflected in all the api annotations. Respond with a markdown diff block containing the entire change"


def search(query: str, get_context=True, num_result=10):
    """
    
    :param query: the query to search for. should be a single word, no spaces.
    :param get_context: 
    :param num_result: 
    :return: 
    """
    url = f"{zoekt_addr}/api/search"

    results = {}
    data = {
        "Q": query,
        # might be useful later
        # https://github.com/sourcegraph/zoekt/blob/main/grpc/protos/zoekt/webserver/v1/webserver.proto#L46
        "opts": {
            "total_max_match_count": num_result,
        },
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        response_json = json.loads(response.text)
        if (response_json is not None
                and "Result" in response_json
                and "Files" in response_json["Result"]
                and response_json["Result"]["Files"] is not None):
            for file in response_json["Result"]["Files"]:
                filename = file['FileName']

                current_file_context = []
                current_file_dict = {
                    "repository": file['Repository'],
                    "context": current_file_context
                }
                # TODO: clip the overall object if result is too long. (estimate tokens of entire result object)
                results[filename] = current_file_dict
                if get_context:
                    for line_match in file["LineMatches"]:
                        decoded_line = base64.b64decode(line_match["Line"]).decode('utf-8')
                        if len(decoded_line) > 1024:
                            decoded_line = decoded_line[:1024]
                            print(f"Truncating line from long output for file {filename}. New output {decoded_line}")
                        current_file_context.append(decoded_line)

    if len(results) == 0:
        results["invalid"] = f"adjust your query. it didn't return any results."
    print(f"search tool invoked, query {query}, result {results}")
    return results


def get_file(file_path: str):
    print(f"get file invoked, path {file_path}")
    url = f"{zoekt_addr}/api/search"
    data = {
        "Q": f"file:{file_path}",
        # might be useful later
        # https://github.com/sourcegraph/zoekt/blob/main/grpc/protos/zoekt/webserver/v1/webserver.proto#L46
        "opts": {
            "total_max_match_count": 1,
            "whole": True
        },
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        response_json = json.loads(response.text)
        return base64.b64decode(response_json["Result"]["Files"][0]["Content"]).decode('utf-8')
    else:
        return "File not found"


def get_tools():
    # class SearchInput(BaseModel):
    #    a: int = Field(description="first number")
    #    b: int = Field(description="second number")

    search_tool = StructuredTool.from_function(
        func=search,
        name="code_search",
        description="""search for code based on name. don't use natural language in your query. enter single words only, no spaces.""",
        # args_schema=SearchInput,
    )

    get_file_tool = StructuredTool.from_function(
        func=get_file,
        name="get_file",
        description="get a complete file. use the exact filename as returned in the search results previously. You can use this if the context provided in search wasn't enough. it will fail if you don't provide the correct filename",
    )
    return [search_tool, get_file_tool]


# def convert_intermediate_steps(intermediate_steps):
#    log = ""
#    for action, observation in intermediate_steps:
#        log += (
#            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}</tool_input><observation>{observation}</observation>"
#        )
#    return f"<intermediate_steps>{log}</intermediate_steps>"
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    print(f"returning log {log}, int steps: {intermediate_steps}")
    return log


from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class Payload(BaseModel):
    temperature: float
    stream: bool
    model: str
    messages: List[Message]
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float


class Choice(BaseModel):
    index: int
    finish_reason: Optional[str]
    message: dict
    delta: dict


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]


def main():
    tools = get_tools()
    model = ChatOpenAI(model="gpt-4", temperature=0.8, openai_api_key=openai_api_key,
                       openai_api_base=openai_api_address, streaming=True)

    # TODO probably use these refs, maybe try xml agent? idk
    # https://python.langchain.com/docs/expression_language/cookbook/agent
    # https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
    # https://python.langchain.com/docs/expression_language/cookbook/tools
    from langchain import hub

    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(model, tools, prompt)
    # prompt = hub.pull("hwchase17/react-chat-json")
    # agent = create_json_chat_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    @app.post("/v1/chat/completions")
    async def create_completion(payload: dict):
        transformed = ""
        payload = Payload.parse_obj(payload)
        for message in payload.messages:
            #if message.role == "system":
                #transformed += "### System prompt\nMake sure to use the provided tools when applicable.\n"
            if message.role == "user":  # Changed 'if' to 'elif' to ensure exclusive conditions
                transformed += "### User Message\n"
                transformed += message.content
            #else:
            #    print(f"role {message.role} unknown")

        print(f"message: {transformed}") It''

        async def generator():
            async for event in agent_executor.astream({"input": transformed}):
                print(f"received event: {event}")
                if "output" in event:
                    item = ChatCompletionChunk(
                        id="chatcmpl",
                        object="chat.completions.chunk",
                        created=1707817924,
                        model="idk",
                        choices=[
                            {
                                "index": 0,
                                "finish_reason": None,
                                "message": {"role": "assistant", "content": event["output"]},
                                "delta": {"role": "assistant", "content": event["output"]},
                            }
                        ],
                    )
                    yield json.dumps(jsonable_encoder(item))

        return StreamingResponse(generator(), media_type="application/json")

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # agent_executor.invoke({"input": system_msg.format(query=query)})

    # TODO tools new:
    #  * zoekt search: looks for keywords, find files, return their name and zoekt key
    #  * zoekt get: allows passing zoekt key to get file contents directly
    #  * filetree?
    #  * LSP integration with some externally hosted lsp? for finding usage etc.
    #  * some simple vector db, ideally without remote embedding function? something more universal?


if __name__ == "__main__":
    main()
