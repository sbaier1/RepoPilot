from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
import requests
import base64
import json
import re

# Start it with -rpc
zoekt_addr = "http://localhost:6070"
openai_api_address = "http://192.168.178.28:5000/v1"
openai_api_key = "none"

query = "Can you explain what the console-backend in the hivemq-cloud repo does? look up code if necessary"

def search(name, get_context=True, num_result=10):
    """
    Performs a search on the Zoekt server.

    Args:
        name (string): A list of names to search for.
        num_result (int, optional): The number of search results to retrieve. Defaults to 2.

    Returns:
        dict: A dictionary containing: filename (key) -> repo, file content (context, list)

    """
    print(f"search tool invoked, name {name}")
    url = f"{zoekt_addr}/api/search"

    results = {}
    data = {
        "Q": name,
        # might be useful later
        # https://github.com/sourcegraph/zoekt/blob/main/grpc/protos/zoekt/webserver/v1/webserver.proto#L46
        "opts": {
            "total_max_match_count": num_result,
        },
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        response_json = json.loads(response.text)
        for file in response_json["Result"]["Files"]:
            filename = file['FileName']

            current_file_context = []
            current_file_dict = {
                "repository": file['Repository'],
                "context": current_file_context
            }
            results[filename] = current_file_dict
            if get_context:
                for line_match in file["LineMatches"]:
                    # TODO: clip the context if result is too long. (estimate tokens)
                    decoded_line = base64.b64decode(line_match["Line"]).decode('utf-8')
                    current_file_context.append(decoded_line)

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
        base64.b64decode(response_json["Result"]["Files"][0]["Content"]).decode('utf-8')
    else:
        return "File not found"


def get_tools():
    # class SearchInput(BaseModel):
    #    a: int = Field(description="first number")
    #    b: int = Field(description="second number")

    search_tool = StructuredTool.from_function(
        func=search,
        name="code_search",
        description="search for code based on name or context",
        # args_schema=SearchInput,
        return_direct=True,
    )

    get_file_tool = StructuredTool.from_function(
        func=get_file,
        name="get_file",
        description="get a complete file. use the exact filename as returned in the search results previously. You can use this if the context provided in search wasn't enough. it will fail if you don't provide the correct filename",
        return_direct=True,
    )
    return [search_tool, get_file_tool]


def main():
    tools = get_tools()
    tool_strings = []
    for tool in tools:
        args_schema = re.sub("}", "}}", re.sub("{", "{{", str(tool.args)))
        tool_strings.append(f"- {tool.name}: {tool.description}, args: {args_schema}")
    formatted_tools = "\n".join(tool_strings)
    system_msg = \
        f"""You are a coding assistant. Make sure to think step by step first, think out loud.
When responding with code, use markdown syntax.
You can use various tools to help you achieve the task:
{formatted_tools}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", query)
    ])

    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai_api_key,
                       openai_api_base=openai_api_address)
    output_parser = StrOutputParser()

    chain = prompt | model.bind_tools(tools) | output_parser

    print(chain.invoke({}))

    # TODO tools new:
    #  * zoekt search: looks for keywords, find files, return their name and zoekt key
    #  * zoekt get: allows passing zoekt key to get file contents directly
    #  * filetree?
    #  * LSP integration with some externally hosted lsp? for finding usage etc.
    #  * some simple vector db, ideally without remote embedding function? something more universal?


if __name__ == "__main__":
    main()
