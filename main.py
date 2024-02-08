import os
import logging
from repopilot import RepoPilot
from langchain.callbacks.manager import get_openai_callback
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger('codetext').setLevel(logging.WARNING)
logging.getLogger('repopilot').setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("multilspy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Start!")
    api_key = os.environ.get("OPENAI_API_KEY")

    if (os.environ.get("OPENAI_API") is not None
            and os.environ.get("ZOEKT_API") is not None):
        openai_api = os.environ.get("OPENAI_API")
        zoekt_api = os.environ.get("ZOEKT_API")
        language = os.environ.get("LANGUAGE")
        pwd = os.environ.get("PWD")
        logger.info("Running in non-interactive mode")
        logger.info(f"Using OpenAI API at {openai_api}, zoekt API at {zoekt_api}, using pwd {pwd} as root dir")
        pilot = RepoPilot(repo_path=pwd, commit=None, openai_api_key=api_key,
                          language=language, clone_dir="data/repos",
                          zoekt_address=zoekt_api, openai_api_address=openai_api)
        #pilot.query_codebase()
        app = FastAPI(
            title="LangChain Server",
            version="1.0",
            description="A simple api server using Langchain's Runnable interfaces",
        )

        add_routes(
            app,
            pilot.get_system(),
            path="/openai",
        )
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        repo = input("Please provide a valid folder path or GitHub URL: ")
        commit = input("Please provide a commit: (default: HEAD if enter)")
        language = input("Please provide a programming language: (default: python if enter)")
        question = input("Please provide a question: ")
        pilot = RepoPilot(repo, commit=commit, openai_api_key=api_key, language=language, clone_dir="data/repos")
        logger.info("Setup done!")

        with get_openai_callback() as cb:
            pilot.query_codebase(question)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
