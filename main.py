import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file) # NOTE: [done]
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        # Classify CXR pathologies (18-condition classifier)
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device), # NOTE: [done]
        # Segment anatomical structures and return metrics
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device), # NOTE: [done]
        # LLaVA-Med visual QA over medical images
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True), # NOTE: [done]
        # CheXagent chest X-ray QA and interpretation (single/multi-image prompts)
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device), # NOTE: [done]
        # Generate findings and impression text from a chest X-ray image
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool( # NOTE: [done]
            cache_dir=model_dir, device=device
        ),
        # Ground (link text to specific regions of image) a medical phrase to chest X-ray regions with bounding boxes
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(   # NOTE: [done]
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device 
        ),
        # Generate synthetic chest X-ray images from text prompts
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool( # NOTE: [done]
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        # Surface an image path for UI display with optional title/description metadata
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),# NOTE: [done]
        # Convert DICOM files to standard image format and extract key metadata
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir), # NOTE: [done]
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]() # instantiate the tool with () i.e ( execute the lambda function)

    # Stores LangGraph checkpoints in RAM and retrieves them by thread_id,
    # so chat/workflow state can continue across turns (until process restart).
    checkpointer = MemorySaver()  # NOTE: [done]
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs) # NOTE: [done]
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        "LlavaMedTool",
        "XRayPhraseGroundingTool",
        "ChestXRayGeneratorTool",
    ]

    # Collect the ENV variables
    openai_kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/model-weights",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cpu",  # Change this to the device you want to use
        model="gpt-4o",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="127.0.0.1", server_port=8585, share=True)
