import re
import base64
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage


class ChatInterface:
    """
    A chat interface for interacting with a medical AI agent through Gradio.

    Handles file uploads, message processing, and chat history management.
    Supports both regular image files and DICOM medical imaging files.
    """

    def __init__(self, agent, tools_dict):
        """
        Initialize the chat interface.

        Args:
            agent: The medical AI agent to handle requests
            tools_dict (dict): Dictionary of available tools for image processing
        """
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Keep two paths:
        # - original_file_path: what tools/LLM should reason over (can be .dcm)
        # - display_file_path: what Gradio should render (always displayable image)
        self.original_file_path = None
        self.display_file_path = None

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths.

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            str: Display path for UI, or None if no file uploaded
        """
        if not file_path:
            return None

        source = Path(file_path)
        timestamp = int(time.time())

        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)

        # Handle DICOM conversion for display only
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved_path)

        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        """
        Add a new message to the chat history.

        Args:
            message (str): Text message to add
            display_image (str): Path to image being displayed
            history (List[dict]): Current chat history

        Returns:
            Tuple[List[dict], gr.Textbox]: Updated history and textbox component
        """
        image_path = self.original_file_path or display_image
        if image_path is not None:
            # UI-only history item so the chat window can show the user image.
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator[Tuple[List[ChatMessage], Optional[str], str], None]:
        """
        Process a message and generate responses.

        Args:
            message (str): User message to process
            display_image (Optional[str]): Path to currently displayed image
            chat_history (List[ChatMessage]): Current chat history

        Yields:
            Tuple[List[ChatMessage], Optional[str], str]: Updated chat history, display path, and empty string
        """
        chat_history = chat_history or []

        # A stable thread_id lets LangGraph's checkpointer restore prior state for this chat.
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self.original_file_path or display_image

        if image_path is not None:
            # Plain text hint used by tool-oriented prompts (image path for local tools).
            messages.append({"role": "user", "content": f"image_path: {image_path}"})

            # Multimodal payload for model-side image understanding.
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        }
                    ],
                }
            )

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        try:
            # Stream node outputs from Agent.workflow (defined in medrax/agent/agent.py).
            # Per turn: process -> (optional execute) -> process ... until no more tool calls.
            for event in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": self.current_thread_id}},
            ):
                if isinstance(event, dict):
                    if "process" in event:
                        content = event["process"]["messages"][-1].content
                        # "process" returns an AIMessage produced by model.invoke(...).
                        if content:
                            content = re.sub(r"temp/[^\s]*", "", content)
                            chat_history.append(ChatMessage(role="assistant", content=content))
                            yield chat_history, self.display_file_path, ""

                    elif "execute" in event:
                        # "execute" returns ToolMessage objects for each invoked tool.
                        for message in event["execute"]["messages"]:
                            tool_name = message.name
                            # ToolMessage.content is serialized text from Agent.execute_tools.
                            # This UI expects a list-like payload and renders the first item.
                            tool_result = eval(message.content)[0] # convert string from this line `content=str(result),` in agent .py into a python obejct ( dict or list ...)

                            if tool_result:
                                metadata = {"title": f"🖼️ Image from tool: {tool_name}"}
                                formatted_result = " ".join(
                                    line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                                metadata["description"] = formatted_result
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    )
                                )

                            # image_visualizer can return a new file to display in the UI.
                            if tool_name == "image_visualizer":
                                self.display_file_path = tool_result["image_path"]
                                chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        # content=gr.Image(value=self.display_file_path),
                                        content={"path": self.display_file_path},
                                    )
                                )

                            yield chat_history, self.display_file_path, ""

        except Exception as e:
            chat_history.append(
                ChatMessage(
                    role="assistant", content=f"❌ Error: {str(e)}", metadata={"title": "Error"}
                )
            )
            yield chat_history, self.display_file_path


def create_demo(agent, tools_dict):
    """
    Create a Gradio demo interface for the medical AI agent.

    Args:
        agent: The medical AI agent to handle requests
        tools_dict (dict): Dictionary of available tools for image processing

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    # Shared runtime state used by upload, chat, and thread callbacks.
    interface = ChatInterface(agent, tools_dict)

    # Top-level Gradio layout container.
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
            # 🏥 MedRAX
            Medical Reasoning Agent for Chest X-ray
            """
            )

            with gr.Row():
                # Left panel: conversation history and text input.
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        # Gradio 6 uses message format by default; no "type" kwarg.
                        label="Agent",
                        avatar_images=(
                            None,
                            "assets/medrax_logo.jpg",
                        ),
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the X-ray...",
                                container=False,
                            )

                # Right panel: image preview and file controls.
                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image", type="filepath", height=700, container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        # Event handlers
        def clear_chat():
            # Reset uploaded file state and clear chat/image widgets.
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread():
            # New thread_id starts a fresh checkpointed conversation.
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file):
            # Save upload and return a displayable image path.
            return interface.handle_upload(file.name)

        # Submit flow: add user message -> stream agent response -> re-enable textbox.
        chat_msg = txt.submit(
            interface.add_message, inputs=[txt, image_display, chatbot], outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt],
        )
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [txt])

        # Both upload buttons share the same upload handler.
        upload_button.upload(handle_file_upload, inputs=upload_button, outputs=image_display)

        dicom_upload.upload(handle_file_upload, inputs=dicom_upload, outputs=image_display)

        # Session controls.
        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo
