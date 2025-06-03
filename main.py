from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from mcp_use import MCPAgent, MCPClient
from flask import Flask, request, jsonify
import asyncio
import os

app = Flask(__name__)

# Global variables to store agent and client
global_agent = None
global_client = None


async def initialize_agent():
    """Initialize the agent once at startup"""
    global global_agent, global_client

    load_dotenv()

    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file = "browser_mcp.json"

    global_client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
    global_agent = await MCPAgent.create(
        llm=llm,
        client=global_client,
        max_steps=15,
        memory_enabled=True
    )

    return global_agent, global_client


def run_async_in_sync(coro):
    """Helper function to run async code in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Initialize agent if not already done
        if global_agent is None:
            run_async_in_sync(initialize_agent())

        # Run the agent
        async def run_chat():
            response = await global_agent.run(user_input, thread_id="web_thread")
            return response

        response = run_async_in_sync(run_chat())
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


@app.route('/clear', methods=['POST'])
def clear():
    try:
        # Initialize agent if not already done
        if global_agent is None:
            run_async_in_sync(initialize_agent())

        # Clear conversation history
        async def clear_history():
            await global_agent.clear_conversation_history(thread_id="web_thread")

        run_async_in_sync(clear_history())
        return jsonify({"message": "Conversation history cleared"})

    except Exception as e:
        return jsonify({"error": f"Error clearing history: {str(e)}"}), 500


@app.route('/')
def index():
    """Serve the main HTML page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Multi-Tool Creative Agent</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            #chat-container { 
                max-width: 800px; 
                margin: auto; 
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            #chat-output { 
                border: 1px solid #ccc; 
                padding: 15px; 
                height: 400px; 
                overflow-y: scroll; 
                background: #fafafa;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .message {
                margin-bottom: 10px;
                padding: 8px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
                border-left: 4px solid #2196f3;
            }
            .assistant-message {
                background-color: #f3e5f5;
                border-left: 4px solid #9c27b0;
            }
            #user-input { 
                width: calc(100% - 120px); 
                padding: 10px; 
                margin-top: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button { 
                padding: 10px 15px; 
                margin-left: 5px;
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #1976d2;
            }
            .clear-btn {
                background-color: #f44336;
            }
            .clear-btn:hover {
                background-color: #d32f2f;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h2>🤖 Multi-Tool Creative Agent</h2>
            <p>I can help you with stories, image generation, web searches, and more!</p>
            <div id="chat-output"></div>
            <div>
                <input type="text" id="user-input" placeholder="Type your message..." onkeypress="handleEnter(event)">
                <button onclick="sendMessage()">Send</button>
                <button onclick="clearHistory()" class="clear-btn">Clear</button>
            </div>
        </div>
        <script>
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: message })
                    });
                    
                    const data = await response.json();
                    if (data.error) {
                        addMessage('Error: ' + data.error, 'assistant');
                    } else {
                        addMessage(data.response, 'assistant');
                    }
                } catch (error) {
                    addMessage('Error: Failed to get response', 'assistant');
                }
            }

            async function clearHistory() {
                try {
                    const response = await fetch('/clear', { method: 'POST' });
                    const data = await response.json();
                    document.getElementById('chat-output').innerHTML = '';
                    addMessage('Conversation history cleared.', 'system');
                } catch (error) {
                    addMessage('Error clearing history', 'system');
                }
            }

            function addMessage(text, sender) {
                const output = document.getElementById('chat-output');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                
                if (sender === 'user') {
                    messageDiv.className += ' user-message';
                    messageDiv.innerHTML = `<strong>You:</strong> ${text}`;
                } else if (sender === 'assistant') {
                    messageDiv.className += ' assistant-message';
                    messageDiv.innerHTML = `<strong>Assistant:</strong> ${text}`;
                } else {
                    messageDiv.innerHTML = `<em>${text}</em>`;
                }
                
                output.appendChild(messageDiv);
                output.scrollTop = output.scrollHeight;
            }
            
            function handleEnter(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
        </script>
    </body>
    </html>
    """


async def run_memory_chat():
    """CLI version of the chat"""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment variables")
        return

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file = "browser_mcp.json"

    print("Initializing chat...")
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
    agent = await MCPAgent.create(llm=llm, client=client, max_steps=15, memory_enabled=True)

    print("\n===== Multi-Tool Creative Agent =====")
    print("Available tools:")
    print("- Story writing: Ask me to write a story about any topic")
    print("- Image generation: Ask me to generate or create images")
    print("- Web search: Ask me to search for information")
    print("- ASCII art: Ask me to create ASCII art")
    print("\nType 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("=====================================\n")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break
            if user_input.lower() == "clear":
                await agent.clear_conversation_history(thread_id="cli_thread")
                print("Conversation history cleared.")
                continue

            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input, thread_id="cli_thread")
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
    finally:
        if client:
            await client.close_all_sessions()

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "web":
        print("Starting web server...")
        app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=True)
    else:
        print("Starting CLI mode...")
        asyncio.run(run_memory_chat())


# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# from mcp_use import MCPAgent, MCPClient
# from flask import Flask, request, jsonify
# import asyncio
# import os

# app = Flask(__name__)


# async def create_agent():
#     load_dotenv()
#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#     config_file = "browser_mcp.json"
#     client = MCPClient.from_config_file(config_file)
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
#     agent = await MCPAgent.create(llm=llm, client=client, max_steps=15, memory_enabled=True)
#     return agent, client


# @app.route('/chat', methods=['POST'])
# async def chat():
#     user_input = request.json.get('input')
#     if not user_input:
#         return jsonify({"error": "No input provided"}), 400
#     agent, client = await create_agent()
#     try:
#         response = await agent.run(user_input, thread_id="web_thread")
#         return jsonify({"response": response})
#     finally:
#         await client.close_all_sessions()


# @app.route('/clear', methods=['POST'])
# async def clear():
#     agent, client = await create_agent()
#     try:
#         await agent.clear_conversation_history(thread_id="web_thread")
#         return jsonify({"response": "Conversation history cleared"})
#     finally:
#         await client.close_all_sessions()


# async def run_memory_chat():
#     load_dotenv()
#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#     config_file = "browser_mcp.json"
#     print("Initializing chat...")
#     client = MCPClient.from_config_file(config_file)
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
#     agent = await MCPAgent.create(llm=llm, client=client, max_steps=15, memory_enabled=True)

#     print("\n===== Multi-Tool Creative Agent =====")
#     print("Type 'exit' or 'quit' to end the conversation")
#     print("Type 'clear' to clear conversation history")
#     print("=====================================\n")

#     try:
#         while True:
#             user_input = input("\nYou: ")
#             if user_input.lower() in ["exit", "quit"]:
#                 print("Ending conversation...")
#                 break
#             if user_input.lower() == "clear":
#                 await agent.clear_conversation_history(thread_id="cli_thread")
#                 print("Conversation history cleared.")
#                 continue
#             print("\nAssistant: ", end="", flush=True)
#             try:
#                 response = await agent.run(user_input, thread_id="cli_thread")
#                 print(response)
#             except Exception as e:
#                 print(f"\nError: {e}")
#     finally:
#         if client:
#             await client.close_all_sessions()

# if __name__ == "__main__":
#     if os.getenv("FLASK_ENV") == "development":
#         app.run(port=int(os.getenv("PORT", 5000)))
#     else:
#         asyncio.run(run_memory_chat())
