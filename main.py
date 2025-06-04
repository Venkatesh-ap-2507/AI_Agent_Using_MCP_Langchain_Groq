from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from mcp_use import MCPAgent, MCPClient
from flask import Flask, request, jsonify, send_from_directory
import asyncio
import os
import re
import glob

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
    llm = ChatGroq(model="llama3-70b-8192", max_tokens=250, temperature=0.7)
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


def get_latest_generated_images():
    """Get the most recently generated images"""
    images_dir = 'generated_images'
    if not os.path.exists(images_dir):
        return []

    # Get all image files with their modification times
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif']:
        pattern = os.path.join(images_dir, ext)
        files = glob.glob(pattern)
        for file in files:
            image_files.append({
                'filename': os.path.basename(file),
                'mtime': os.path.getmtime(file)
            })

    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: x['mtime'], reverse=True)

    return [img['filename'] for img in image_files]


def extract_image_paths_from_response(response_text):
    """Extract image file paths from the agent's response"""
    # Common patterns for image file references
    patterns = [
        r'generated_images[/\\][\w\-_.]+\.(?:png|jpg|jpeg|gif)',
        r'image_\d+\.(?:png|jpg|jpeg|gif)',
        r'[\w\-_.]+\.(?:png|jpg|jpeg|gif)',
    ]

    found_images = []
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        found_images.extend(matches)

    return found_images


# Add route to serve generated images
@app.route('/generated_images/<filename>')
def serve_image(filename):
    """Serve images from the generated_images directory"""
    try:
        # Make sure the directory exists
        images_dir = os.path.join(os.getcwd(), 'generated_images')
        if not os.path.exists(images_dir):
            return jsonify({"error": "Images directory not found"}), 404

        # Check if file exists
        file_path = os.path.join(images_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Image not found"}), 404

        return send_from_directory('generated_images', filename)
    except Exception as e:
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500


@app.route('/latest_images')
def get_latest_images():
    """Get the latest generated images"""
    try:
        images = get_latest_generated_images()
        return jsonify({"images": images[:5]})  # Return latest 5 images
    except Exception as e:
        return jsonify({"error": f"Error getting images: {str(e)}"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Get images before processing
        images_before = set(get_latest_generated_images())

        # Initialize agent if not already done
        if global_agent is None:
            run_async_in_sync(initialize_agent())

        # Run the agent
        async def run_chat():
            response = await global_agent.run(user_input, thread_id="web_thread")
            return response

        response = run_async_in_sync(run_chat())

        # Get images after processing
        images_after = set(get_latest_generated_images())

        # Find newly generated images
        new_images = list(images_after - images_before)

        # Also check for images mentioned in the response
        mentioned_images = extract_image_paths_from_response(response)

        # Combine and deduplicate
        all_images = list(
            set(new_images + [os.path.basename(img) for img in mentioned_images]))

        return jsonify({
            "response": response,
            "images": all_images
        })

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
                margin-bottom: 15px;
                padding: 10px;
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
            .system-message {
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
                font-style: italic;
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
            .generated-image {
                max-width: 100%;
                max-height: 300px;
                height: auto;
                border-radius: 8px;
                margin: 10px 0;
                display: block;
                border: 2px solid #ddd;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .image-container {
                text-align: center;
                margin: 15px 0;
                background: white;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            .image-caption {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .loading {
                display: none;
                color: #666;
                font-style: italic;
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
                <div id="loading" class="loading">Processing...</div>
            </div>
            <script>
                async function sendMessage() {
                    const input = document.getElementById('user-input');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    addMessage(message, 'user');
                    input.value = '';
                    
                    // Show loading indicator
                    document.getElementById('loading').style.display = 'block';
                    
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ input: message })
                        });
                        
                        const data = await response.json();
                        
                        // Hide loading indicator
                        document.getElementById('loading').style.display = 'none';
                        
                        if (data.error) {
                            addMessage('Error: ' + data.error, 'assistant');
                        } else {
                            addMessage(data.response, 'assistant');
                            
                            // Display any new images
                            if (data.images && data.images.length > 0) {
                                displayImages(data.images);
                            }
                        }
                    } catch (error) {
                        document.getElementById('loading').style.display = 'none';
                        addMessage('Error: Failed to get response', 'assistant');
                    }
                }

                function displayImages(imageList) {
                    const output = document.getElementById('chat-output');
                    
                    imageList.forEach(imageName => {
                        const container = document.createElement('div');
                        container.className = 'image-container';
                        
                        const img = document.createElement('img');
                        img.src = '/generated_images/' + imageName;
                        img.className = 'generated-image';
                        img.alt = 'Generated image: ' + imageName;
                        
                        const caption = document.createElement('div');
                        caption.className = 'image-caption';
                        caption.textContent = '🖼️ Generated: ' + imageName;
                        
                        img.onload = function() {
                            console.log('Image loaded successfully:', imageName);
                        };
                        
                        img.onerror = function() {
                            console.error('Failed to load image:', imageName);
                            container.innerHTML = '<div style="color: #f44336; padding: 10px;">❌ Failed to load image: ' + imageName + '</div>';
                        };
                        
                        container.appendChild(img);
                        container.appendChild(caption);
                        output.appendChild(container);
                    });
                    
                    // Scroll to bottom to show new images
                    output.scrollTop = output.scrollHeight;
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
                    } else if (sender === 'system') {
                        messageDiv.className += ' system-message';
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
                
                // Load latest images on page load
                window.addEventListener('load', async function() {
                    try {
                        const response = await fetch('/latest_images');
                        const data = await response.json();
                        if (data.images && data.images.length > 0) {
                            addMessage('Recent images found:', 'system');
                            displayImages(data.images);
                        }
                    } catch (error) {
                        console.log('No recent images found');
                    }
                });
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

# # Global variables to store agent and client
# global_agent = None
# global_client = None


# async def initialize_agent():
#     """Initialize the agent once at startup"""
#     global global_agent, global_client

#     load_dotenv()

#     # Check if API key is set
#     if not os.getenv("GROQ_API_KEY"):
#         raise ValueError("GROQ_API_KEY not found in environment variables")

#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#     config_file = "browser_mcp.json"

#     global_client = MCPClient.from_config_file(config_file)
#     llm = ChatGroq(model="llama3-70b-8192", max_tokens=250, temperature=0.7)
#     global_agent = await MCPAgent.create(
#         llm=llm,
#         client=global_client,
#         max_steps=15,
#         memory_enabled=True
#     )

#     return global_agent, global_client


# def run_async_in_sync(coro):
#     """Helper function to run async code in sync context"""
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         return loop.run_until_complete(coro)
#     finally:
#         loop.close()


# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_input = request.json.get('input')
#         if not user_input:
#             return jsonify({"error": "No input provided"}), 400

#         # Initialize agent if not already done
#         if global_agent is None:
#             run_async_in_sync(initialize_agent())

#         # Run the agent
#         async def run_chat():
#             response = await global_agent.run(user_input, thread_id="web_thread")
#             return response

#         response = run_async_in_sync(run_chat())
#         return jsonify({"response": response})

#     except Exception as e:
#         return jsonify({"error": f"Error processing request: {str(e)}"}), 500


# @app.route('/clear', methods=['POST'])
# def clear():
#     try:
#         # Initialize agent if not already done
#         if global_agent is None:
#             run_async_in_sync(initialize_agent())

#         # Clear conversation history
#         async def clear_history():
#             await global_agent.clear_conversation_history(thread_id="web_thread")

#         run_async_in_sync(clear_history())
#         return jsonify({"message": "Conversation history cleared"})

#     except Exception as e:
#         return jsonify({"error": f"Error clearing history: {str(e)}"}), 500


# @app.route('/')
# def index():
#     """Serve the main HTML page"""
#     return """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>Multi-Tool Creative Agent</title>
#         <style>
#             body {
#                 font-family: Arial, sans-serif;
#                 margin: 20px;
#                 background-color: #f5f5f5;
#             }
#             #chat-container {
#                 max-width: 800px;
#                 margin: auto;
#                 background: white;
#                 border-radius: 10px;
#                 padding: 20px;
#                 box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#             }
#             #chat-output {
#                 border: 1px solid #ccc;
#                 padding: 15px;
#                 height: 400px;
#                 overflow-y: scroll;
#                 background: #fafafa;
#                 border-radius: 5px;
#                 margin-bottom: 10px;
#             }
#             .message {
#                 margin-bottom: 10px;
#                 padding: 8px;
#                 border-radius: 5px;
#             }
#             .user-message {
#                 background-color: #e3f2fd;
#                 border-left: 4px solid #2196f3;
#             }
#             .assistant-message {
#                 background-color: #f3e5f5;
#                 border-left: 4px solid #9c27b0;
#             }
#             #user-input {
#                 width: calc(100% - 120px);
#                 padding: 10px;
#                 margin-top: 10px;
#                 border: 1px solid #ccc;
#                 border-radius: 5px;
#             }
#             button {
#                 padding: 10px 15px;
#                 margin-left: 5px;
#                 background-color: #2196f3;
#                 color: white;
#                 border: none;
#                 border-radius: 5px;
#                 cursor: pointer;
#             }
#             button:hover {
#                 background-color: #1976d2;
#             }
#             .clear-btn {
#                 background-color: #f44336;
#             }
#             .clear-btn:hover {
#                 background-color: #d32f2f;
#             }
#         </style>
#     </head>
#         <body>
#             <div id="chat-container">
#                 <h2>🤖 Multi-Tool Creative Agent</h2>
#                 <p>I can help you with stories, image generation, web searches, and more!</p>
#                 <div id="chat-output"></div>
#                 <img src="/generated_images/image_1187.png" alt="Generated image">
#                 <div>
#                     <input type="text" id="user-input" placeholder="Type your message..." onkeypress="handleEnter(event)">
#                     <button onclick="sendMessage()">Send</button>
#                     <button onclick="clearHistory()" class="clear-btn">Clear</button>
#                 </div>
#             </div>
#             <script>
#                 async function sendMessage() {
#                     const input = document.getElementById('user-input');
#                     const message = input.value.trim();
#                     if (!message) return;

#                     addMessage(message, 'user');
#                     input.value = '';

#                     try {
#                         const response = await fetch('/chat', {
#                             method: 'POST',
#                             headers: { 'Content-Type': 'application/json' },
#                             body: JSON.stringify({ input: message })
#                         });

#                         const data = await response.json();
#                         if (data.error) {
#                             addMessage('Error: ' + data.error, 'assistant');
#                         } else {
#                             addMessage(data.response, 'assistant');
#                         }
#                     } catch (error) {
#                         addMessage('Error: Failed to get response', 'assistant');
#                     }
#                 }

#                 async function clearHistory() {
#                     try {
#                         const response = await fetch('/clear', { method: 'POST' });
#                         const data = await response.json();
#                         document.getElementById('chat-output').innerHTML = '';
#                         addMessage('Conversation history cleared.', 'system');
#                     } catch (error) {
#                         addMessage('Error clearing history', 'system');
#                     }
#                 }

#                 function addMessage(text, sender) {
#                     const output = document.getElementById('chat-output');
#                     const messageDiv = document.createElement('div');
#                     messageDiv.className = 'message';

#                     if (sender === 'user') {
#                         messageDiv.className += ' user-message';
#                         messageDiv.innerHTML = `<strong>You:</strong> ${text}`;
#                     } else if (sender === 'assistant') {
#                         messageDiv.className += ' assistant-message';
#                         messageDiv.innerHTML = `<strong>Assistant:</strong> ${text}`;
#                     } else {
#                         messageDiv.innerHTML = `<em>${text}</em>`;
#                     }

#                     output.appendChild(messageDiv);
#                     output.scrollTop = output.scrollHeight;
#                 }

#                 function handleEnter(event) {
#                     if (event.key === 'Enter') {
#                         sendMessage();
#                     }
#                 }
#             </script>
#         </body>
#         </html>
#         """


# async def run_memory_chat():
#     """CLI version of the chat"""
#     load_dotenv()

#     if not os.getenv("GROQ_API_KEY"):
#         print("Error: GROQ_API_KEY not found in environment variables")
#         return

#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#     config_file = "browser_mcp.json"

#     print("Initializing chat...")
#     client = MCPClient.from_config_file(config_file)
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
#     agent = await MCPAgent.create(llm=llm, client=client, max_steps=15, memory_enabled=True)

#     print("\n===== Multi-Tool Creative Agent =====")
#     print("Available tools:")
#     print("- Story writing: Ask me to write a story about any topic")
#     print("- Image generation: Ask me to generate or create images")
#     print("- Web search: Ask me to search for information")
#     print("- ASCII art: Ask me to create ASCII art")
#     print("\nType 'exit' or 'quit' to end the conversation")
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
#     if len(os.sys.argv) > 1 and os.sys.argv[1] == "web":
#         print("Starting web server...")
#         app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=True)
#     else:
#         print("Starting CLI mode...")
#         asyncio.run(run_memory_chat())
