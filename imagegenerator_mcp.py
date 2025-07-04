from fastmcp import FastMCP
import requests
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("imagegenerator")


@mcp.tool()
async def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Generate an image using a free API service"""
    try:
        # Using Pollinations AI (free image generation API)
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}?width={width}&height={height}"

        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Save image temporarily and return path or base64
            image = Image.open(BytesIO(response.content))

            # Create images directory if it doesn't exist
            os.makedirs("generated_images", exist_ok=True)

            # Save image
            filename = f"generated_images/image_{hash(prompt) % 10000}.png"
            image.save(filename)

            return f"Image generated successfully and saved as: {filename}\nPrompt: {prompt}"
        else:
            return f"Failed to generate image. Status code: {response.status_code}"

    except Exception as e:
        return f"Error generating image: {str(e)}"


@mcp.tool()
async def create_ascii_art(text: str) -> str:
    """Create simple ASCII art from text"""
    try:
        ascii_chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]

        # Simple ASCII art generation
        lines = []
        for char in text.upper():
            if char == ' ':
                lines.append("     ")  # 5 spaces for each character
            elif char.isalpha():
                # Create a simple pattern for each letter
                patterns = {
                    'A': ["  #  ", " # # ", "#####", "#   #", "#   #"],
                    'B': ["#### ", "#   #", "#### ", "#   #", "#### "],
                    'C': [" ####", "#    ", "#    ", "#    ", " ####"],
                    'D': ["#### ", "#   #", "#   #", "#   #", "#### "],
                    'E': ["#####", "#    ", "###  ", "#    ", "#####"],
                    'F': ["#####", "#    ", "###  ", "#    ", "#    "],
                    'G': [" ####", "#    ", "# ###", "#   #", " ####"],
                    'H': ["#   #", "#   #", "#####", "#   #", "#   #"],
                    'I': ["#####", "  #  ", "  #  ", "  #  ", "#####"],
                    'J': ["#####", "    #", "    #", "#   #", " ### "],
                }

                if char in patterns:
                    if not lines:
                        lines = patterns[char][:]
                    else:
                        for i in range(5):
                            lines[i] += " " + patterns[char][i]
                else:
                    # Default pattern for unknown characters
                    default = ["#####", "#   #", "#   #", "#   #", "#####"]
                    if not lines:
                        lines = default[:]
                    else:
                        for i in range(5):
                            lines[i] += " " + default[i]

        return "ASCII Art:\n" + "\n".join(lines) if lines else "No valid characters to convert"

    except Exception as e:
        return f"Error creating ASCII art: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")

# from fastmcp import FastMCP
# from craiyon import Craiyon
# import os
# import base64
# from datetime import datetime
# from PIL import Image
# import io
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# mcp = FastMCP("imagegenerator")
# generator = Craiyon()


# @mcp.tool()
# async def generate_image(description: str) -> str:
#     logger.debug("Generating image for description: %s", description)
#     try:
#         # Generate image using Craiyon
#         result = generator.generate(description)
#         if not result.images:
#             logger.error("No images generated by Craiyon")
#             return "Error: No images generated."

#         # Create output directory
#         output_dir = "generated_images"
#         os.makedirs(output_dir, exist_ok=True)

#         # Use timestamp for unique file name
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         image_path = os.path.join(output_dir, f"image_{timestamp}.png")

#         # Decode base64 image and save
#         image_data = base64.b64decode(result.images[0])
#         image = Image.open(io.BytesIO(image_data))
#         image.save(image_path, "PNG")

#         logger.info("Image saved at: %s", image_path)
#         return f"Generated image saved at: {image_path}"
#     except Exception as e:
#         logger.error("Image generation failed: %s", str(e))
#         return f"Image generation failed: {str(e)}"

# if __name__ == "__main__":
#     logger.info("Starting imagegenerator MCP server")
#     mcp.run(transport="stdio")
