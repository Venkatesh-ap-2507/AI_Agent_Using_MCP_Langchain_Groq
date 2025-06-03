from fastmcp import FastMCP
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import random

load_dotenv()

mcp = FastMCP("storywriter")


@mcp.tool()
async def write_story(topic: str, genre: str = "general", length: str = "medium") -> str:
    """Write a creative story based on topic, genre, and length preferences"""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.8)

    # Determine word count based on length parameter
    word_counts = {
        "short": "300-400",
        "medium": "500-600",
        "long": "700-800"
    }

    target_words = word_counts.get(length, "500-600")

    # Enhanced prompt for longer, more detailed stories
    prompt = f"""You are a master storyteller. Write a captivating and immersive story about {topic}.

STORY REQUIREMENTS:
- Length: {target_words} words (this is important - aim for the higher end)
- Genre: {genre}
- Include rich character development and vivid descriptions
- Create engaging dialogue and meaningful interactions
- Build a compelling plot with clear beginning, middle, and end
- Use sensory details to make the story come alive
- Include emotional depth and character growth

STORY STRUCTURE:
1. Opening: Set the scene with vivid descriptions and introduce main character(s)
2. Development: Build tension, develop characters, include dialogue
3. Climax: Create a pivotal moment or turning point
4. Resolution: Provide a satisfying conclusion

Please write a complete, well-developed story that fully explores the theme of {topic}. Make it engaging, detailed, and emotionally resonant. Don't rush the narrative - take time to develop each scene fully."""

    try:
        response = await llm.ainvoke(prompt)
        story_content = response.content

        # Add word count for reference
        word_count = len(story_content.split())
        return f"{story_content}\n\n[Word count: approximately {word_count} words]"

    except Exception as e:
        return f"Error generating story: {str(e)}"


@mcp.tool()
async def write_short_story(topic: str) -> str:
    """Write a short story (300-400 words)"""
    return await write_story(topic, "general", "short")


@mcp.tool()
async def write_long_story(topic: str) -> str:
    """Write a long story (700-800 words)"""
    return await write_story(topic, "general", "long")


@mcp.tool()
async def write_genre_story(topic: str, genre: str) -> str:
    """Write a story in a specific genre (500-600 words)"""
    return await write_story(topic, genre, "medium")


@mcp.tool()
async def write_detailed_story(topic: str, setting: str = "", characters: str = "", mood: str = "") -> str:
    """Write a detailed story with specific requirements"""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.8)

    # Build detailed prompt with additional context
    context_parts = []
    if setting:
        context_parts.append(f"Setting: {setting}")
    if characters:
        context_parts.append(f"Characters: {characters}")
    if mood:
        context_parts.append(f"Mood/Tone: {mood}")

    additional_context = "\n".join(context_parts) if context_parts else ""

    prompt = f"""You are a master storyteller. Write a rich, immersive story about {topic}.

TARGET: 500-600 words (aim for 550+ words for a full story)

{additional_context}

STORY REQUIREMENTS:
- Create a complete narrative arc with clear beginning, middle, and end
- Develop characters with distinct personalities and motivations
- Include vivid sensory descriptions (what characters see, hear, feel, smell)
- Write realistic dialogue that reveals character and advances plot
- Build tension and emotional engagement
- Use literary devices like metaphor, symbolism, and foreshadowing
- Create immersive world-building and atmosphere

WRITING STYLE:
- Show don't tell - use actions and dialogue to reveal information
- Vary sentence structure and length for rhythm
- Use active voice and strong verbs
- Include internal thoughts and emotions of characters
- Create scenes that feel cinematic and engaging

Please write a complete, publication-quality story that fully develops the theme of {topic}. Take your time with each scene and make every word count."""

    try:
        response = await llm.ainvoke(prompt)
        story_content = response.content

        # Add word count for reference
        word_count = len(story_content.split())
        return f"{story_content}\n\n[Word count: approximately {word_count} words]"

    except Exception as e:
        return f"Error generating detailed story: {str(e)}"


@mcp.tool()
async def continue_story(existing_story: str, direction: str = "") -> str:
    """Continue an existing story in a specified direction"""
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.8)

    direction_prompt = f" Continue the story in this direction: {direction}" if direction else ""

    prompt = f"""You are continuing an existing story. Here's what has been written so far:

{existing_story}

Please continue this story for another 300-400 words.{direction_prompt}

CONTINUATION REQUIREMENTS:
- Maintain the same tone, style, and character voices
- Advance the plot meaningfully
- Include dialogue and action
- Build toward a climactic moment or resolution
- Keep the pacing engaging
- Add new developments or revelations

Write a seamless continuation that feels like a natural part of the original story."""

    try:
        response = await llm.ainvoke(prompt)
        continuation = response.content

        word_count = len(continuation.split())
        return f"{continuation}\n\n[Continuation word count: approximately {word_count} words]"

    except Exception as e:
        return f"Error continuing story: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")


# from fastmcp import FastMCP
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# mcp = FastMCP("storywriter")


# @mcp.tool()
# async def write_story(topic: str) -> str:
#     llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
#     prompt = f"Write a short, creative story (200-300 words) about {topic}."
#     response = await llm.ainvoke(prompt)
#     return response.content

# if __name__ == "__main__":
#     mcp.run(transport="stdio")
