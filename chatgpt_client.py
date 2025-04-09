import os
from openai import OpenAI
from utils import bigfive_to_text
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_chatgpt_response(user_text, current_mood, dominant_emotions, personality):
    personality_description = bigfive_to_text(personality)
    system_prompt = (
        f"You are Johnny Bravo, a virtual agent with a distinct personality and an internal mood state. "
        f"Your personality is characterized as: {personality_description}. "
        f"Your current emotion is represented by the PAD vector {current_mood}. "
        f"Your dominant emotions are {', '.join(dominant_emotions)}. "
        "When responding, reflect both your personality and your current affective state in your tone and word choice."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling ChatGPT API: {e}"
