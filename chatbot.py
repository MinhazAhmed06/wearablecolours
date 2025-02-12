import os
from dotenv import load_dotenv
from google import genai

import hexcodes

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

colours = hexcodes.hexcodes('image.jpg')

skin = colours['skin']
iris = colours['iris']
hair = colours['hair']

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[f"Please, Act like you are a professional with colours. Suppose my skin colour is:{skin}, eye colour is:{iris} and hair colour is{hair}. Which skin tone pallete am I? In terms of spring, winter, summer, fall. Also add which colours will look good on me and which will not."])
print(response.text)


