from openai import OpenAI
import textwrap
import requests
with open('openai_api_key.txt', 'r') as f:
    api_key = f.read()

client = OpenAI(api_key=api_key)

def process_response(response, file_name=None):
    file_name = file_name or 'dalle3_image.jpg'
    for i, d in enumerate(response.data):
        display_data(d, file_name= f"{i}_{file_name}")

def display_data(data, file_name=None):
    print("Revised prompt:")
    print(textwrap.fill(data.revised_prompt, width=80))
    print("Image url:")
    print(data.url)
    with open(f"dalle_images/{file_name}", 'wb') as f:
        f.write(requests.get(data.url).content)
    print(f"Image saved to {file_name}")


if __name__ == '__main__':

    query_args = {
        "model": "dall-e-3",
        "size": "1024x1024",
        "n": 1,
    }
    query_args["quality"] = "standard"


    query_args["prompt"] = "cosmic wallpaper with galaxies, stars, and planet to put on ceiling. Dark purple background"
    file_name = "wallpaper.jpeg"


    response = client.images.generate(**query_args)
    process_response(response, file_name=file_name)