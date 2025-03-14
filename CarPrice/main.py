import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import json
import pickle
import numpy as np

load_dotenv()
client = OpenAI()

with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('mappings.json', 'rb') as file:
    mappings_dict = json.load(file)

mappings_dict_adjusted = {}

for key, value in mappings_dict.items():
    d = {}
    for ind, val in enumerate(value):
        d[val.lower().strip()] = ind

    mappings_dict_adjusted[key.lower().strip()] = d


def get_car_price(openai_dict):
    print(openai_dict)

    x_pred = []

    for key, value in openai_dict.items():
        key_mapping = mappings_dict_adjusted.get(key)
        if key_mapping:
            x_pred.append(key_mapping.get(value))
        else:
            x_pred.append(value)

    x_pred_np = np.array(x_pred)
    x_pred_np = np.expand_dims(x_pred_np, axis=0)

    y_pred = loaded_model.predict(x_pred_np)

    return y_pred.squeeze()

tools = [{
    "type": "function",
    "function": {
        "name": "get_car_price",
        "description": "Get car price given brand, model, year, engine size, fuel_type, transmission, mileage, doors, and owner. You should call this function only if all parameters are available.",
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": 'Brand of the car like Audi, BMW, etc.'
                },
                "model": {
                    "type": "string",
                    "description": 'Model of the car like A3, A4, Civic, etc.'
                },
                "year": {
                    "type": "number",
                    "description": "Car's year of production"
                },
                "engine_size": {
                    "type": "number",
                    "description": "Number of liters for the engine like 4.2, 3.5, etc."
                },
                "fuel_type": {
                    "type": "string",
                    "description": "Type of fuel. Whether 'Diesel', 'Electric', 'Hybrid', 'Petrol'"
                },
                "transmission": {
                    "type": "string",
                    "description": "Type of transmission. Whether 'Automatic', 'Manual', 'Semi Automatic'"
                },
                "mileage": {
                    "type": "number",
                    "description": "Number of miles the car took"
                },
                "doors": {
                    "type": "number",
                    "description": "Number of doors of the car"
                },
                "owners_count": {
                    "type": "number",
                    "description": "Number of owners that previously owned the car"
                }
            },
            "required": [
                "brand",
                "model",
                "year",
                "engine_size",
                "fuel_type",
                "transmission",
                "mileage",
                "doors",
                "owners_count"
            ],
            "additionalProperties": False
        }
    }
}]

def chat_callback(message, history):
    messages = []
    messages.append({"role": "system", "content": "You are working for customer service company that sells used cars. You should ask the client about the car's brand, model, year of production, engine size, fuel type, transmission, mileage, number of doors, number of owners that previously owned the car to get the price of the car"})
    
    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
        # stream=True
    )

    first_choice = response.choices[0]
    if first_choice.finish_reason == "tool_calls":
        tool_msg = first_choice.message
        messages.append(tool_msg)
        tool_call = tool_msg.tool_calls[0]
        func = tool_call.function
        price = eval(func.name)(json.loads(func.arguments))
        messages.append({"role": "tool", "content": f"price={price}", "tool_call_id": tool_call.id})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

    return response.choices[0].message.content

    # response_str = ""
    # for chunk in response:
        # print(chunk)
        # delta = chunk.choices[0].delta.content
        # if delta:            
            # response_str += delta
        # else:
            # response_str += ""
        
        # yield response_str

interface = gr.ChatInterface(chat_callback, type="messages")
interface.launch()
