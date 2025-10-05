import os
from openai import OpenAI

if __name__ == "__main__":
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    def list_models():
        models = client.models.list()
        for model in models.data:
            print(model.id)

    list_models()