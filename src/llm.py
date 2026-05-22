# rits_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEFAULT_RITS_BASE_URL = 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com'


class Rits(OpenAI):

    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model = kwargs.pop('model')
        else:
            self.model = os.environ.get("RITS_MODEL")
        
        if self.model is None or self.model == '':
            raise Exception("model argument must be passed or RITS_MODEL environment var must be defined as full RITS model id (e.g., ibm-granite/granite-3.0-8b-instruct).")
            
        # BASE_URL logic
        base_url = kwargs.get('base_url')
        if base_url is None:
            base_url = os.environ.get("RITS_BASE_URL",
                                      os.environ.get('OPENAI_BASE_URL', DEFAULT_RITS_BASE_URL))
        
        # update the RITS base url per model
        model_url_id = self.model.strip().split('/')[-1].replace('.', '-')
        if self.model == 'ibm-granite/granite-20b-code-instruct-unified-api': 
            model_url_id = 'granite-20b-code-instruct-uapi'
        if self.model == 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8':
            model_url_id = 'llama-4-mvk-17b-128e-fp8'
        if self.model == 'mistralai/Mistral-Small-3.1-24B-Instruct-2503':
            model_url_id = 'mistral-small-3-1-24b-2503'
        if self.model == 'mistralai/Mistral-Small-3.2-24B-Instruct-2506':
            model_url_id = 'mistral-small-3-2-24b-2506'
        base_url = os.path.join(base_url, model_url_id, 'v1')
        rits_key = os.environ.get('RITS_API_KEY')
        api_key = kwargs.get('api_key')
        default_headers = kwargs.get('default_headers', {})
        
        if default_headers.get('RITS_API_KEY'):
            api_key = default_headers.get('RITS_API_KEY')
        elif api_key is None:
            if rits_key:
                api_key = rits_key
            else:
                api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None or api_key == '':
            raise Exception("RITS_API_KEY environment var not defined.")
        
        default_headers['RITS_API_KEY'] = api_key
        
        kwargs['base_url'] = base_url
        kwargs['api_key'] = api_key
        kwargs['default_headers'] = default_headers
        
        super().__init__(**kwargs)

    def generate_text(self, prompt: str, max_tokens: int = 50):
        """
        Generate text using the RITS client.
        
        Args:
            prompt (str): The input text prompt.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 50.
            
        Returns:
            str: The generated text.
        """
        try:
            response = self.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature = 0.0,
                # decoding_method = 'greedy',
                # min_tokens=0,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    client = Rits(model='openai/gpt-oss-120b')
    input = """SYSTEM: You are a helpful assistant and your task is to find all the necessary slot arguments and their values from the input text. Please do not add anything after slot 4:
    Slot-1: what is the widget being created? Allowed values ("timeseries","Big number", "Pie Chart")
    Slot-2: what are the metric names ? Allowed values (Calls, Erroneous Calls (Count), Erroneous Calls (rate), Latency, number of)
    Slot-3:"Aggregation type for the metric. Allowed Values ("min", "max","Sum","Mean","Mod", "Median", "25th Percentile","50th percentile", 75th percentile, 90th percentile, 95th percentile, 98th percentile, 99th percentile, "per second", "per minute", "per hour")
    Slot-4:"what is the service name?"
    USER: {}".
    ASSISTANT: 
    """
    prompt = "Generate a big number widget that shows the sum total latency for catalogue service."
    llm_input = input.format(prompt)
    generated_text = client.generate_text(llm_input, max_tokens=1024)
