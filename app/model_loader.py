from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_phi_model():
    print("Chargement du modèle Phi-3.5 pour CPU...")
    model_name = "microsoft/phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 # Use float32 for CPU compatibility
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model
