from app.rag import retrieve_relevant_docs
import torch

def generate_response_phi(question, tokenizer, model, embedding_model, index, docs, max_new_tokens=250):
    docs_found = retrieve_relevant_docs(question, embedding_model, index, docs)
    context = "\n".join(docs_found)
    prompt = (
        f"Voici des informations sur Roger :\n{context}\n\n"
        f"Question : {question}\n"
        f"Réponse détaillée :"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip(), context
 