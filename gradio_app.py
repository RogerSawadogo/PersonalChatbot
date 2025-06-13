import gradio as gr
import torch


# Load models and index
embedding_model, index, docs = load_index_and_docs()
tokenizer, model = load_phi_model()

# Define chatbot logic
def chat_with_roger(message, history):
    # RAG: find context for the question
    docs_found = retrieve_relevant_docs(message, embedding_model, index, docs)
    context = "\n".join(docs_found)

    # Format the prompt
    prompt = (
        f"Voici des informations sur Roger :\n{context}\n\n"
        f"Question : {message}\n"
        f"Réponse détaillée :"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()

    return answer

# Launch Gradio chat app
if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat_with_roger,
        title="RogerBot 🤖",
        description="Pose ta question à Roger !",
        theme=gr.themes.Soft(primary_hue="indigo"),
        chatbot=gr.Chatbot(height=500, label="RogerBot")
    ).launch()