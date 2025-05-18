import gradio as gr
from fastai.vision.all import *
# Load the exported model
learn = load_learner('shoe_classifier.pkl')

def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    # Return class probabilities as a dictionary
    return {str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Shoe Image Classifier",
    description="Upload a shoe image. The model will predict whether it is AI-generated (fake), a real shoe (adidas/nike/converse), or real other."
)

if __name__ == "__main__":
    demo.launch()
