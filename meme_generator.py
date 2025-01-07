import os
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import streamlit as st

# Set up Streamlit interface
st.title("Adaptive Meme Generator")
st.write("Upload an image and let the AI generate a meme for it!")

# Step 1: Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Step 2: Generate Image Description
    st.write("Analyzing the image...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=img, return_tensors="pt")
    outputs = model.generate(**inputs)
    image_description = processor.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Image Description: {image_description}")
    
    # Step 3: Generate Meme Caption
    st.write("Generating meme text...")

    # Initialize the meme generator pipeline (removed use_auth_token here)
    TOKEN = "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"  # Replace with your Hugging Face token
    meme_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")  # Removed use_auth_token

    def generate_meme_text(description):
        prompt = f"Write a funny caption for an image showing {description}:"
        # Ensure no unnecessary kwargs are passed to this call
        result = meme_generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
        return result[0]["generated_text"]

    
    meme_caption = generate_meme_text(image_description)
    st.write(f"Meme Caption: {meme_caption}")
    
    # Step 4: Overlay Meme Text on Image
    def add_text_to_image(image, text, output_path="meme_output.jpg"):
        # Ensure the image is in RGB mode (JPEG does not support RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)
        font_path = "arial.ttf"  # Change if Arial font is unavailable; provide path to any .ttf font
        try:
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text size using font.getbbox
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate position (bottom-center)
        width, height = image.size
        position = ((width - text_width) // 2, height - text_height - 20)

        # Draw text on the image
        draw.text(position, text, fill="white", font=font, stroke_width=2, stroke_fill="black")
        image.save(output_path, "JPEG")
        return output_path



    
    st.write("Creating your meme...")
    output_path = "meme_output.jpg"
    add_text_to_image(img.copy(), meme_caption, output_path)
    
    # Display the final meme
    st.image(output_path, caption="Generated Meme", use_column_width=True)
    st.write("Download your meme below!")
    with open(output_path, "rb") as file:
        st.download_button(label="Download Meme", data=file, file_name="meme_output.jpg", mime="image/jpeg")
