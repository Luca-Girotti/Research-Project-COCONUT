import os
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import streamlit as st
from textwrap import wrap

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
        prompt = f"Write a short and sarcastic meme caption for an image showing {description}. Keep it under 20 words and make it hilarious:"
        result = meme_generator(
            prompt,
            max_length=50,  # Allow slightly more room for processing
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9  # Adjust randomness
        )
        # Clean up the generated text
        generated_text = result[0]["generated_text"]
        meme_text = generated_text.replace(prompt, "").strip()
        return meme_text





    
    meme_caption = generate_meme_text(image_description)
    st.write(f"Meme Caption: {meme_caption}")
    
    # Step 4: Overlay Meme Text on Image
    def add_text_to_image(image, text, output_path="meme_output.jpg"):
        # Ensure the image is in RGB mode (JPEG does not support RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Initialize the drawing context
        draw = ImageDraw.Draw(image)

        # Specify a font path
        font_path = "arial.ttf"  # Update with the path to a valid .ttf font file
        try:
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            st.write("Font not found, using default font.")
            font = ImageFont.load_default()

        # Dynamically wrap text to fit within the image width
        max_width = image.width - 40  # Margin of 20 pixels on each side
        wrapped_text = []
        for line in text.split("\n"):
            wrapped_text.extend(wrap(line, width=30))  # Adjust width for your use case

        # Adjust font size dynamically
        while True:
            line_heights = [font.getbbox(line)[3] for line in wrapped_text]
            total_height = sum(line_heights)
            if total_height < image.height * 0.7:  # Text should occupy no more than 70% of the image height
                break
            font = ImageFont.truetype(font_path, font.size - 2)

        # Calculate text position (centered)
        y = (image.height - total_height) // 2
        for line in wrapped_text:
            text_width = font.getbbox(line)[2]
            x = (image.width - text_width) // 2
            draw.text((x, y), line, fill="white", font=font, stroke_width=2, stroke_fill="black")
            y += font.getbbox(line)[3]

        # Save the image
        image.save(output_path)
        return output_path





    
    st.write("Creating your meme...")
    output_path = "meme_output.jpg"
    add_text_to_image(img.copy(), meme_caption, output_path)
    
    # Display the final meme
    st.image(output_path, caption="Generated Meme", use_column_width=True)
    st.write("Download your meme below!")
    with open(output_path, "rb") as file:
        st.download_button(label="Download Meme", data=file, file_name="meme_output.jpg", mime="image/jpeg")
