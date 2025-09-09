#!/usr/bin/env python3
"""
Test script to verify LM Studio integration with a dummy image
"""
import base64
import requests
import json
from io import BytesIO
from PIL import Image

def create_dummy_image():
    """Create a simple dummy image for testing"""
    # Create a 100x100 red square
    img = Image.new('RGB', (100, 100), color='red')

    # Save to BytesIO buffer
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Convert to base64
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_data}"

def test_vlm_with_dummy_image():
    """Test LM Studio with a dummy image"""
    print("🔍 Testing LM Studio integration with dummy image...")

    # Create dummy image
    dummy_image = create_dummy_image()
    print("✅ Created dummy image (red square)")

    # Test data for the VLM
    test_data = {
        "model": "google/gemma-3-12b-it",  # Default model from settings
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": dummy_image
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    # Make request to LM Studio
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Authorization": "Bearer lm-studio",
        "Content-Type": "application/json"
    }

    try:
        print("📡 Sending request to LM Studio...")
        response = requests.post(url, headers=headers, json=test_data, timeout=30)
        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ LM Studio responded successfully!")
            print("🤖 Response:", result.get("choices", [{}])[0].get("message", {}).get("content", "No content"))

            # Also test the API endpoint
            print("\n🔍 Testing API endpoint...")
            api_response = requests.post(
                "http://localhost:8080/ask",
                json={
                    "question": "What is this?",
                    "k": 1,
                    "m": 1
                },
                timeout=30
            )
            print(f"📊 API Response status: {api_response.status_code}")

            if api_response.status_code == 200:
                print("✅ API endpoint working!")
                # Print first 200 chars of streaming response
                content = api_response.text[:200]
                print(f"📝 API Response preview: {content}...")
            else:
                print("❌ API endpoint failed")
                print("Error:", api_response.text)

        else:
            print("❌ LM Studio request failed")
            print("Error:", response.text)

    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", str(e))

if __name__ == "__main__":
    test_vlm_with_dummy_image()
