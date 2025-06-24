# backend_api.py - Flask backend to integrate with your existing code
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import os
import random

app = Flask(__name__)
CORS(app)

# Load your existing data and model (same as your code)
with open("cleaned_data.json", "r") as f:
    actor_profiles = json.load(f)

with open("summarized_full_data.json", "r") as f:
    actor_data = json.load(f)
actor_summaries = {actor: details['summary'] for actor, details in actor_data.items()}

model = SentenceTransformer("all-mpnet-base-v2")

# Configure the path to your Bollywood Actor Images folder
IMAGES_BASE_PATH = r"C:\Users\SHASHANK\Downloads\FindingActor-main\FindingActor-main\Bollywood Actor Images\Bollywood Actor Images"

# Your existing functions (copy them here)
def compute_trait_score(actor_traits, query_traits):
    match_count = sum(1 for qt in query_traits if any(qt.lower() in at['trait'].lower() for at in actor_traits))
    return match_count / len(query_traits) if query_traits else 0

def compute_feature_score(actor_features, query_features):
    score = 0
    max_score = 0
    
    if 'age' in actor_features and 'age' in query_features:
        age_diff = abs(actor_features['age'] - query_features['age'])
        age_score = max(0, 1 - age_diff / 30)
        score += age_score
        max_score += 1
    
    if 'dominant_emotion' in actor_features and 'dominant_emotion' in query_features:
        score += int(actor_features['dominant_emotion'] == query_features['dominant_emotion'])
        max_score += 1
    
    return score / max_score if max_score else 0

def get_structured_score(actor_name, role):
    traits_score = compute_trait_score(actor_profiles[actor_name]['role_traits'], role['traits'])
    features_score = compute_feature_score(actor_profiles[actor_name]['aggregated_features'], role['facial_features'])
    return (traits_score + features_score) / 2

def get_summary_similarity(actor_name, role_text):
    actor_summary = actor_summaries.get(actor_name, "")
    if not actor_summary:
        return 0
    query_embedding = model.encode(role_text, convert_to_tensor=True)
    summary_embedding = model.encode(actor_summary, convert_to_tensor=True)
    return util.cos_sim(query_embedding, summary_embedding).item()

def hybrid_recommend_actors(role, role_text_description, top_n=5, w_structured=0.6, w_text=0.4):
    scores = []
    
    for actor_name in actor_profiles.keys():
        structured_score = get_structured_score(actor_name, role)
        text_score = get_summary_similarity(actor_name, role_text_description)
        final_score = w_structured * structured_score + w_text * text_score
        scores.append((actor_name, final_score))
    
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]

def get_actor_folder_name(actor_name):
    """Convert actor name to folder name format"""
    # Handle common name variations and special characters
    folder_name = actor_name.replace(' ', '_').replace('-', '_')
    folder_name = ''.join(c for c in folder_name if c.isalnum() or c == '_')
    return folder_name.lower()

def find_actor_image(actor_name):
    """Find any random image from actor's folder"""
    try:
        folder_name = get_actor_folder_name(actor_name)
        
        # Try different possible folder name variations
        possible_folders = [
            folder_name,
            actor_name.replace(' ', '_'),
            actor_name.replace(' ', '').lower(),
            actor_name.lower().replace(' ', '_')
        ]
        
        for folder_variant in possible_folders:
            actor_folder = os.path.join(IMAGES_BASE_PATH, folder_variant)
            
            if os.path.exists(actor_folder) and os.path.isdir(actor_folder):
                # Get all image files from the folder
                image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG')
                image_files = [f for f in os.listdir(actor_folder) 
                              if f.endswith(image_extensions)]
                
                if image_files:
                    # Pick a random image
                    random_image = random.choice(image_files)
                    return os.path.join(actor_folder, random_image)
        
        return None
        
    except Exception as e:
        print(f"Error finding image for {actor_name}: {str(e)}")
        return None

@app.route('/api/recommend-actors', methods=['POST'])
def recommend_actors():
    try:
        data = request.json
        role_text = data.get('role_text', '')
        example_role = data.get('example_role', {})
        
        # Get recommendations
        top_actors = hybrid_recommend_actors(example_role, role_text)
        
        # Format response with image URLs
        actors = []
        for actor, score in top_actors:
            actor_data = {
                'name': actor,
                'score': score,
                'image_url': f"/api/actor-image/{actor}"
            }
            actors.append(actor_data)
        
        return jsonify({
            'success': True,
            'actors': actors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/actor-image/<actor_name>')
def get_actor_image(actor_name):
    """Serve a random image for the specified actor"""
    try:
        image_path = find_actor_image(actor_name)
        
        if image_path and os.path.exists(image_path):
            return send_file(image_path)
        else:
            # Return a 404 if no image found
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        print(f"Error serving image for {actor_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/actor-image-info/<actor_name>')
def get_actor_image_info(actor_name):
    """Get information about available images for an actor"""
    try:
        folder_name = get_actor_folder_name(actor_name)
        possible_folders = [
            folder_name,
            actor_name.replace(' ', '_'),
            actor_name.replace(' ', '').lower(),
            actor_name.lower().replace(' ', '_')
        ]
        
        for folder_variant in possible_folders:
            actor_folder = os.path.join(IMAGES_BASE_PATH, folder_variant)
            
            if os.path.exists(actor_folder) and os.path.isdir(actor_folder):
                image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG')
                image_files = [f for f in os.listdir(actor_folder) 
                              if f.endswith(image_extensions)]
                
                if image_files:
                    return jsonify({
                        'success': True,
                        'actor_name': actor_name,
                        'folder_name': folder_variant,
                        'image_count': len(image_files),
                        'image_files': image_files,
                        'image_url': f"/api/actor-image/{actor_name}"
                    })
        
        return jsonify({
            'success': False,
            'error': 'No images found for this actor',
            'searched_folders': possible_folders
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-images')
def test_images():
    """Test endpoint to check available actor folders"""
    try:
        if not os.path.exists(IMAGES_BASE_PATH):
            return jsonify({
                'error': 'Images base path does not exist',
                'path': IMAGES_BASE_PATH
            }), 404
        
        folders = []
        for item in os.listdir(IMAGES_BASE_PATH):
            item_path = os.path.join(IMAGES_BASE_PATH, item)
            if os.path.isdir(item_path):
                image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG')
                image_files = [f for f in os.listdir(item_path) 
                              if f.endswith(image_extensions)]
                folders.append({
                    'folder_name': item,
                    'image_count': len(image_files),
                    'sample_images': image_files[:3]  # Show first 3 images
                })
        
        return jsonify({
            'success': True,
            'base_path': IMAGES_BASE_PATH,
            'total_folders': len(folders),
            'folders': folders[:10]  # Show first 10 folders
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
