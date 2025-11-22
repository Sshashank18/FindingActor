# Actor Recommendation System

- https://www.kaggle.com/code/deucalionsash/findingactor (Kaggle Link)

A comprehensive machine learning system that analyzes Bollywood actors' facial features and role traits to recommend the best actors for specific character requirements. The system combines computer vision, natural language processing, and deep learning to provide intelligent casting recommendations.

## Overview

This project processes a dataset of Bollywood actor images and uses advanced AI techniques to extract facial features, analyze role patterns, and create a recommendation engine for film casting decisions. The system employs both structured data analysis and semantic text matching to provide accurate recommendations.

## Features

- **Facial Feature Extraction**: Uses DeepFace and ResNet-50 to analyze facial characteristics including age, gender, race, and emotions
- **Role Pattern Analysis**: Leverages OpenAI's GPT-4 to identify common character traits actors portray across their filmography
- **Hybrid Recommendation Engine**: Combines structured feature matching with semantic similarity for optimal results
- **Comprehensive Actor Profiles**: Creates detailed profiles including aggregated facial features and role traits
- **Real-time Processing**: Batch processing with progress saving and checkpoint management

## Architecture

### Core Components

**FacialFeatureExtractor**: Custom PyTorch neural network based on ResNet-50 for extracting facial features

**DeepFace Integration**: Analyzes facial attributes including:
- Age estimation
- Gender classification
- Race detection
- Emotion recognition
- Facial expression analysis

**Role Trait Analyzer**: Uses OpenAI API to extract character traits from actors' filmographies

**Hybrid Recommendation System**: Combines two approaches:
- Structured matching (60% weight): Direct comparison of facial features and traits
- Semantic similarity (40% weight): Text embedding comparison using sentence transformers

## Installation

pip install deepface openai torch torchvision transformers sentence-transformers


## Dataset Structure

Bollywood Actor Images/
├── Actor1/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Actor2/
│ └── ...
└── List of Actors.txt

## Database Link
https://www.kaggle.com/datasets/iamsouravbanerjee/indian-actor-images-dataset


## Usage

### Basic Processing

Process all actors in the dataset
base_dir = "path/to/Bollywood Actor Images"
processed_actors = process_actors_batch(base_dir)

### Making Recommendations

Define role requirements
role_requirements = {
'traits': ['intense', 'authoritative', 'Strong and bold voice'],
'facial_features': {
'age': 40,
'gender': 'Male',
'dominant_emotion': 'angry'
}
}

role_description = "Army look, strong personality, aggressive eyes"

Get recommendations
recommendations = hybrid_recommend_actors(
role_requirements,
role_description,
top_n=5
)

## Data Processing Pipeline

### Stage 1: Feature Extraction
- Loads actor images from organized directories
- Applies DeepFace analysis for facial feature detection
- Extracts age, gender, race, emotion, and custom facial metrics

### Stage 2: Role Analysis
- Queries OpenAI API with actor names to identify common character traits
- Processes and structures trait data into JSON format

### Stage 3: Data Aggregation
- Combines multiple images per actor into aggregated feature profiles
- Handles numeric averaging and categorical mode selection

### Stage 4: Summary Generation
- Uses BART transformer model to create comprehensive actor summaries
- Combines role traits and facial features into coherent descriptions

### Stage 5: Embedding Creation
- Generates semantic embeddings using sentence transformers
- Enables similarity-based matching for text queries

## Recommendation Methods

### Method 1: Structured Matching
Direct comparison of:
- Role trait keywords
- Facial feature proximity
- Demographic characteristics

### Method 2: Semantic Similarity
- Embeds user queries and actor summaries
- Computes cosine similarity scores
- Ranks actors by semantic relevance

### Method 3: Hybrid Approach
Combines both methods with weighted scoring:
- 60% structured matching
- 40% semantic similarity

## File Structure

project/
├── processed_actors.json # Raw processed data
├── cleaned_data.json # Structured actor profiles
├── summarized_full_data.json # Actor summaries
├── actor_processing_checkpoint.json # Progress tracking
└── findingactor.ipynb

## API Configuration

The system requires OpenAI API access for role trait analysis. Configure your API key:

token = 'YOUR_API_KEY'
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o"

## Performance Optimization

- **Batch Processing**: Processes actors sequentially with progress saving
- **Checkpoint System**: Resumes processing from last successful actor
- **Rate Limiting**: Includes delays to respect API limits
- **Error Handling**: Graceful failure recovery for individual actors

## Output Format

Each actor profile contains:
- **Role Traits**: List of character types and personality traits
- **Aggregated Features**: Averaged facial characteristics across images
- **Summary**: Natural language description combining all attributes

## Dependencies

- PyTorch & TorchVision
- DeepFace
- OpenAI API
- Transformers (Hugging Face)
- Sentence Transformers
- PIL/Pillow
- NumPy
- Requests


## How to Run

- python backend_api.py (To run the backend flask api request)
- cd actor-suggestion -> npm start (To run the frontend application on React)


## Screenshots

<img src="https://github.com/Sshashank18/FindingActor/blob/main/Screenshots/Screenshot%20(34).png"
     style="float: left; margin-right: 10px;"/>
<img src="https://github.com/Sshashank18/FindingActor/blob/main/Screenshots/Screenshot%20(35).png"
     style="float: left; margin-right: 10px;"/>
