# LanguageMate: An AI-Powered Multilingual Voice Assistant

## Overview

LanguageMate is an innovative AI-powered voice assistant designed to seamlessly handle multilingual voice interactions. The system integrates automatic speech recognition (ASR), natural language processing (NLP) via a large language model (LLM), and text-to-speech (TTS) technologies to provide a comprehensive conversational experience. It supports real-time audio processing, dynamic language detection, and context-aware responses, making it ideal for various applications ranging from language learning to customer support.



## Key Features

-   **Multilingual Support**: Capable of understanding and responding in multiple languages, facilitating global communication.
-   **Real-time Audio Processing**: Utilizes ASR to transcribe spoken language into text in real-time.
-   **Intelligent Conversational AI**: Employs an LLM to generate contextually relevant and coherent responses.
-   **High-Quality Speech Synthesis**: Converts text responses into natural-sounding speech using advanced TTS engines.
-   **Session Management**: Maintains conversation history for context-aware interactions.
-   **Flexible Deployment**: Designed for both CPU and GPU environments, with optimized performance on each.
-   **Easy Integration**: Provides a straightforward API for integration into various applications and platforms.

## Architecture

LanguageMate is built upon a modular architecture that allows for easy customization and extension. The core components include:

*   **ASR Engine**: Converts spoken audio into text using state-of-the-art speech recognition models.
*   **LLM**: Processes the transcribed text to understand the user's intent and generate appropriate responses.
*   **TTS Engine**: Synthesizes natural-sounding speech from the LLM-generated text.
*   **API**: Exposes endpoints for interacting with the LanguageMate system, including processing voice input and managing conversations.



## Directory Structure
```├── overlordiam-languagemate/
├── improvements.txt # Notes on ongoing and planned improvements
├── backend/ # Backend code including API and services
│ ├── init.py # Initializes the backend package
│ ├── instructions.txt # Instructions for setting up dependencies and running the backend
│ ├── requirements.txt # List of Python dependencies
│ ├── setup.py # Installation script for the backend
│ ├── .gitignore # Specifies intentionally untracked files that Git should ignore
│ ├── src/ # Source code for the backend
│ │ ├── init.py # Initializes the source package
│ │ ├── main.py # Main application file for the FastAPI backend
│ │ ├── api/ # API related modules
│ │ │ ├── init.py # Initializes the API package
│ │ │ ├── orchestrator.py # Orchestrates the pipeline of ASR, LLM, and TTS services
│ │ │ └── routers/ # API routers for different functionalities
│ │ │ └── chatbotInteraction.py # Defines API endpoints for chatbot interactions
│ │ └── services/ # Contains different services like ASR, LLM, and TTS
│ │ ├── init.py # Initializes the services package
│ │ ├── asr/ # Automatic Speech Recognition service
│ │ │ ├── audio_processing.py # Handles audio recording and transcription
│ │ │ └── keyboard_handler.py # Manages keyboard input for controlling recording
│ │ ├── llm/ # Large Language Model service
│ │ │ ├── llm_inference_engine.py # Manages LLM inference
│ │ │ └── test_LLM_instance.py # Tests the LLM instance
│ │ └── tts/ # Text-to-Speech service
│ │ ├── tts_inference_engine.py # Manages TTS inference
│ │ └── test_TTS_instance.py # Tests the TTS instance
│ └── voice_samples/ # Audio samples for TTS
├── frontend/ # Frontend code
│ ├── README.md # README for the frontend
│ ├── package-lock.json # Records the exact versions of dependencies
│ ├── package.json # Lists dependencies and scripts for the frontend
│ ├── .gitignore # Specifies intentionally untracked files that Git should ignore
│ ├── public/ # Public assets
│ │ ├── index.html # Main HTML file
│ │ ├── manifest.json # Metadata for the web app
│ │ └── robots.txt # Instructions for web crawlers
│ └── src/ # Source code for the frontend
│ ├── App.css # CSS for the main App component
│ ├── App.js # Main App component
│ ├── index.css # Global CSS styles
│ └── index.js # Entry point for the React app
```


## Backend Setup

### Prerequisites

-   Python 3.10+
-   `pip` package installer

### Installation Instructions

1.  **Clone the repository:**

    ```
    git clone [repository-url]
    cd overlordiam-languagemate/backend
    ```

2.  **Install dependencies:**

    *   For CPU:

        ```
        pip install -r requirements.txt
        ```

    *   For GPU (CUDA):

        ```
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r requirements.txt
        ```

3.  **Install additional dependencies for Japanese speech synthesis (if needed):**

    ```
    pip install cutlet unidic
    python -m unidic download
    ```

4.  **Address potential build errors:**

    If you encounter the error message "Microsoft Visual C++ 14.0 or greater is required," follow these steps:

    *   Install Microsoft C++ Build Tools from the official website: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    *   During installation, select the "Desktop development with C++" workload to install the necessary compilers and libraries.

5.  **Run on a Virtual Machine (VM):**

    If running the backend in a VM, execute the following commands first:

    ```
    sudo apt update
    sudo apt install portaudio19-dev
    curl -fsSL https://ollama.com/install.sh | sh
    ```

6.  **Install the backend package:**

    ```
    pip install -e .
    ```

### Configuration

-   **Environment Variables**:  

    *Ollama Path*: Ensure that the Ollama server is accessible. The LLM Inference Engine automatically starts the Ollama server if it is not already running.
    *Whisper Model*: Configure the ASR engine by selecting an appropriate Whisper model size. This can be adjusted based on resource availability and accuracy requirements.

### Running the Backend

1.  **Start the FastAPI application:**

    ```
    cd src
    python main.py
    ```

    or

    ```
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

The backend API will be accessible at `http://localhost:8000/api`.

## Frontend Setup

### Prerequisites

-   Node.js 18+
-   npm 6+

### Installation Instructions

1.  **Navigate to the frontend directory:**

    ```
    cd frontend
    ```

2.  **Install dependencies:**

    ```
    npm install
    ```

### Configuration

-   **API Endpoint**:  
    Configure the frontend to communicate with the backend API by setting the correct endpoint in your React components.

### Running the Frontend

1.  **Start the development server:**

    ```
    npm start
    ```

The frontend application will be accessible at `http://localhost:3000`.

## API Endpoints

### `POST /api/process`

Processes voice input and returns an audio response along with the conversation history.

-   **Request**:
    -   Method: `POST`
    -   Headers:
        -   `Content-Type`: `multipart/form-data`
        -   `X-Session-ID` (optional): Session identifier. If not provided, a new session ID will be generated.
    -   Body:
        -   `file`: Audio file (`.wav`, `.mp3`, etc.)

-   **Response**:
    -   Headers:
        -   `X-Session-ID`: The session identifier.
        -   `X-Conversation-History`: The conversation history.
    -   Body:
        -   Audio file (`audio/wav`)

### `POST /api/new-conversation`

Starts a new conversation session.

-   **Request**:
    -   Method: `POST`
    -   Headers:
        -   `X-Session-ID` (optional): Session identifier. If not provided, a new session ID will be generated.

-   **Response**:
    ```
    {
        "session_id": "unique_session_id"
    }
    ```

## Improvements

Refer to the `improvements.txt` file for a list of ongoing and planned enhancements to the LanguageMate system. These include system prompt improvements, optimization strategies, and new feature implementations.

## Contributing

Contributions to LanguageMate are welcome! Please fork the repository, create a feature branch, and submit a pull request with detailed explanations of the changes.


