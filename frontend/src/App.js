import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [conversation, setConversation] = useState([]);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

  useEffect(() => {
    // Start a new conversation when component mounts
    startNewConversation();
  }, []);

  const startNewConversation = async () => {
    try {
      const response = await fetch(
        "http://localhost:5000/new-conversation",
        {
          method: "POST",
        }
      );
      const data = await response.json();
      setSessionId(data.session_id);
      setConversation([]);
    } catch (error) {
      console.error("Error starting new conversation:", error);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const audioFile = new File([audioBlob], "audio.wav", { type: "audio/wav" });
        await sendAudioToBackend(audioFile);
      };

      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert(
        "Error accessing microphone. Please ensure you have given permission."
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append("file", audioBlob);

      console.log(sessionId ? sessionId : "No sessionId")
      const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
        headers: sessionId ? { "X-Session-ID": sessionId } : {},
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      // Get conversation history from headers
      const newSessionId = response.headers.get("X-Session-ID");
      const conversationHistory = JSON.parse(
        response.headers.get("X-Conversation-History") || "[]"
      );

      if (newSessionId) {
        setSessionId(newSessionId);
      }
      setConversation(conversationHistory);

      // Handle audio response
      const audioResponse = await response.blob();
      const audioUrl = URL.createObjectURL(audioResponse);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error("Error sending audio to backend:", error);
      alert("Error processing audio. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <h1>Voice Chatbot</h1>
        <div className="conversation">
          {conversation.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-content">{message.content}</div>
            </div>
          ))}
        </div>
        <p className="status">
          {isProcessing
            ? "Processing..."
            : isRecording
            ? "Recording..."
            : "Ready to record"}
        </p>
        <div className="button-container">
          <button
            className={`record-button ${isRecording ? "recording" : ""}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
          >
            {isRecording ? "Stop Recording" : "Start Recording"}
          </button>
          <button
            className="new-chat-button"
            onClick={startNewConversation}
            disabled={isProcessing || isRecording}
          >
            New Chat
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
