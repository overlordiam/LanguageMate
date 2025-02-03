import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

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
      // mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const sendAudioToBackend = async (audioFile) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append("file", audioFile);

      const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
  
      // Verify content type
      const contentType = response.headers.get("content-type");
      if (!contentType?.startsWith("audio/")) {
        throw new Error("Invalid audio response from server");
      }
  
      // Create audio URL and play
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      console.log(audio);
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
        <p className="status">
          {isProcessing
            ? "Processing..."
            : isRecording
            ? "Recording..."
            : "Ready to record"}
        </p>
        <button
          className={`record-button ${isRecording ? "recording" : ""}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </button>
      </div>
    </div>
  );
}

export default App;
