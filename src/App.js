import { useState } from "react";
import "./App.css";

export default function SmartNotesApp() {
  const [videoUrl, setVideoUrl] = useState("");
  const [summary, setSummary] = useState("");
  const [pdfUrl, setPdfUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true); // Set loading before sending the request
    setSummary("");
    setPdfUrl(null);
  
    try {
      const response = await fetch("http://127.0.0.1:5001/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_url: videoUrl }),
      });
  
      if (!response.ok) {
        throw new Error("Failed to process video");
      }
  
      const data = await response.json();
      setSummary(data.summary);
      setPdfUrl(data.pdf_url);
    } catch (error) {
      console.error("Error:", error);
      alert("Error processing video. Please try again.");
    } finally {
      setLoading(false); // Always stop loading after response
    }
  };
  

  return (
    <div className="container">
      <h1>Smart Learning Notes</h1>
      <input
        type="text"
        placeholder="Enter YouTube Video URL"
        value={videoUrl}
        onChange={(e) => setVideoUrl(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Processing..." : "Generate Notes"}
      </button>
      {summary && (
        <div className="summary-box">
          <h2>Summary:</h2>
          <p>{summary}</p>
          {pdfUrl && (
            <a href={pdfUrl} download="summary.pdf">Download PDF</a>
          )}
        </div>
      )}
    </div>
  );
}
