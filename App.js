import React, { useState } from "react";
import "./App.css";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { ClipLoader } from "react-spinners";
import { motion } from "framer-motion";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Dashboard from "./Dashboard"; // Import the Dashboard component

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [wbcCounts, setWbcCounts] = useState({});
  const [evaluationMetrics, setEvaluationMetrics] = useState({});
  const [confusionMatrixImage, setConfusionMatrixImage] = useState("");

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);

    if (files.length > 0) {
      toast.info(`${files.length} file(s) selected.`, { theme: "dark" });
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast.error("Please select at least one file!", { theme: "dark" });
      return;
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    setLoading(true);
    toast.info("Uploading and predicting...", { theme: "dark" });

    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPredictions(data.predictions || []);
      const counts = {};
      (data.predictions || []).forEach((item) => {
        counts[item.prediction] = (counts[item.prediction] || 0) + 1;
      });
      setWbcCounts(counts);
      setEvaluationMetrics(data.evaluation_metrics || {});
      setConfusionMatrixImage(data.confusion_matrix || "");
      toast.success("Prediction completed!", { theme: "dark" });
    } catch (error) {
      setPredictions([]);
      setWbcCounts({});
      setEvaluationMetrics({});
      setConfusionMatrixImage("");
      toast.error("Upload failed. Try again.", { theme: "dark" });
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    const csvRows = [
      ["Filename", "Prediction", "Confidence"],
      ...predictions.map((pred) => [
        pred.filename || "",
        pred.prediction,
        pred.confidence,
      ]),
      ["", "", ""],
      ["WBC Type", "Count"],
      ...Object.entries(wbcCounts).map(([type, count]) => [type, count]),
    ];

    const csvContent = csvRows.map((row) => row.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "wbc_predictions.csv";
    link.click();
  };

  return (
    <Router>
      <div className="App">
        <ToastContainer position="top-right" autoClose={3000} />
        <header>
          <h2>MedCell PRO:WBC Classi-Countysis</h2>
        </header>

        <Routes>
          {/* Home page route */}
          <Route
            path="/"
            element={
              <div>
                <div className="upload-section">
                  <label htmlFor="fileInput"><b>Upload Blood Sample</b></label>
                  <input
                    type="file"
                    id="fileInput"
                    onChange={handleFileChange}
                    multiple
                  />
                  <br />
                  <button
                    className="btn"
                    onClick={handleUpload}
                    disabled={loading}
                  >
                    {loading ? (
                      <ClipLoader size={20} color="#00e5ff" loading={true} />
                    ) : (
                      "Upload and Predict"
                    )}
                  </button>
                </div>

                {predictions.length > 0 && (
                  <>
                    <div className="predictions-container">
                      {predictions.map((pred, idx) => (
                        <motion.div
                          key={idx}
                          className="prediction-item"
                          whileHover={{ scale: 1.05 }}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.05 }}
                        >
                          {pred.cropped_image && (
                            <motion.img
                              src={`data:image/jpeg;base64,${pred.cropped_image}`}
                              alt={`Predicted WBC ${idx + 1}`}
                              className="preview-image"
                              whileHover={{ scale: 1.1 }}
                            />
                          )}
                          <p><strong>{pred.filename || `Image ${idx + 1}`}</strong></p>
                          <p><strong>Prediction:</strong> {pred.prediction}</p>
                          <p><strong>Confidence:</strong> {pred.confidence}</p>
                        </motion.div>
                      ))}
                    </div>

                    <div className="download-section">
                      <button className="btn" onClick={handleDownloadCSV}>
                        Download CSV Report
                      </button>
                    </div>
                  </>
                )}

                {Object.keys(wbcCounts).length > 0 && (
                  <>
                    <div>
                      <h3 style={{ textAlign: "center", marginTop: "40px" }}>
                        WBC COUNT
                      </h3>
                      <table>
                        <thead>
                          <tr>
                            <th>WBC Type</th>
                            <th>Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(wbcCounts).map(([wbcType, count], idx) => (
                            <tr key={idx}>
                              <td>{wbcType}</td>
                              <td>{count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Link to go to the Dashboard */}
                    <div className="navigate-to-dashboard">
                      <Link to="/dashboard" className="btn">
                        Go to Dashboard
                      </Link>
                    </div>
                  </>
                )}
              </div>
            }
          />

          {/* Dashboard page route */}
          <Route
            path="/dashboard"
            element={
              <Dashboard
                wbcCounts={wbcCounts}
                evaluationMetrics={evaluationMetrics}
                confusionMatrixImage={confusionMatrixImage}
              />
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
