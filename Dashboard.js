import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  PieChart, Pie, Cell, ResponsiveContainer, CartesianGrid
} from "recharts";
import PredictionChart from "./PredictionChart";
import RadarChartComponent from "./RadarChartComponent";
import { Link } from "react-router-dom";
import ClipLoader from "react-spinners/ClipLoader"; // ✅ Add this
import "./Dashboard.css";

const COLORS = ["#00e5ff", "#00bfa5", "#ff5252", "#ffab00", "#7c4dff"];

const Dashboard = ({ wbcCounts, confusionMatrixImage }) => {
  const [loading] = useState(false); // ✅ Initialize loading state if needed
  const wbcData = Object.entries(wbcCounts).map(([type, count]) => ({
    type,
    count,
  }));

  return (
    <div className="dashboard-container">
      <h2 style={{ textAlign: "center", marginTop: "40px" }}>
        Live WBC Statistics
      </h2>

      {/* Bar Chart */}
      <div className="chart-section">
        <h2 style={{ textAlign: "center", color: "cyan", marginBottom: "20px" }}>
          Bar-Chart Distribution
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={wbcData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="type" stroke="#00e5ff" />
            <YAxis stroke="#00e5ff" />
            <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", borderColor: "#00e5ff" }} />
            <Bar dataKey="count">
              {wbcData.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pie Chart */}
      <div className="chart-section">
        <h2 style={{ textAlign: "center", color: "cyan", marginBottom: "20px" }}>
          WBC Pie Chart – Cell Type Distribution
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={wbcData}
              dataKey="count"
              nameKey="type"
              cx="50%"
              cy="50%"
              outerRadius={100}
              innerRadius={50}
              label={({ type }) => type}
              labelLine={false}
            >
              {wbcData.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", borderColor: "#00e5ff" }} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Radar Chart */}
      <div className="chart-section">
        <RadarChartComponent data={wbcCounts} />
      </div>

      {/* Area Chart */}
      <div className="chart-section">
        <h3 style={{ textAlign: "center", color: "cyan", marginBottom: "20px" }}>WBC Area Chart</h3>
        <PredictionChart wbcCounts={wbcCounts} />
      </div>

      {/* Confusion Matrix */}
      {confusionMatrixImage && (
        <div className="confusion-matrix-section">
          <h3 style={{ textAlign: "center" }}>Confusion Matrix</h3>
          <img
            src={`data:image/png;base64,${confusionMatrixImage}`}
            alt="Confusion Matrix"
            className="confusion-matrix-image"
          />
        </div>
      )}

      {/* Link back to Home */}
      <div className="navigate-back-home">
        {loading ? (
          <ClipLoader size={20} color="#00e5ff" loading={true} />
        ) : (
          <Link to="/" className="btn">Back to Home</Link>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
