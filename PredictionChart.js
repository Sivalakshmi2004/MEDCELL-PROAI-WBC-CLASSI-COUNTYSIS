import React, { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const COLORS = ["#00e5ff", "#00bfa5", "#ff5252", "#ffab00", "#7c4dff"];

const PredictionChart = ({ wbcCounts }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    const updatedData = Object.entries(wbcCounts).map(([type, count]) => ({
      type,
      count,
    }));
    setChartData(updatedData);
  }, [wbcCounts]);

  const handleReset = () => {
    setChartData([]);
  };

  return (
    <div style={{ marginBottom: "40px" }}>
      <h3 style={{ textAlign: "center" }}>Live Prediction Timeline</h3>
      <div style={{ textAlign: "center", marginBottom: "10px" }}>
        <button
          onClick={handleReset}
          style={{
            backgroundColor: "#ff5252",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            padding: "8px 16px",
            cursor: "pointer",
          }}
        >
          Reset Chart
        </button>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            {COLORS.map((color, idx) => (
              <linearGradient key={idx} id={`color${idx}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.8} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="type" stroke="#00e5ff" />
          <YAxis stroke="#00e5ff" />
          <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", borderColor: "#00e5ff" }} />
          <Area
            type="monotone"
            dataKey="count"
            stroke="#00e5ff"
            fill="url(#color0)"
            animationDuration={600}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionChart;
