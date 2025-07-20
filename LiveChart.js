// src/LiveChart.jsx
import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

const LiveChart = () => {
  const [counts, setCounts] = useState({});

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/predictions");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const label = data.label;

      setCounts((prevCounts) => ({
        ...prevCounts,
        [label]: (prevCounts[label] || 0) + 1,
      }));
    };

    return () => {
      ws.close();
    };
  }, []);

  const chartData = Object.entries(counts).map(([label, value]) => ({
    name: label,
    count: value,
  }));

  return (
    <div style={{ marginTop: "50px" }}>
      <h2 style={{ textAlign: "center" , color: "cyan", marginBottom: "20px"}}>Live WBC Predictions</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" stroke="#ccc" />
          <YAxis stroke="#ccc" />
          <Tooltip />
          <Bar dataKey="count" fill="#ff6b6b" radius={[10, 10, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LiveChart;
