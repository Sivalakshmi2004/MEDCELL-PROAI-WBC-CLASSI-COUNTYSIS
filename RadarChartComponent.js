import React from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

function RadarChartComponent({ data }) {
  const formattedData = Object.entries(data).map(([type, count]) => ({
    type,
    count,
  }));

  return (
    <div>
      <h2 style={{ textAlign: "center", color: "cyan", marginBottom: "20px" }}>
        WBC Types Distribution
      </h2>
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart data={formattedData}>
          <PolarGrid stroke="#444" />
          <PolarAngleAxis dataKey="type" stroke="#00e5ff" />
          <PolarRadiusAxis stroke="#00e5ff" />
          <Tooltip contentStyle={{ backgroundColor: "#1e1e1e", borderColor: "#00e5ff", color: "white" }} />
          <Radar
            name="WBC Types"
            dataKey="count"
            stroke="#00e5ff"
            fill="#00e5ff"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default RadarChartComponent;
