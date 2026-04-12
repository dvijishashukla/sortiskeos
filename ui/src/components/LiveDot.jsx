import React from 'react';

export default function LiveDot({ pipelineStatus, usingFallback }) {
  let color = "#22c55e";
  let text = "Live";
  let animate = true;

  if (pipelineStatus === "running") {
    color = "#7c3aed";
    text = "Running";
  } else if (usingFallback) {
    color = "#ef4444";
    text = "Offline";
    animate = false;
  }

  return (
    <span style={{ 
      display: "inline-flex", 
      alignItems: "center", 
      gap: 6, 
      fontSize: 12, 
      color: color,
      fontWeight: 600,
      letterSpacing: 0.5
    }}>
      <span style={{
        width: 10, height: 10, borderRadius: "50%", background: color,
        animation: animate ? "pulse 1.6s ease-out infinite" : "none", display: "inline-block",
      }} />
      {text}
      <style>{`
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 ${color}77; }
          70% { box-shadow: 0 0 0 8px ${color}00; }
          100% { box-shadow: 0 0 0 0 ${color}00; }
        }
      `}</style>
    </span>
  );
}
