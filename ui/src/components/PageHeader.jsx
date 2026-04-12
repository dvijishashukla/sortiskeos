import React, { useState, useEffect } from 'react';
import LiveDot from './LiveDot';

export default function PageHeader({ 
  title, 
  subtitle, 
  onBack, 
  actions, 
  usingFallback, 
  pipelineStatus 
}) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header className="no-print" style={{
      background: "rgba(13, 13, 20, 0.75)",
      WebkitBackdropFilter: "blur(12px)",
      backdropFilter: "blur(12px)",
      borderBottom: "1px solid #1e1e2e", 
      padding: "0 32px",
      position: "sticky", 
      top: 0, 
      zIndex: 100,
      boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
      height: 80,
      display: "flex",
      alignItems: "center",
      boxSizing: "border-box",
    }}>
      <div style={{ width: "100%", maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <h1 style={{ fontSize: 20, fontFamily: "'Inter', 'Roboto', sans-serif", fontWeight: 700, letterSpacing: 0.5, color: "#f8fafc", margin: 0 }}>
                {title}
              </h1>
              {subtitle && (
                <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: 1, textTransform: "uppercase", fontWeight: 500, marginTop: 2 }}>
                  {subtitle}
                </div>
              )}
            </div>
          </div>
        </div>
        
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <span style={{ fontFamily: "monospace", fontSize: 12, color: "#64748b" }}>
              {time.toLocaleTimeString()} · backend: {usingFallback ? "offline" : "online"}
            </span>
            <LiveDot pipelineStatus={pipelineStatus} usingFallback={usingFallback} />
          </div>
          
          {actions && (
            <div style={{ display: "flex", alignItems: "center", gap: 12, borderLeft: "1px solid #1e1e2e", paddingLeft: 24 }}>
              {actions}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
