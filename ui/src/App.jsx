import { useState } from "react";
import Dashboard from "./Dashboard";
import RootCauseDetail from "./pages/RootCauseDetail";
import AnomaliesDetail from "./pages/AnomaliesDetail";
import CrashHistory from "./pages/CrashHistory";
import RawLogs from "./pages/RawLogs";
import Settings from "./pages/Settings";
import AuditLog from "./pages/AuditLog";
import { checkHealth } from "./api.js";
import { useEffect } from "react";
import { toDisplayText } from "./utils/displayValue.js";

// ── Simple hash-based router (no react-router needed) ─────────────────────────
function useRoute() {
  const [route, setRoute] = useState({ page: "dashboard", params: {} });
  const navigate = (page, params = {}) => setRoute({ page, params });
  return { route, navigate };
}

// ── Sidebar nav items ─────────────────────────────────────────────────────────
const NAV = [
  { id: "dashboard",  icon: "⬡", label: "Dashboard",     sub: "Overview" },
  { id: "anomalies",  icon: "◈", label: "Anomalies",      sub: "ML detections" },
  { id: "rootcause",  icon: "⊕", label: "Root Cause",     sub: "Cluster analysis" },
  { id: "crashes",    icon: "⚡", label: "Crash History",  sub: "All events" },
  { id: "logs",       icon: "≡", label: "Raw Logs",       sub: "Log stream" },
  { id: "audit",      icon: "⊛", label: "Audit Log",     sub: "Security ledger" },
  { id: "settings",   icon: "◎", label: "Settings",       sub: "Config" },
];

function Sidebar({ active, onNavigate, collapsed, setCollapsed, health }) {
  return (
    <aside style={{
      width: collapsed ? 64 : 220,
      minHeight: "100vh",
      background: "rgba(19, 19, 31, 0.75)",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      borderRight: "1px solid #1e1e2e",
      display: "flex",
      flexDirection: "column",
      transition: "width 0.25s cubic-bezier(.4,0,.2,1)",
      position: "fixed",
      top: 0, left: 0, bottom: 0,
      zIndex: 200,
      overflow: "hidden",
    }}>
      {/* Logo */}
      <div style={{
        height: 80,
        padding: "0 20px",
        borderBottom: "1px solid #1e1e2e",
        display: "flex", alignItems: "center",
        justifyContent: collapsed ? "center" : "space-between",
        boxSizing: "border-box",
        background: "rgba(255,255,255,0.02)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <img 
            src="/logo.png" 
            alt="SortisKeos Logo" 
            style={{ 
              width: collapsed ? 36 : 40, 
              height: collapsed ? 36 : 40, 
              borderRadius: "50%",
              objectFit: "cover",
              boxShadow: "0 0 20px rgba(124, 58, 237, 0.2)",
              transition: "all 0.3s ease",
            }} 
          />
          {!collapsed && (
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <div style={{ 
                fontSize: 18, 
                fontFamily: "'Roboto', sans-serif", 
                fontWeight: 800, 
                color: "#f8fafc", 
                letterSpacing: -0.5,
                lineHeight: 1
              }}>SortisKeos</div>
              <div style={{ 
                fontSize: 10, 
                color: "#7c3aed", 
                letterSpacing: 1.5, 
                textTransform: "uppercase", 
                marginTop: 4, 
                fontWeight: 600,
                opacity: 0.8
              }}>Log Analysis System</div>
            </div>
          )}
        </div>
        {!collapsed && (
          <button onClick={() => setCollapsed(true)} style={{
            background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", 
            cursor: "pointer", color: "#64748b", fontSize: 16, padding: "4px 8px", borderRadius: 6,
            display: "flex", alignItems: "center", transition: "all 0.2s ease",
          }} onMouseEnter={e => e.currentTarget.style.color = "#e2e8f0"}
             onMouseLeave={e => e.currentTarget.style.color = "#64748b"}>
            ‹
          </button>
        )}
      </div>

      {/* Toggle area below logo (collapsed only) */}
      {collapsed && (
        <div style={{ 
          padding: "8px 0", 
          display: "flex", 
          justifyContent: "center",
          borderBottom: "1px solid #1e1e2e",
          background: "rgba(255,255,255,0.01)"
        }}>
          <button onClick={() => setCollapsed(false)} style={{
            background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", 
            cursor: "pointer", color: "#64748b", fontSize: 16, padding: "4px 12px", borderRadius: 6,
            display: "flex", alignItems: "center", transition: "all 0.2s ease",
          }} onMouseEnter={e => e.currentTarget.style.color = "#e2e8f0"}
             onMouseLeave={e => e.currentTarget.style.color = "#64748b"}>
            ›
          </button>
        </div>
      )}

      {/* Nav items */}
      <nav style={{ flex: 1, padding: "12px 0" }}>
        {NAV.map(item => {
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              title={collapsed ? item.label : undefined}
              className="sidebar-link"
              style={{
                width: "100%", background: "none", border: "none", cursor: "pointer",
                padding: collapsed ? "12px 0" : "10px 16px",
                display: "flex", alignItems: "center",
                gap: 12,
                justifyContent: collapsed ? "center" : "flex-start",
                borderLeft: isActive ? "2px solid #7c3aed" : "2px solid transparent",
                background: isActive ? "rgba(124,58,237,0.1)" : "transparent",
                transition: "all 0.2s ease",
                position: "relative",
              }}
              onMouseEnter={e => { 
                if (!isActive) e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                e.currentTarget.style.transform = "translateX(2px)";
              }}
              onMouseLeave={e => { 
                if (!isActive) e.currentTarget.style.background = "transparent";
                e.currentTarget.style.transform = "translateX(0)";
              }}
            >
              <span style={{
                fontSize: 16,
                color: isActive ? "#7c3aed" : "#64748b",
                transition: "color 0.15s",
                minWidth: 20, textAlign: "center",
              }}>{item.icon}</span>
              {!collapsed && (
                <div style={{ textAlign: "left" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: isActive ? "#e2e8f0" : "#64748b", fontFamily: "'Roboto', sans-serif", letterSpacing: 0.5 }}>{item.label}</div>
                  <div style={{ fontSize: 10, color: isActive ? "#64748b" : "#1e1e2e", marginTop: 1, letterSpacing: 0.2 }}>{item.sub}</div>
                </div>
              )}
            </button>
          );
        })}
      </nav>

      {/* Bottom status */}
      {!collapsed && (
        <div style={{
          padding: "16px 20px",
          borderTop: "1px solid #1e1e2e",
          fontSize: 10, color: "#64748b",
          letterSpacing: 1,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: health?.elasticsearch ? "#3fb950" : "#f59e0b",
              display: "inline-block",
              boxShadow: `0 0 8px ${health?.elasticsearch ? "rgba(63,185,80,0.4)" : "rgba(245,158,11,0.35)"}`,
            }} />
            <span style={{ color: health?.elasticsearch ? "#3fb950" : "#f59e0b", fontWeight: 600 }}>
              {health?.elasticsearch ? "ELASTICSEARCH" : "LOCAL MODE"}
            </span>
          </div>
          <div style={{ fontWeight: 500 }}>{health?.elasticsearch ? "localhost:9200" : "Standalone Repository"}</div>
          <div style={{ marginTop: 2, opacity: 0.6 }}>{health?.mode === "local" ? "Processing fallback data" : "v2.4.1 (Stable)"}</div>
        </div>
      )}
      {collapsed && (
        <div style={{ padding: "14px 0", display: "flex", justifyContent: "center", borderTop: "1px solid #1e1e2e" }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%",
            background: health?.elasticsearch ? "#3fb950" : "#f59e0b",
            display: "inline-block",
            boxShadow: `0 0 6px ${health?.elasticsearch ? "#3fb950" : "#f59e0b"}`,
          }} />
        </div>
      )}
    </aside>
  );
}

// ── Placeholder pages ─────────────────────────────────────────────────────────
function PlaceholderPage({ title, icon }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      minHeight: "70vh", gap: 16, color: "#64748b",
    }}>
      <div style={{ fontSize: 48 }}>{icon}</div>
      <div style={{ fontSize: 20, fontFamily: "'Roboto', sans-serif", fontWeight: 700, color: "#e2e8f0" }}>{title}</div>
      <div style={{ fontSize: 13, color: "#64748b" }}>Connect to FastAPI backend to load data</div>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState({ status: "offline", elasticsearch: false, mode: "local" });
  const { route, navigate } = useRoute();

  // Health check polling
  useEffect(() => {
    async function checkStatus() {
      const status = await checkHealth();
      setHealth(status || { status: "offline", elasticsearch: false, mode: "local" });
    }

    // Check immediately on mount
    checkStatus();

    // Poll every 30 seconds
    const interval = setInterval(checkStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  const sidebarWidth = collapsed ? 64 : 220;

  const renderPage = () => {
    switch (route.page) {
      case "dashboard":   return <Dashboard onNavigate={navigate} />;
      case "anomalies":   return <AnomaliesDetail data={route.params} onBack={() => navigate("dashboard")} />;
      case "rootcause":   return <RootCauseDetail data={route.params} onBack={() => navigate("dashboard")} />;
      case "crashes":     return <CrashHistory />;
      case "logs":        return <RawLogs />;
      case "settings":    return <Settings />;
      case "audit":       return <AuditLog />;
      default:            return <Dashboard onNavigate={navigate} />;
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700;900&family=Roboto+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
          background: #0d0d14; 
          color: #e2e8f0; 
          font-family: 'Roboto', sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          text-rendering: optimizeLegibility;
        }
        
        /* Natural Scrollbar */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.1); border-radius: 10px; transition: background 0.3s; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.2); }
        
        .stat-card { 
          transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1); 
          box-shadow: 0 0 12px #7c3aed18;
        }
        .stat-card:hover { 
          transform: translateY(-4px); 
          box-shadow: 0 12px 32px rgba(124,58,237,0.2) !important; 
          cursor: default; 
          border-color: #7c3aed44 !important;
        }
        
        .table-row { transition: background 0.2s ease; }
        .table-row:hover { background: rgba(255,255,255,0.02) !important; }
        
        .action-btn { transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1); }
        .action-btn:hover { 
          transform: translateY(-1px); 
          filter: brightness(1.1); 
          box-shadow: 0 4px 16px rgba(0,0,0,0.2); 
        }

        .sidebar-link { transition: all 0.2s ease; }
        
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={{ display: "flex" }}>
        <Sidebar
          active={route.page}
          onNavigate={(page) => navigate(page)}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          health={health}
        />
        <main style={{
          marginLeft: sidebarWidth,
          flex: 1,
          minHeight: "100vh",
          transition: "margin-left 0.25s cubic-bezier(.4,0,.2,1)",
        }}>
          {health?.mode === "local" && (
            <div style={{
              position: "relative",
              zIndex: 250,
              padding: "12px 24px",
              background: "linear-gradient(90deg, rgba(245,158,11,0.18), rgba(230,115,75,0.12))",
              borderBottom: "1px solid rgba(245,158,11,0.28)",
              color: "#fbbf24",
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: 0.3,
            }}>
              Running in Local Mode. Dashboard data is being served from `system_logs.json` and `ml_results.json` because Elasticsearch is offline.
            </div>
          )}
          {renderPage()}
        </main>
      </div>
    </>
  );
}

export function Toast({ message, type, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3000);
    return () => clearTimeout(t);
  }, []);
  return (
    <div style={{
      position: 'fixed',
      bottom: 24,
      right: 24,
      background: type === 'success' 
        ? '#22c55e' : '#ef4444',
      color: 'white',
      padding: '12px 20px',
      borderRadius: '8px',
      fontSize: '14px',
      zIndex: 9999,
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
      animation: 'fadeSlideUp 0.3s ease'
    }}>
      {toDisplayText(message, "Operation finished")}
    </div>
  );
}
