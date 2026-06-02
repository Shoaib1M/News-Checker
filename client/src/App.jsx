import { useState, useEffect, useCallback, useRef } from "react";
import "./App.css";
import Header from "./components/Header";
import ScoreGauge from "./components/ScoreGauge";
import ScoreBreakdown from "./components/ScoreBreakdown";
import EvidenceCard from "./components/EvidenceCard";
import LoadingSkeleton from "./components/LoadingSkeleton";
import HistoryPanel from "./components/HistoryPanel";
import ModelEvaluation from "./components/ModelEvaluation";
import ModelComparison from "./components/ModelComparison";
import HowItWorks from "./components/HowItWorks";

const API_BASE = import.meta.env.VITE_API_URL || "";

// ─── Hash Router ────────────────────────────────────────────────────
function useHashRouter() {
  const getPage = () => {
    const hash = window.location.hash.replace("#/", "").replace("#", "");
    return hash || "";
  };

  const [page, setPage] = useState(getPage);

  useEffect(() => {
    const handler = () => setPage(getPage());
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const navigate = (p) => {
    window.location.hash = `#/${p}`;
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return [page, navigate];
}

function App() {
  const [page, navigate] = useHashRouter();
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("nc_token"));
  const [statement, setStatement] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const textareaRef = useRef(null);

  // ─── Restore user session ───────────────────────────────────────────
  useEffect(() => {
    if (!token) return;
    fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Session expired");
        return res.json();
      })
      .then(setUser)
      .catch(() => {
        localStorage.removeItem("nc_token");
        setToken(null);
        setUser(null);
      });
  }, [token]);

  // ─── Initialize Google Sign-In ──────────────────────────────────────
  useEffect(() => {
    if (user) return;
    const initGsi = () => {
      if (!window.google?.accounts?.id) return;
      const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      if (!clientId) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredential,
      });
      const btnContainer = document.getElementById("google-signin-btn");
      if (btnContainer) {
        window.google.accounts.id.renderButton(btnContainer, {
          theme: "outline",
          size: "medium",
          shape: "pill",
          text: "signin_with",
        });
      }
    };
    if (window.google?.accounts?.id) {
      initGsi();
    } else {
      const timer = setInterval(() => {
        if (window.google?.accounts?.id) {
          clearInterval(timer);
          initGsi();
        }
      }, 200);
      return () => clearInterval(timer);
    }
  }, [user]);

  const handleGoogleCredential = async (response) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential: response.credential }),
      });
      if (!res.ok) throw new Error("Auth failed");
      const data = await res.json();
      localStorage.setItem("nc_token", data.token);
      setToken(data.token);
      setUser(data.user);
    } catch (err) {
      console.error("Google sign-in failed:", err);
      setError("Sign-in failed. Please try again.");
    }
  };

  const handleSignOut = () => {
    localStorage.removeItem("nc_token");
    setToken(null);
    setUser(null);
    setHistory([]);
    if (window.google?.accounts?.id) {
      window.google.accounts.id.disableAutoSelect();
    }
  };

  // ─── Fetch history ──────────────────────────────────────────────────
  const fetchHistory = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/api/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setHistory(data.checks || []);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  }, [token]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  // ─── Fact-check submission ──────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = statement.trim();
    if (trimmed.length < 5) {
      setError("Enter at least 5 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const res = await fetch(`${API_BASE}/api/check`, {
        method: "POST",
        headers,
        body: JSON.stringify({ statement: trimmed }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      if (token) fetchHistory();
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleHistorySelect = async (item) => {
    setHistoryOpen(false);
    try {
      const res = await fetch(`${API_BASE}/api/history/${item._id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error("Failed to load");
      const full = await res.json();
      setStatement(full.statement);
      setResult({
        statement: full.statement,
        ml_score: full.mlScore,
        ml_verdict: full.mlVerdict,
        evidence_score: full.evidenceScore,
        evidence_stance: full.evidenceStance,
        combined_score: full.combinedScore,
        combined_verdict: full.combinedVerdict,
        top_evidence: full.topEvidence,
        processing_time_seconds: full.processingTime,
      });
    } catch {
      setStatement(item.statement);
    }
  };

  const handleHistoryDelete = async (id) => {
    try {
      await fetch(`${API_BASE}/api/history/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      setHistory((prev) => prev.filter((h) => h._id !== id));
    } catch (err) {
      console.error("Failed to delete:", err);
    }
  };

  // ─── Route-based rendering ──────────────────────────────────────────
  const renderPage = () => {
    switch (page) {
      case "evaluation":
        return <ModelEvaluation />;
      case "comparison":
        return <ModelComparison />;
      case "how-it-works":
        return <HowItWorks />;
      default:
        return renderHome();
    }
  };

  const renderHome = () => (
    <>
      {/* Intro */}
      <section className="intro">
        <p className="intro-tag">Built on the LIAR dataset · MLP + live web evidence</p>
        <h2 className="intro-heading">
          Paste a claim. I'll tell you if it holds up.
        </h2>
        <p className="intro-desc">
          My model scores the statement, then I scrape the web for supporting
          or contradicting evidence and combine both into a single credibility score.
        </p>
      </section>

      {/* Input */}
      <form className="input-card" onSubmit={handleSubmit} id="check-form">
        <textarea
          id="statement-input"
          ref={textareaRef}
          className="input-textarea"
          placeholder="e.g. The United States spends more on its military than the next 10 countries combined."
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          maxLength={2000}
          disabled={loading}
        />
        <div className="input-footer">
          <span className="char-count">{statement.length}/2000</span>
          <button
            type="submit"
            className="btn-check"
            disabled={loading || statement.trim().length < 5}
            id="btn-check"
          >
            {loading ? (
              <>
                <span className="spinner" />
                Checking…
              </>
            ) : (
              "Check this"
            )}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="error-banner" id="error-message">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && <LoadingSkeleton />}

      {/* Results */}
      {result && !loading && (
        <section className="results" id="results-section">
          <div className="results-label">
            <span>Result</span>
            {result.processing_time_seconds && (
              <span className="results-time">{result.processing_time_seconds}s</span>
            )}
          </div>

          <div className="score-card">
            <ScoreGauge
              score={result.combined_score}
              verdict={result.combined_verdict}
            />
            <div className="score-details">
              <p className="verdict-text" data-score={result.combined_score}>
                {result.combined_verdict}
              </p>
              <p className="verdict-statement">
                "{result.statement}"
              </p>
              <ScoreBreakdown
                mlScore={result.ml_score}
                evidenceScore={result.evidence_score}
                stanceNet={result.evidence_stance?.net || 0}
              />
            </div>
          </div>

          {result.top_evidence && result.top_evidence.length > 0 && (
            <div className="evidence-section" id="evidence-section">
              <p className="section-label">
                Sources found — {result.top_evidence.length} articles
              </p>
              <div className="evidence-grid">
                {result.top_evidence.map((ev, i) => (
                  <EvidenceCard key={ev.url || i} evidence={ev} index={i} />
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </>
  );

  return (
    <div className="app-layout">
      <Header
        user={user}
        onSignOut={handleSignOut}
        onHistoryToggle={() => setHistoryOpen(!historyOpen)}
        historyCount={history.length}
        currentPage={page}
        onNavigate={navigate}
      />

      <main className="app-main">
        {renderPage()}
      </main>

      <footer className="app-footer">
        <p>
          Built by Shoaib — MLP trained on the LIAR dataset ·
          Evidence scraped in real time via DuckDuckGo, GNews, Guardian &amp; NewsAPI
        </p>
      </footer>

      {historyOpen && (
        <HistoryPanel
          history={history}
          onClose={() => setHistoryOpen(false)}
          onSelect={handleHistorySelect}
          onDelete={handleHistoryDelete}
        />
      )}
    </div>
  );
}

export default App;
