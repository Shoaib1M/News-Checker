/*
FILE PURPOSE:
This is the root component of the entire React Frontend.
It holds the main application state (user, token, the statement being checked, results, and history).
It also acts as the central router, deciding which page or component to show.

FLOW:
1. useHashRouter: Manages simple navigation without needing a complex library like React Router.
2. User Session: Checks if the user is logged in via Google OAuth.
3. API Calls: Handles sending statements to the backend and fetching history.
4. Render: Displays the Header, the main content (Home, Evaluation, Comparison, etc.), and the Footer.

WHY THIS EXISTS:
We need a central "brain" for the frontend to manage the state that is shared across multiple components
(e.g., the Header needs to know if the user is logged in, and the Home page needs to know the check results).
*/

import { useState, useEffect, useCallback, useRef } from "react";
import "./App.css";

// Import all the modular UI pieces (components)
import Header from "./components/Header";
import ScoreGauge from "./components/ScoreGauge";
import ScoreBreakdown from "./components/ScoreBreakdown";
import EvidenceCard from "./components/EvidenceCard";
import LoadingSkeleton from "./components/LoadingSkeleton";
import HistoryPanel from "./components/HistoryPanel";
import ModelEvaluation from "./components/ModelEvaluation";
import ModelComparison from "./components/ModelComparison";
import HowItWorks from "./components/HowItWorks";

// The backend API URL (e.g., http://localhost:3000)
const API_BASE = import.meta.env.VITE_API_URL || "";

// ─── Hash Router ────────────────────────────────────────────────────
/*
PURPOSE: A custom, lightweight router using the URL hash (e.g., #/how-it-works).
WHY THIS EXISTS: Keeps the URL updated and allows users to use the Back button, 
without needing a heavy third-party library like `react-router-dom` for a simple app.
*/
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
  // ─── STATE MANAGEMENT ────────────────────────────────────────────────

  // Router state
  const [page, navigate] = useHashRouter();
  
  // Auth state
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("nc_token"));
  
  // Fact-check state
  const [statement, setStatement] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  // History state
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState([]);
  
  const textareaRef = useRef(null);

  // ─── Restore user session ───────────────────────────────────────────
  /*
  PURPOSE: Automatically logs the user back in if they refresh the page.
  It sends their saved token to the backend to verify it's still valid.
  */
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
        // If the token is invalid (expired/tampered), clear it out
        localStorage.removeItem("nc_token");
        setToken(null);
        setUser(null);
      });
  }, [token]);

  // ─── Initialize Google Sign-In ──────────────────────────────────────
  /*
  PURPOSE: Loads the official Google Sign-In button.
  WHY THIS EXISTS: We don't want to manage passwords ourselves. 
  Google handles the security and just gives us a verified token.
  */
  useEffect(() => {
    if (user) return;
    const initGsi = () => {
      if (!window.google?.accounts?.id) return;
      const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      if (!clientId) return;
      
      // Tell Google what to do when the user successfully logs in
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredential,
      });
      
      // Draw the actual button inside our HTML container
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
    
    // Wait until Google's external script has finished loading
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

  /*
  PURPOSE: Takes the Google token and exchanges it for OUR backend token.
  */
  const handleGoogleCredential = async (response) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential: response.credential }),
      });
      if (!res.ok) throw new Error("Auth failed");
      const data = await res.json();
      
      // Save our custom JWT to localStorage so they stay logged in
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
  /*
  PURPOSE: Grabs the user's past fact-checks from the database.
  */
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
  /*
  PURPOSE: Sends the user's typed statement to our backend to be checked.
  */
  const handleSubmit = async (e) => {
    e.preventDefault(); // Stop the page from refreshing on form submit
    
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
      if (token) headers["Authorization"] = `Bearer ${token}`; // Send token if logged in
      
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
      
      if (token) fetchHistory(); // Refresh history sidebar if logged in
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  /*
  PURPOSE: When a user clicks a past item in their history sidebar, load it into the main view.
  */
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
        assessment_status: full.assessmentStatus,
        claim_assessments: full.claimAssessments,
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
  /*
  PURPOSE: Switches which page component is shown based on the URL hash.
  */
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

      <aside className="claim-examples" aria-label="Claim examples">
        <p className="claim-examples-title">Few examples</p>
        <p><span>"Water freezes at 0°C at sea level."</span> — true</p>
        <p><span>"A four-day workweek improves productivity."</span> — middle (50%) ish</p>
        <p><span>"The Great Wall of China is visible from the Moon with the naked eye."</span> — false</p>
      </aside>

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

          {/* Core Score UI */}
          <div className="score-card">
            <ScoreGauge
              score={result.combined_score}
              verdict={result.combined_verdict}
              assessmentStatus={result.assessment_status}
            />
            <div className="score-details">
              <p className="verdict-text" data-score={result.combined_score}>
                {result.combined_verdict}
              </p>
              <p className="verdict-statement">
                "{result.statement}"
              </p>
              {result.assessment_status === "insufficient_evidence" && (
                <p className="assessment-note">
                  No verdict was made: there is not enough classified, NLI-checked evidence yet.
                </p>
              )}
              <ScoreBreakdown
                mlScore={result.ml_score}
                evidenceScore={result.evidence_score}
                stanceNet={result.evidence_stance?.net || 0}
              />
            </div>
          </div>

          {/* Evidence Articles */}
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
