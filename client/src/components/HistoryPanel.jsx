function getScoreColor(score) {
  if (score <= 25) return { bg: "#fef2f2", color: "#ef4444" };
  if (score <= 40) return { bg: "#fff7ed", color: "#f97316" };
  if (score <= 60) return { bg: "#fffbeb", color: "#f59e0b" };
  if (score <= 75) return { bg: "#ecfdf5", color: "#10b981" };
  return { bg: "#ecfdf5", color: "#059669" };
}

function formatTime(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export default function HistoryPanel({ history, onClose, onSelect, onDelete }) {
  return (
    <>
      <div className="history-overlay" onClick={onClose} />
      <div className="history-panel" id="history-panel">
        <div className="history-header">
          <h3>Check History</h3>
          <button className="btn-close" onClick={onClose} id="btn-close-history">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="history-list">
          {history.length === 0 ? (
            <div className="history-empty">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              <p>No fact-checks yet.</p>
              <p>Your checked statements will appear here.</p>
            </div>
          ) : (
            history.map((item) => {
              const colors = getScoreColor(item.combinedScore);
              return (
                <div
                  className="history-item"
                  key={item._id}
                  onClick={() => onSelect(item)}
                >
                  <div
                    className="history-score-badge"
                    style={{ background: colors.bg, color: colors.color }}
                  >
                    {item.combinedScore}
                  </div>
                  <div className="history-item-content">
                    <div className="history-statement">{item.statement}</div>
                    <div className="history-meta">
                      {item.combinedVerdict} · {formatTime(item.createdAt)}
                    </div>
                  </div>
                  <button
                    className="history-delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(item._id);
                    }}
                    title="Delete"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </>
  );
}
