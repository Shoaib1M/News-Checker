import { CheckCircle2, X } from "lucide-react";

export default function Header({ user, onSignOut, onHistoryToggle, historyCount, currentPage, onNavigate }) {
  const navItems = [
    { key: "", label: "Check" },
    { key: "evaluation", label: "Evaluation" },
    { key: "comparison", label: "Comparison" },
    { key: "how-it-works", label: "How It Works" },
  ];

  return (
    <header className="header">
      <div className="header-inner">
        <a href="#/" className="header-brand" onClick={(e) => { e.preventDefault(); onNavigate(""); }}>
          <span className="header-mark"><CheckCircle2 size={18} /></span>
          <span className="header-title">newschecker</span>
        </a>

        <nav className="header-nav">
          {navItems.map((item) => (
            <a
              key={item.key}
              href={`#/${item.key}`}
              className={`header-nav-link ${currentPage === item.key ? "active" : ""}`}
              onClick={(e) => {
                e.preventDefault();
                onNavigate(item.key);
              }}
            >
              {item.label}
            </a>
          ))}
        </nav>

        <div className="header-actions">
          {user && (
            <button className="btn-history" onClick={onHistoryToggle} id="btn-toggle-history">
              History{historyCount > 0 && ` (${historyCount})`}
            </button>
          )}

          <div className="auth-section">
            {user ? (
              <div className="user-pill">
                {user.avatar ? (
                  <img src={user.avatar} alt="" className="user-avatar" referrerPolicy="no-referrer" />
                ) : (
                  <span className="user-avatar-placeholder">
                    {user.name?.[0]?.toUpperCase() || "?"}
                  </span>
                )}
                <span className="user-name">{user.name?.split(" ")[0]}</span>
                <button className="btn-signout" onClick={onSignOut} id="btn-signout" aria-label="Sign out">
                  <X size={14} />
                </button>
              </div>
            ) : (
              <div id="google-signin-btn" />
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
