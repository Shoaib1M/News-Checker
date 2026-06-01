function getFaviconUrl(url) {
  try {
    const domain = new URL(url).hostname;
    return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
  } catch {
    return null;
  }
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace("www.", "");
  } catch {
    return url;
  }
}

export default function EvidenceCard({ evidence, index }) {
  const {
    title,
    url,
    similarity,
    stance,
    source,
    best_sentence,
  } = evidence;

  const favicon = getFaviconUrl(url);
  const domain = source || getDomain(url);

  return (
    <div
      className="evidence-card"
      style={{ animationDelay: `${index * 0.08}s` }}
      id={`evidence-card-${index}`}
    >
      <div className="evidence-card-header">
        <div className="evidence-source">
          {favicon && (
            <img
              src={favicon}
              alt=""
              className="evidence-favicon"
              loading="lazy"
              onError={(e) => { e.target.style.display = "none"; }}
            />
          )}
          <span className="evidence-domain">{domain}</span>
        </div>
        <span className={`stance-badge ${stance}`}>
          {stance}
        </span>
      </div>

      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="evidence-link"
      >
        {title || "Untitled"}
      </a>

      <div className="evidence-similarity">
        <span className="similarity-label">Match</span>
        <div className="similarity-bar-track">
          <div
            className="similarity-bar-fill"
            style={{ width: `${Math.round(similarity * 100)}%` }}
          />
        </div>
        <span className="similarity-value">
          {Math.round(similarity * 100)}%
        </span>
      </div>

      {best_sentence && (
        <p className="evidence-quote">{best_sentence}</p>
      )}
    </div>
  );
}
