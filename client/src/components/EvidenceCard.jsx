/*
FILE PURPOSE:
This component displays a single piece of evidence (a news article) returned by the scraper.
It shows the article title, domain, favicon, how closely it matches the claim, and a key quote.

FLOW:
1. Receives the `evidence` object as a prop.
2. Extracts the clean domain name (e.g., "nytimes.com").
3. Fetches the website's favicon using a Google API.
4. Renders a card with a visual similarity bar and a badge for its stance (Supports/Contradicts).

WHY THIS EXISTS:
We need a clean, reusable way to display the source articles so the user can verify the AI's logic.
*/

function getFaviconUrl(url) {
  try {
    const domain = new URL(url).hostname;
    // We use Google's free favicon service to grab the site's logo automatically.
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
      // Stagger the entrance animation based on the item's index (0.08s, 0.16s, etc.)
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
              // If the favicon fails to load, just hide the broken image icon
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

      {/* Visual bar showing how relevant the article is to the claim */}
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
