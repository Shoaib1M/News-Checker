/*
FILE PURPOSE:
This component displays a single piece of evidence (a news article) returned by the scraper.
It shows the article title, domain, source tier, and a relevant passage extracted by NLI.

CHANGES FROM LEGACY:
- Removed misleading "Match" percentage (was lexical overlap, not relevance)
- Added source tier indicator (primary, reporting, fact-check, unclassified)
- Shows best_sentence from NLI-verified passage, not arbitrary snippet
- Explains why this source matters for the claim
- Shows support/contradiction scores instead of generic similarity
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

function getSourceTierBadge(tier) {
  const tierLabels = {
    'primary': 'Primary Source',
    'fact-check': 'Fact Check',
    'reporting': 'News Reporting',
    'unclassified': 'General Source',
  };
  return tierLabels[tier] || 'Source';
}

export default function EvidenceCard({ evidence, index }) {
  const {
    title,
    url,
    stance,
    source,
    best_sentence,
    support_score,
    contradiction_score,
    source_tier,
    nli_available,
  } = evidence;

  const favicon = getFaviconUrl(url);
  const domain = source || getDomain(url);
  const tierLabel = getSourceTierBadge(source_tier);

  // Determine if this evidence supports or contradicts
  const isSupportive = support_score > contradiction_score;
  const strength = Math.max(support_score, contradiction_score);
  const strengthLabel = strength > 0.7 ? 'Strong' : strength > 0.4 ? 'Moderate' : 'Weak';

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
          <div className="evidence-source-info">
            <span className="evidence-domain">{domain}</span>
            <span className="evidence-tier">{tierLabel}</span>
          </div>
        </div>
        <span className={`stance-badge ${stance} ${strengthLabel.toLowerCase()}`}>
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

      {/* Evidence strength indicator */}
      {nli_available && (
        <div className="evidence-strength">
          <span className="strength-label">{strengthLabel} Evidence</span>
          <div className="strength-bar-track">
            <div
              className={`strength-bar-fill ${isSupportive ? 'support' : 'contradiction'}`}
              style={{ width: `${Math.round(strength * 100)}%` }}
            />
          </div>
          <span className="strength-value">{Math.round(strength * 100)}%</span>
        </div>
      )}

      {/* Most relevant passage extracted by NLI */}
      {best_sentence && (
        <div className="evidence-passage">
          <p className="passage-label">Key passage:</p>
          <p className="passage-text">"{best_sentence}"</p>
        </div>
      )}

      {/* Explanation of why this source matters */}
      <div className="evidence-explanation">
        <p className="explanation-text">
          {nli_available ? (
            <>
              This {tierLabel.toLowerCase()} {isSupportive ? 'supports' : 'contradicts'} the claim.
              {source_tier === 'primary' && ' Primary sources carry significant weight.'}
              {source_tier === 'fact-check' && ' Fact-checking organizations provide expert analysis.'}
            </>
          ) : (
            'Evidence classification requires NLI model, which was unavailable.'
          )}
        </p>
      </div>
    </div>
  );
}
