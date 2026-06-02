import { useState } from "react";
import {
  Package, FileText, Brain, CheckCircle2, Globe, Target,
  Layers, Calculator, Code2
} from "lucide-react";

const PIPELINE_STEPS = [
  {
    id: "dataset",
    icon: <Package className="hiw-step-icon" />,
    title: "LIAR Dataset",
    short: "10K+ labelled political statements",
    detail: `The LIAR dataset (Wang, 2017) contains 12,836 short political statements from PolitiFact, 
each labelled by professional fact-checkers as one of six truthfulness levels: pants-fire, false, 
barely-true, half-true, mostly-true, and true.

Each sample includes rich metadata: the speaker's name, job title, state, party affiliation, 
and their historical truth counts across all five non-true categories. We collapse the six labels 
into a binary classification: pants-fire / false / barely-true → "Fake-ish" and half-true / 
mostly-true / true → "True-ish".

Split: 10,240 training · 1,284 validation · 1,267 test samples.`,
  },
  {
    id: "features",
    icon: <FileText className="hiw-step-icon" />,
    title: "TF-IDF Vectorization",
    short: "Convert text to numerical features",
    detail: `We built a custom TF-IDF (Term Frequency–Inverse Document Frequency) vectorizer from scratch — 
no scikit-learn. It generates unigram + bigram features with a minimum document frequency of 2.

For the Binary Truth MLP, we concatenate multiple text fields: statement + subject + speaker + 
job + state + party + context into a single text feature, then apply TF-IDF. Each document 
vector is L2-normalised so all vectors sit on the unit sphere.

We also engineer 5 additional features from the speaker's historical truth counts 
(barely-true, false, half-true, mostly-true, pants-fire), applying log(1+x) scaling and 
min-max normalisation. The final feature vector is [TF-IDF features | history features].`,
  },
  {
    id: "model",
    icon: <Brain className="hiw-step-icon" />,
    title: "MLP Neural Network",
    short: "1 hidden layer, 64 neurons, ReLU + sigmoid",
    detail: `The Binary Truth MLP is a fully-connected neural network built from scratch in NumPy 
(no PyTorch/TensorFlow):

Input Layer → Dense(64, ReLU) → Dense(1, Sigmoid)

Training details:
• Mini-batch SGD with batch size 128
• Learning rate: 0.05
• 70 training epochs
• Binary cross-entropy loss
• Xavier/He weight initialisation
• Threshold tuned on the validation set (not fixed at 0.5)

Backpropagation is implemented manually: we compute gradients for every weight and bias, 
then update using vanilla gradient descent. This is a deliberate choice — we understand and 
can explain every line of the training loop.`,
  },
  {
    id: "classification",
    icon: <CheckCircle2 className="hiw-step-icon" />,
    title: "Binary Classification",
    short: "Fake-ish vs True-ish with tuned threshold",
    detail: `The sigmoid output gives a probability between 0 and 1. Rather than using the standard 
0.5 threshold, we sweep thresholds from 0.30 to 0.70 on the validation set and pick the one 
that maximises accuracy.

This tuned threshold typically lands around 0.45–0.55, and the small adjustment can yield 
1–2% accuracy improvement. The model outputs:
• A probability score (0 = very likely fake, 1 = very likely true)
• A class label based on the tuned threshold
• A human-readable explanation ("probably correct", "uncertain or mixed", etc.)`,
  },
  {
    id: "evidence",
    icon: <Globe className="hiw-step-icon" />,
    title: "Evidence Scraping",
    short: "Multi-source web scraping + stance detection",
    detail: `The ML model alone isn't enough — so we scrape the web in real time for corroborating or 
contradicting evidence:

1. Search: We query DuckDuckGo, GNews, The Guardian API, and NewsAPI to find relevant articles
2. Fetch: Full article text is extracted from up to 12 sources using our custom HTML parser
3. Similarity: Each article is TF-IDF vectorised and compared to the claim via cosine similarity
4. Stance Detection: For each article sentence, we compute:
   • Keyword overlap (relevance)
   • Directional agreement/opposition (e.g., "increase" vs "decrease")
   • Negation detection (flips stance)
   • Numeric alignment (do the numbers match?)
   
Each article gets a support score and contradiction score. The overall evidence stance is 
the net balance across all sources.`,
  },
  {
    id: "scoring",
    icon: <Target className="hiw-step-icon" />,
    title: "Combined Score",
    short: "40% ML + 35% evidence + 25% stance → 0–100",
    detail: `The final credibility score blends three signals:

Formula: score = 40% × ML_confidence + 35% × evidence_similarity + 25% × stance_net

Where:
• ML_confidence: the model's sigmoid output (0–1)
• evidence_similarity: average cosine similarity of top evidence articles (0–1)
• stance_net: (support - contradiction) normalised from [-1,1] to [0,1]

The raw score is scaled to 0–100 and mapped to a verdict:
• 0–25:  Very Likely False
• 26–40: Likely False
• 41–60: Uncertain / Mixed
• 61–75: Likely True
• 76–100: Very Likely True

This weighted approach means: even if the ML model is uncertain, strong web evidence 
supporting the claim can push the score up — and vice versa.`,
  },
];

export default function HowItWorks() {
  const [expanded, setExpanded] = useState(null);

  const toggle = (id) => setExpanded(expanded === id ? null : id);

  return (
    <div className="hiw-page">
      <section className="intro" style={{ animationDelay: "0s" }}>
        <p className="intro-tag">System Architecture · End-to-End Pipeline</p>
        <h2 className="intro-heading">How It Works</h2>
        <p className="intro-desc">
          From raw dataset to credibility score — every step explained.
          Click any stage to see the full technical details.
        </p>
      </section>

      {/* Visual Pipeline */}
      <div className="hiw-pipeline" id="pipeline-diagram">
        {PIPELINE_STEPS.map((step, i) => (
          <div key={step.id} className="hiw-step-wrapper" style={{ animationDelay: `${i * 0.08}s` }}>
            {i > 0 && (
              <div className="hiw-arrow">
                <svg viewBox="0 0 40 20" className="hiw-arrow-svg">
                  <line x1="0" y1="10" x2="32" y2="10" />
                  <polygon points="30,5 40,10 30,15" />
                </svg>
              </div>
            )}
            <button
              className={`hiw-step-card ${expanded === step.id ? "hiw-step-expanded" : ""}`}
              onClick={() => toggle(step.id)}
              id={`step-${step.id}`}
            >
              {step.icon}
              <span className="hiw-step-title">{step.title}</span>
              <span className="hiw-step-short">{step.short}</span>
              <span className="hiw-step-toggle">{expanded === step.id ? "−" : "+"}</span>
            </button>
          </div>
        ))}
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="hiw-detail-panel" id="detail-panel">
          <div className="hiw-detail-header">
            <span className="hiw-detail-icon">
              {PIPELINE_STEPS.find((s) => s.id === expanded)?.icon}
            </span>
            <h3>{PIPELINE_STEPS.find((s) => s.id === expanded)?.title}</h3>
          </div>
          <div className="hiw-detail-body">
            {PIPELINE_STEPS.find((s) => s.id === expanded)?.detail
              .split("\n")
              .map((line, i) => {
                const trimmed = line.trim();
                if (!trimmed) return <br key={i} />;
                if (trimmed.startsWith("•")) {
                  return <li key={i}>{trimmed.slice(1).trim()}</li>;
                }
                return <p key={i}>{trimmed}</p>;
              })}
          </div>
        </div>
      )}

      {/* Architecture Diagram */}
      <div className="hiw-arch-card" id="architecture-diagram">
        <h3 className="eval-section-title">
          <Layers className="eval-icon" size={20} /> Neural Network Architecture
        </h3>
        <div className="hiw-arch-visual">
          <NeuralNetDiagram />
        </div>
        <p className="hiw-arch-caption">
          Binary Truth MLP: input features → 64-neuron hidden layer with ReLU activation → single sigmoid output.
          All weights trained via backpropagation with mini-batch SGD.
        </p>
      </div>

      {/* Scoring Formula */}
      <div className="hiw-formula-card" id="scoring-formula">
        <h3 className="eval-section-title">
          <Calculator className="eval-icon" size={20} /> Scoring Formula
        </h3>
        <div className="hiw-formula">
          <div className="hiw-formula-eq">
            <span className="hiw-f-label">Final Score</span>
            <span className="hiw-f-eq">=</span>
            <span className="hiw-f-term hiw-f-ml">
              <span className="hiw-f-weight">0.40</span>
              <span className="hiw-f-name">ML Score</span>
            </span>
            <span className="hiw-f-op">+</span>
            <span className="hiw-f-term hiw-f-ev">
              <span className="hiw-f-weight">0.35</span>
              <span className="hiw-f-name">Evidence</span>
            </span>
            <span className="hiw-f-op">+</span>
            <span className="hiw-f-term hiw-f-st">
              <span className="hiw-f-weight">0.25</span>
              <span className="hiw-f-name">Stance</span>
            </span>
          </div>
        </div>
        <div className="hiw-formula-legend">
          <div className="hiw-legend-item">
            <span className="hiw-legend-dot" style={{ background: "var(--purple)" }} />
            <span><strong>ML Score</strong> — Model's sigmoid probability (0–1)</span>
          </div>
          <div className="hiw-legend-item">
            <span className="hiw-legend-dot" style={{ background: "var(--blue)" }} />
            <span><strong>Evidence</strong> — Average cosine similarity of top articles (0–1)</span>
          </div>
          <div className="hiw-legend-item">
            <span className="hiw-legend-dot" style={{ background: "var(--green)" }} />
            <span><strong>Stance</strong> — Net support vs contradiction, normalised (0–1)</span>
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="hiw-tech-card" id="tech-stack">
        <h3 className="eval-section-title">
          <Code2 className="eval-icon" size={20} /> Tech Stack
        </h3>
        <div className="hiw-tech-grid">
          {[
            { label: "ML / Data", items: ["Python", "NumPy", "Pandas", "Custom TF-IDF"] },
            { label: "Backend", items: ["FastAPI", "Node.js / Express", "MongoDB"] },
            { label: "Frontend", items: ["React (Vite)", "Vanilla CSS", "Lucide Icons"] },
            { label: "APIs", items: ["DuckDuckGo", "GNews", "NewsAPI", "The Guardian"] },
          ].map((group) => (
            <div className="hiw-tech-group" key={group.label}>
              <h4>{group.label}</h4>
              <ul>
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


// ─── Neural Net Diagram ─────────────────────────────────────────────

function NeuralNetDiagram() {
  const inputCount = 5;
  const hiddenCount = 6;
  const outputCount = 1;
  const width = 500;
  const height = 280;
  const layerX = [80, 250, 420];

  const makeNodes = (count, x, startY, gap) =>
    Array.from({ length: count }, (_, i) => ({
      x,
      y: startY + i * gap,
    }));

  const inputNodes = makeNodes(inputCount, layerX[0], 30, 55);
  const hiddenNodes = makeNodes(hiddenCount, layerX[1], 15, 48);
  const outputNodes = makeNodes(outputCount, layerX[2], height / 2 - 10, 0);

  const inputLabels = ["TF-IDF₁", "TF-IDF₂", "...", "History", "Meta"];
  const hiddenLabels = ["h₁", "h₂", "h₃", "h₄", "h₅", "..."];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="hiw-nn-svg">
      {/* Connections: input → hidden */}
      {inputNodes.map((inp, i) =>
        hiddenNodes.map((hid, j) => (
          <line
            key={`ih-${i}-${j}`}
            x1={inp.x + 20} y1={inp.y}
            x2={hid.x - 20} y2={hid.y}
            className="hiw-nn-conn"
            style={{ opacity: 0.25 + Math.random() * 0.3 }}
          />
        ))
      )}

      {/* Connections: hidden → output */}
      {hiddenNodes.map((hid, i) =>
        outputNodes.map((out, j) => (
          <line
            key={`ho-${i}-${j}`}
            x1={hid.x + 20} y1={hid.y}
            x2={out.x - 20} y2={out.y}
            className="hiw-nn-conn hiw-nn-conn-out"
            style={{ opacity: 0.4 + Math.random() * 0.3 }}
          />
        ))
      )}

      {/* Input nodes */}
      {inputNodes.map((node, i) => (
        <g key={`in-${i}`}>
          <circle cx={node.x} cy={node.y} r={16} className="hiw-nn-node hiw-nn-input" />
          <text x={node.x} y={node.y + 4} className="hiw-nn-label">{inputLabels[i]}</text>
        </g>
      ))}

      {/* Hidden nodes */}
      {hiddenNodes.map((node, i) => (
        <g key={`hid-${i}`}>
          <circle cx={node.x} cy={node.y} r={16} className="hiw-nn-node hiw-nn-hidden" />
          <text x={node.x} y={node.y + 4} className="hiw-nn-label">{hiddenLabels[i]}</text>
        </g>
      ))}

      {/* Output node */}
      {outputNodes.map((node, i) => (
        <g key={`out-${i}`}>
          <circle cx={node.x} cy={node.y} r={20} className="hiw-nn-node hiw-nn-output" />
          <text x={node.x} y={node.y + 4} className="hiw-nn-label hiw-nn-output-label">σ</text>
        </g>
      ))}

      {/* Layer labels */}
      <text x={layerX[0]} y={height - 5} className="hiw-nn-layer-label">Input Layer</text>
      <text x={layerX[1]} y={height - 5} className="hiw-nn-layer-label">Hidden (64, ReLU)</text>
      <text x={layerX[2]} y={height - 5} className="hiw-nn-layer-label">Output (σ)</text>
    </svg>
  );
}
