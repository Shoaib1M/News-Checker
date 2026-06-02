import React, { useState, useEffect } from "react";
import { Database, Grid2x2, TrendingUp, List, Settings } from "lucide-react";

export default function ModelEvaluation() {
  const [data, setData] = useState(null);
  const [selectedModel, setSelectedModel] = useState("binary_mlp");
  const [cmView, setCmView] = useState("binary_mlp");

  useEffect(() => {
    fetch("/evaluation_results.json")
      .then((res) => res.json())
      .then(setData)
      .catch((err) => console.error("Failed to load evaluation data:", err));
  }, []);

  if (!data) {
    return (
      <div className="eval-loading">
        <div className="spinner" style={{ width: 24, height: 24, borderWidth: 3 }} />
        <p>Loading evaluation data…</p>
      </div>
    );
  }

  const { dataset, models } = data;
  const model = models[selectedModel];
  const cmModel = models[cmView];

  return (
    <div className="eval-page">
      {/* Page Header */}
      <section className="intro" style={{ animationDelay: "0s" }}>
        <p className="intro-tag">Model Performance · LIAR Test Set ({dataset.test_size} samples)</p>
        <h2 className="intro-heading">Model Evaluation</h2>
        <p className="intro-desc">
          Real metrics computed on the LIAR test set — not cherry-picked.
          Every number here is reproducible by running <code>evaluate_models.py</code>.
        </p>
      </section>

      {/* Dataset Overview */}
      <div className="eval-dataset-card" id="dataset-overview">
        <h3 className="eval-section-title">
          <Database className="eval-icon" size={20} /> Dataset: {dataset.name}
        </h3>
        <div className="eval-dataset-grid">
          <div className="eval-dataset-stat">
            <span className="eval-stat-number">{dataset.train_size.toLocaleString()}</span>
            <span className="eval-stat-label">Training</span>
          </div>
          <div className="eval-dataset-stat">
            <span className="eval-stat-number">{dataset.valid_size.toLocaleString()}</span>
            <span className="eval-stat-label">Validation</span>
          </div>
          <div className="eval-dataset-stat">
            <span className="eval-stat-number">{dataset.test_size.toLocaleString()}</span>
            <span className="eval-stat-label">Test</span>
          </div>
        </div>
        <div className="eval-label-dist">
          <p className="eval-label-dist-title">Test set label distribution:</p>
          <div className="eval-label-bars">
            {dataset.labels_6class.map((label) => {
              const count = dataset.label_distribution[label] || 0;
              const pct = ((count / dataset.test_size) * 100).toFixed(1);
              return (
                <div className="eval-label-bar-row" key={label}>
                  <span className="eval-label-name">{label}</span>
                  <div className="eval-label-bar-track">
                    <div
                      className="eval-label-bar-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="eval-label-count">{count} ({pct}%)</span>
                </div>
              );
            })}
          </div>
        </div>
        <p className="eval-binary-note">{dataset.binary_mapping}</p>
      </div>

      {/* Model Selector */}
      <div className="eval-model-selector" id="model-selector">
        {Object.entries(models).map(([key, m]) => (
          <button
            key={key}
            className={`eval-model-tab ${selectedModel === key ? "active" : ""}`}
            onClick={() => setSelectedModel(key)}
          >
            {m.name}
            {m.is_production && <span className="eval-prod-badge">IN USE</span>}
          </button>
        ))}
      </div>

      {/* Metrics Dashboard */}
      <div className="eval-metrics-grid" id="metrics-dashboard">
        <MetricCard label="Accuracy" value={model.accuracy} color="var(--purple)" />
        <MetricCard label="Precision" value={model.precision} color="var(--blue)" />
        <MetricCard label="Recall" value={model.recall} color="var(--green)" />
        <MetricCard label="F1 Score" value={model.f1} color="var(--amber)" />
      </div>

      {/* Confusion Matrix + ROC side by side */}
      <div className="eval-charts-row">
        {/* Confusion Matrix */}
        <div className="eval-chart-card" id="confusion-matrix">
          <div className="eval-chart-header">
            <h3 className="eval-section-title">
              <Grid2x2 className="eval-icon" size={20} /> Confusion Matrix
            </h3>
            <div className="eval-cm-toggle">
              {Object.entries(models).map(([key, m]) => (
                <button
                  key={key}
                  className={`eval-cm-btn ${cmView === key ? "active" : ""}`}
                  onClick={() => setCmView(key)}
                >
                  {key === "binary_mlp" ? "Binary MLP" : key === "logistic_regression" ? "Log. Reg." : "6-Class"}
                </button>
              ))}
            </div>
          </div>
          <ConfusionMatrix
            matrix={cmModel.confusion_matrix}
            labels={cmModel.labels}
          />
        </div>

        {/* ROC Curve */}
        {model.roc_curve && (
          <div className="eval-chart-card" id="roc-curve">
            <h3 className="eval-section-title">
              <TrendingUp className="eval-icon" size={20} /> ROC Curve
              <span className="eval-auc-badge">AUC = {model.auc}</span>
            </h3>
            <ROCCurve points={model.roc_curve} auc={model.auc} />
            {model.threshold && (
              <p className="eval-threshold-note">
                Decision threshold: <strong>{model.threshold}</strong>
                {model.is_production && " (tuned on validation set)"}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Per-class metrics for 6-class */}
      {model.per_class_metrics && (
        <div className="eval-chart-card eval-full-width" id="per-class-metrics">
          <h3 className="eval-section-title">
            <List className="eval-icon" size={20} /> Per-Class Metrics
          </h3>
          <div className="eval-per-class-table">
            <div className="eval-pc-header">
              <span>Label</span>
              <span>Precision</span>
              <span>Recall</span>
              <span>F1</span>
              <span>Support</span>
            </div>
            {model.per_class_metrics.map((m, i) => (
              <div className="eval-pc-row" key={i} style={{ animationDelay: `${i * 0.05}s` }}>
                <span className="eval-pc-label">{model.labels[i]}</span>
                <span><BarValue value={m.precision} color="var(--blue)" /></span>
                <span><BarValue value={m.recall} color="var(--green)" /></span>
                <span><BarValue value={m.f1} color="var(--amber)" /></span>
                <span className="eval-pc-support">{m.support}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model details */}
      <div className="eval-chart-card eval-full-width" id="model-details">
        <h3 className="eval-section-title">
          <Settings className="eval-icon" size={20} /> Model Details — {model.name}
        </h3>
        <div className="eval-details-grid">
          <div className="eval-detail-item">
            <span className="eval-detail-label">Architecture</span>
            <span className="eval-detail-value">{model.architecture}</span>
          </div>
          <div className="eval-detail-item">
            <span className="eval-detail-label">Input Features</span>
            <span className="eval-detail-value">{model.input_features}</span>
          </div>
          <div className="eval-detail-item">
            <span className="eval-detail-label">Training</span>
            <span className="eval-detail-value">{model.training}</span>
          </div>
          <div className="eval-detail-item">
            <span className="eval-detail-label">Output Classes</span>
            <span className="eval-detail-value">{model.classes}</span>
          </div>
        </div>
      </div>
    </div>
  );
}


// ─── Sub-components ──────────────────────────────────────────────────

function MetricCard({ label, value, color }) {
  const pct = Math.round(value * 100);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value * circumference);

  return (
    <div className="eval-metric-card">
      <div className="eval-metric-gauge">
        <svg viewBox="0 0 100 100" className="eval-metric-svg">
          <circle cx="50" cy="50" r={radius} className="eval-metric-bg" />
          <circle
            cx="50" cy="50" r={radius}
            className="eval-metric-fill"
            style={{
              stroke: color,
              strokeDasharray: circumference,
              strokeDashoffset: offset,
            }}
          />
        </svg>
        <span className="eval-metric-number" style={{ color }}>{pct}%</span>
      </div>
      <span className="eval-metric-label">{label}</span>
    </div>
  );
}


function ConfusionMatrix({ matrix, labels }) {
  const maxVal = Math.max(...matrix.flat());
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const isLarge = labels.length > 2;

  return (
    <div className={`eval-cm ${isLarge ? "eval-cm-large" : ""}`}>
      {/* Column headers */}
      <div className="eval-cm-corner">
        <span className="eval-cm-axis-label eval-cm-axis-pred">Predicted →</span>
        <span className="eval-cm-axis-label eval-cm-axis-actual">Actual ↓</span>
      </div>
      {labels.map((l) => (
        <div className="eval-cm-col-label" key={`col-${l}`}>
          {isLarge ? l.replace("barely-", "b-").replace("mostly-", "m-").replace("pants-fire", "p-fire") : l}
        </div>
      ))}
      {/* Rows */}
      {matrix.map((row, i) => (
        <React.Fragment key={`row-${i}`}>
          <div className="eval-cm-row-label">
            {isLarge ? labels[i].replace("barely-", "b-").replace("mostly-", "m-").replace("pants-fire", "p-fire") : labels[i]}
          </div>
          {row.map((val, j) => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiag = i === j;
            const pct = total > 0 ? ((val / total) * 100).toFixed(1) : "0";
            return (
              <div
                className={`eval-cm-cell ${isDiag ? "eval-cm-diag" : ""}`}
                key={`${i}-${j}`}
                style={{
                  "--intensity": intensity,
                  backgroundColor: isDiag
                    ? `rgba(124, 58, 237, ${0.1 + intensity * 0.55})`
                    : `rgba(220, 38, 38, ${intensity * 0.35})`,
                }}
                title={`Actual: ${labels[i]}, Predicted: ${labels[j]}\n${val} samples (${pct}%)`}
              >
                <span className="eval-cm-val">{val}</span>
                <span className="eval-cm-pct">{pct}%</span>
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
}


function ROCCurve({ points, auc }) {
  const width = 300;
  const height = 300;
  const pad = 40;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;

  const toX = (fpr) => pad + fpr * innerW;
  const toY = (tpr) => pad + (1 - tpr) * innerH;

  const polyline = points
    .map((p) => `${toX(p.fpr)},${toY(p.tpr)}`)
    .join(" ");

  // Area fill path
  const areaPath = `M ${toX(0)},${toY(0)} ` +
    points.map((p) => `L ${toX(p.fpr)},${toY(p.tpr)}`).join(" ") +
    ` L ${toX(points[points.length - 1]?.fpr || 1)},${toY(0)} Z`;

  return (
    <div className="eval-roc-container">
      <svg viewBox={`0 0 ${width} ${height}`} className="eval-roc-svg">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((v) => (
          <g key={v}>
            <line x1={toX(v)} y1={toY(0)} x2={toX(v)} y2={toY(1)} className="eval-roc-grid" />
            <line x1={toX(0)} y1={toY(v)} x2={toX(1)} y2={toY(v)} className="eval-roc-grid" />
          </g>
        ))}

        {/* Baseline diagonal */}
        <line
          x1={toX(0)} y1={toY(0)}
          x2={toX(1)} y2={toY(1)}
          className="eval-roc-baseline"
        />
        <text x={toX(0.62)} y={toY(0.55)} className="eval-roc-baseline-label">Random</text>

        {/* Area under curve */}
        <path d={areaPath} className="eval-roc-area" />

        {/* Curve */}
        <polyline points={polyline} className="eval-roc-line" />

        {/* Axes */}
        <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(0)} className="eval-roc-axis" />
        <line x1={toX(0)} y1={toY(0)} x2={toX(0)} y2={toY(1)} className="eval-roc-axis" />

        {/* Axis labels */}
        <text x={toX(0.5)} y={height - 4} className="eval-roc-axis-text" textAnchor="middle">
          False Positive Rate
        </text>
        <text x={8} y={toY(0.5)} className="eval-roc-axis-text" textAnchor="middle"
          transform={`rotate(-90, 8, ${toY(0.5)})`}>
          True Positive Rate
        </text>

        {/* Tick labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={`tick-${v}`}>
            <text x={toX(v)} y={toY(0) + 14} className="eval-roc-tick" textAnchor="middle">{v}</text>
            <text x={toX(0) - 6} y={toY(v) + 4} className="eval-roc-tick" textAnchor="end">{v}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}


function BarValue({ value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="eval-bar-value">
      <div className="eval-bar-track">
        <div className="eval-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="eval-bar-num">{pct}%</span>
    </div>
  );
}
