import { useEffect, useMemo, useState, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Center, OrbitControls, useGLTF } from "@react-three/drei";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000").replace(/\/+$/, "");

const RISK_INFO = [
  { key: "cad", label: "Overall CAD", short: "CAD", color: "#B7A5F5", description: "Coronary artery disease" },
  { key: "lad", label: "LAD", short: "LAD", color: "#F2A7B5", description: "Left anterior descending artery" },
  { key: "lcx", label: "LCX", short: "LCX", color: "#91D8C3", description: "Left circumflex artery" },
  { key: "rca", label: "RCA", short: "RCA", color: "#92C5F5", description: "Right coronary artery" },
];

const VESSELS = ["LAD", "LCX", "RCA"];

function HeartModel({ selectedVessel }) {
  const { scene } = useGLTF("/models/Cardio_AI_Heart_Labeled.glb");

  const model = useMemo(() => {
    const copy = scene.clone(true);
    copy.traverse((child) => {
      if (!child.isMesh) return;
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => material.clone())
        : child.material.clone();
    });
    return copy;
  }, [scene]);

  useEffect(() => {
    model.traverse((child) => {
      if (!child.isMesh) return;
      const name = (child.name || "").toLowerCase();
      const isLAD = name.includes("lad");
      const isLCX = name.includes("lcx");
      const isRCA = name.includes("rca");
      const isArtery = isLAD || isLCX || isRCA;
      const selected =
        (selectedVessel === "LAD" && isLAD) ||
        (selectedVessel === "LCX" && isLCX) ||
        (selectedVessel === "RCA" && isRCA);

      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        if (!material) return;
        if (isArtery) {
          material.map = null;
          material.color.set(selected ? "#ff1744" : "#00b8d9");
          material.emissive?.set(selected ? "#8a001c" : "#004b60");
          material.emissiveIntensity = selected ? 0.65 : 0.25;
          material.roughness = 0.4;
          material.metalness = 0;
        } else {
          material.color.set("#c98288");
          material.emissive?.set("#000000");
          material.emissiveIntensity = 0;
        }
        material.needsUpdate = true;
      });
    });
  }, [model, selectedVessel]);

  return <Center><primitive object={model} scale={1.5} /></Center>;
}

useGLTF.preload("/models/Cardio_AI_Heart_Labeled.glb");

function getPrediction(prediction, key) {
  return prediction?.predictions?.[key];
}


function getPercent(prediction, key) {
  if (!prediction) return null;

  const raw = getPrediction(prediction, key)?.probability;

  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    return null;
  }

  return Math.round(
    Math.max(0, Math.min(100, raw <= 1 ? raw * 100 : raw))
  );
}


function getRiskLabel(prediction, key, value) {
  const backendRisk = getPrediction(prediction, key)?.risk;
  if (typeof backendRisk === "string" && backendRisk.trim()) return backendRisk;
  if (value === null) return "Awaiting";
  return value >= 65 ? "High" : value >= 35 ? "Moderate" : "Low";
}

function RiskCard({ item, value, risk }) {
  const displayValue = value === null ? "—" : value;
  return (
    <article className={`risk-card risk-${item.short.toLowerCase()}`}>
      <div className="risk-card-top">
        <span className="risk-icon" style={{ background: `${item.color}55` }}>{item.short}</span>
        <span className={`risk-pill ${risk.toLowerCase()}`}>{risk}</span>
      </div>
      <p className="risk-card-label">{item.label}</p>
      <div className="risk-value">{displayValue}<span>{value === null ? "" : "%"}</span></div>
      <div className="risk-track" role="progressbar" aria-label={`${item.label} probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value ?? 0}>
        <div className="risk-fill" style={{ width: `${value ?? 0}%`, background: item.color }} />
      </div>
      <p className="risk-description">{item.description}</p>
    </article>
  );
}

function BarChart({ values }) {
  return (
    <div className="bar-chart" role="img" aria-label="Bar chart of model probability estimates">
      {RISK_INFO.map((item) => (
        <div className="bar-row" key={item.key}>
          <div className="bar-label">{item.short}</div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${values[item.key] ?? 0}%`, background: item.color }} />
          </div>
          <strong className="bar-value">{values[item.key] === null ? "—" : `${values[item.key]}%`}</strong>
        </div>
      ))}
      <div className="chart-scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
    </div>
  );
}

function DonutChart({ values }) {
  const available = RISK_INFO.filter((item) => values[item.key] !== null);
  const total = available.reduce((sum, item) => sum + values[item.key], 0);
  const segments = available.map((item, index) => ({
    ...item,
    portion: total > 0 ? (values[item.key] / total) * 100 : 0,
    start: available.slice(0, index).reduce(
      (offset, previous) => offset + (total > 0 ? (values[previous.key] / total) * 100 : 0),
      0,
    ),
  }));
  const gradient = segments.length
    ? segments.map((segment) => `${segment.color} ${segment.start}% ${segment.start + segment.portion}%`).join(", ")
    : "#e9e5f2 0% 100%";

  return (
    <div className="donut-layout">
      <div className="donut" style={{ background: `conic-gradient(${gradient})` }} role="img" aria-label="Relative share of available displayed probability estimates">
        <div className="donut-center">
          <strong>{available.length ? `${Math.round(total / available.length)}%` : "—"}</strong>
          <span>{available.length ? "Mean shown" : "No result"}</span>
        </div>
      </div>
      <div className="donut-legend">
        {RISK_INFO.map((item) => (
          <div className="legend-row" key={item.key}>
            <span className="legend-dot" style={{ background: item.color }} />
            <span>{item.short}</span>
            <strong>{values[item.key] === null ? "—" : `${values[item.key]}%`}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function initialValue(item) {
  if (item.default !== undefined && item.default !== null) return item.default;
  if (item.default_value !== undefined && item.default_value !== null) return item.default_value;
  if (Array.isArray(item.allowed_options) && item.allowed_options.length) return item.allowed_options[0];
  if (item.type === "integer" || item.type === "number" || item.type === "float") return item.minimum ?? 0;
  return "";
}

function readableName(item) {
  return item.display_name || item.feature_name.replaceAll("_", " ");
}

function FeatureInput({ item, value, onChange }) {
  const options = item.allowed_options;
  const label = readableName(item);
  const isNumeric = ["integer", "number", "float", "numeric"].includes(String(item.type || "").toLowerCase());

  return (
    <label className="clinical-field">
      <span>{label}</span>
      {Array.isArray(options) && options.length > 0 ? (
        <select name={item.feature_name} value={value ?? ""} onChange={(event) => onChange(item, event.target.value)}>
          {options.map((option) => <option key={String(option)} value={option}>{String(option)}</option>)}
        </select>
      ) : (
        <div className="input-wrap">
          <input
            name={item.feature_name}
            type={isNumeric ? "number" : "text"}
            min={item.minimum ?? undefined}
            max={item.maximum ?? undefined}
            step={String(item.type).toLowerCase() === "integer" ? "1" : "any"}
            value={value ?? ""}
            required
            onChange={(event) => onChange(item, event.target.value)}
            aria-label={label}
          />
          {item.unit && <span>{item.unit}</span>}
        </div>
      )}
      {item.minimum !== null && item.minimum !== undefined && item.maximum !== null && item.maximum !== undefined && (
        <small className="field-range">Allowed range: {item.minimum}–{item.maximum}</small>
      )}
    </label>
  );
}

function ExplanationPanel({ prediction }) {
  const explanationTargets = prediction?.explanations || {};
  const targetKeys = Object.keys(explanationTargets);
  const [target, setTarget] = useState("cad");
  const selectedTarget = targetKeys.includes(target) ? target : targetKeys[0] || "cad";

  const rows = Array.isArray(explanationTargets[selectedTarget]) ? explanationTargets[selectedTarget] : [];
  const maxAbs = Math.max(0.000001, ...rows.map((row) => Math.abs(Number(row.contribution) || 0)));

  return (
    <article className="panel explanation-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MODEL INTERPRETABILITY</span>
          <h2>Why did the model predict this?</h2>
        </div>
        <span className="panel-symbol">✳</span>
      </div>
      <p className="panel-description">
        SHAP feature contributions for the selected model output. Positive values push the model output upward; negative values push it downward.
      </p>
      {!prediction ? (
        <div className="explanation-empty">Run an assessment to see the model's feature contributions.</div>
      ) : targetKeys.length === 0 ? (
        <div className="explanation-empty">The backend did not return SHAP explanations for this assessment.</div>
      ) : (
        <>
          <label className="explanation-target-label">
            <span>Prediction to explain</span>
            <select value={selectedTarget} onChange={(event) => setTarget(event.target.value)}>
              {targetKeys.map((key) => <option value={key} key={key}>{RISK_INFO.find((item) => item.key === key)?.label || key.toUpperCase()}</option>)}
            </select>
          </label>
          {rows.length === 0 ? (
            <div className="explanation-empty">No explanation rows were returned for this target.</div>
          ) : (
            <div className="shap-list">
              {rows.map((row, index) => {
                const contribution = Number(row.contribution) || 0;
                const width = `${(Math.abs(contribution) / maxAbs) * 100}%`;
                const positive = contribution >= 0;
                return (
                  <div className="shap-feature" key={`${row.feature || row.display_name || "feature"}-${index}`}>
                    <div className="shap-feature-heading">
                      <strong>{row.display_name || row.feature || "Feature"}</strong>
                      <span className={positive ? "shap-contribution positive" : "shap-contribution negative"}>
                        {positive ? "+" : ""}{contribution.toFixed(4)}
                      </span>
                    </div>
                    <div className="shap-bar-track">
                      <div className={`shap-bar ${positive ? "positive" : "negative"}`} style={{ width }} />
                    </div>
                    <div className="shap-feature-meta">
                      <span>Patient value: <strong>{row.value === null || row.value === undefined ? "—" : String(row.value)}</strong></span>
                      <span>{positive ? "Pushes output up" : "Pushes output down"}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <p className="shap-caveat">
            SHAP describes how this model responded to these inputs. Contributions are not proof of causation and are not a medical diagnosis.
          </p>
        </>
      )}
    </article>
  );
}

function App() {
  const [featureSchema, setFeatureSchema] = useState([]);
  const [patient, setPatient] = useState({});
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedVessel, setSelectedVessel] = useState("LAD");
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("cardioai-theme") || "light"; }
    catch { return "light"; }
  });

  useEffect(() => {
    try { localStorage.setItem("cardioai-theme", theme); } catch { /* session-only theme */ }
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    async function loadSchema() {
      setSchemaLoading(true);
      try {
        const response = await fetch(`${API_BASE}/schema`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not load the clinical feature schema.");
        const features = Array.isArray(data.features) ? data.features : Array.isArray(data) ? data : [];
        if (!features.length) throw new Error("The backend returned an empty feature schema.");
        if (!cancelled) {
          setFeatureSchema(features);
          setPatient(Object.fromEntries(features.map((item) => [item.feature_name, initialValue(item)])));
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(`${err.message} Make sure the FastAPI server is running at ${API_BASE}.`);
      } finally {
        if (!cancelled) setSchemaLoading(false);
      }
    }
    loadSchema();
    return () => { cancelled = true; };
  }, []);

  const values = useMemo(() => Object.fromEntries(
    RISK_INFO.map((item) => [item.key, getPercent(prediction, item.key)])
  ), [prediction]);

  const groupedFeatures = useMemo(() => {
    const groups = new Map();
    featureSchema.forEach((item) => {
      const category = item.category || "clinical";
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(item);
    });
    return Array.from(groups.entries());
  }, [featureSchema]);

  function updatePatient(item, rawValue) {
    const isNumeric = ["integer", "number", "float", "numeric"].includes(String(item.type || "").toLowerCase());
    setPatient((previous) => ({
      ...previous,
      [item.feature_name]: isNumeric && rawValue !== "" ? Number(rawValue) : rawValue,
    }));
  }

  async function handleAssessment(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      if (!featureSchema.length) throw new Error("The backend feature schema has not loaded yet.");
      const requestBody = {};
      for (const item of featureSchema) {
        const name = item.feature_name;
        let value = patient[name];
        if (value === undefined || value === null || value === "") {
          throw new Error(`Please enter a value for ${readableName(item)}.`);
        }
        const isNumeric = ["integer", "number", "float", "numeric"].includes(String(item.type || "").toLowerCase());
        if (isNumeric) {
          value = Number(value);
          if (!Number.isFinite(value)) throw new Error(`${readableName(item)} must be a valid number.`);
          if (item.minimum !== null && item.minimum !== undefined && value < item.minimum) {
            throw new Error(`${readableName(item)} must be at least ${item.minimum}.`);
          }
          if (item.maximum !== null && item.maximum !== undefined && value > item.maximum) {
            throw new Error(`${readableName(item)} must be no more than ${item.maximum}.`);
          }
        }
        if (Array.isArray(item.allowed_options) && !item.allowed_options.some((option) => String(option) === String(value))) {
          throw new Error(`Choose a valid option for ${readableName(item)}.`);
        }
        requestBody[name] = value;
      }

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const result = await response.json();
      if (!response.ok) {
        const detail = typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail ?? result, null, 2);
        throw new Error(detail || `Backend returned HTTP ${response.status}.`);
      }
      setPrediction(result);
    } catch (err) {
      console.error("Assessment failed:", err);
      setError(err.message || "Could not complete the assessment. Check the backend and input schema.");
    } finally {
      setLoading(false);
    }
  }

  const selectedVesselLabel = selectedVessel === "LAD"
    ? "Left Anterior Descending Artery (LAD)"
    : selectedVessel === "LCX"
      ? "Left Circumflex Artery (LCX)"
      : "Right Coronary Artery (RCA)";

  return (
    <div className={`app theme-${theme}`}>
      <header className="topbar">
        <a className="brand" href="#home" aria-label="CardioAI home">
          <span className="brand-mark">♥</span>
          <span><strong>CardioAI</strong><small>Cardiovascular intelligence</small></span>
        </a>
        <div className="topbar-actions">
          <span className="status-chip"><span className="status-dot" />Risk dashboard</span>
          <button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>
            <span>{theme === "light" ? "☾" : "☀"}</span>{theme === "light" ? "Dark mode" : "Light mode"}
          </button>
        </div>
      </header>

      <main className="dashboard" id="home">
        <section className="welcome">
          <div className="welcome-copy-block">
            <span className="eyebrow">YOUR HEART, AT A GLANCE</span>
            <h1>Health insights,<br /><span>made clearer.</span></h1>
            <p>Explore the heart, review cardiovascular model estimates, and understand the clinical features that influenced each result.</p>
          </div>
          <div className="welcome-art" aria-hidden="true">
            <span className="art-orbit orbit-one" /><span className="art-orbit orbit-two" />
            <span className="art-heart">♥</span><span className="art-bubble bubble-one">CAD</span><span className="art-bubble bubble-two">3D</span>
          </div>
        </section>

        <section className="section-block">
          <div className="section-heading">
            <div><span className="eyebrow">MODEL OUTPUTS</span><h2>Risk overview</h2></div>
            <span className="demo-note">{prediction ? "Backend prediction" : "Waiting for assessment"}</span>
          </div>
          <div className="risk-grid">
            {RISK_INFO.map((item) => {
              const value = values[item.key];
              return <RiskCard key={item.key} item={item} value={value} risk={getRiskLabel(prediction, item.key, value)} />;
            })}
          </div>
        </section>

        <section className="analytics-grid">
          <article className="panel chart-panel">
            <div className="panel-heading"><div><span className="eyebrow">COMPARISON</span><h2>Risk by artery</h2></div><span className="panel-symbol">▥</span></div>
            <p className="panel-description">Compare the model's displayed probability estimate for each target. No demo percentages are shown before a real response.</p>
            <BarChart values={values} />
          </article>
          <article className="panel chart-panel donut-panel">
            <div className="panel-heading"><div><span className="eyebrow">DISTRIBUTION</span><h2>Risk breakdown</h2></div><span className="panel-symbol">◔</span></div>
            <p className="panel-description">Relative share of available displayed estimates, not a validated overall clinical score.</p>
            <DonutChart values={values} />
          </article>
        </section>

        <section className="content-grid">
          <article className="panel heart-panel">
            <div className="panel-heading"><div><span className="eyebrow">INTERACTIVE ANATOMY</span><h2>Explore the heart</h2></div><span className="live-tag">3D MODEL</span></div>
            <p className="panel-description">Drag to rotate the model and scroll to zoom. Select a vessel to inspect its reference image.</p>
            <div className="heart-canvas">
              <Canvas camera={{ position: [0, 0, 3], fov: 40 }} style={{ width: "100%", height: "390px" }}>
                <color attach="background" args={[theme === "light" ? "#F5F2FB" : "#20243A"]} />
                <ambientLight intensity={1.8} />
                <directionalLight position={[5, 5, 5]} intensity={2.5} />
                <directionalLight position={[-5, -2, -5]} intensity={1} />
                <Suspense fallback={null}><HeartModel selectedVessel={selectedVessel} /></Suspense>
                <OrbitControls enableDamping minDistance={1.5} maxDistance={7} />
              </Canvas>
            </div>
            <div className="vessel-selector">
              <span>Explore vessel</span>
              {VESSELS.map((vessel) => (
                <button key={vessel} type="button" className={selectedVessel === vessel ? "vessel-button active" : "vessel-button"} onClick={() => setSelectedVessel(vessel)} aria-pressed={selectedVessel === vessel}>{vessel}</button>
              ))}
            </div>
            <p className="selection-note">Selected vessel: <strong>{selectedVessel}</strong>. Use the controls above to explore the 3D model.</p>
            <div className="vessel-info-panel">
              <h3>{selectedVesselLabel}</h3>
              <img className="vessel-reference-image" src={`/models/${selectedVessel}.png`} alt={`${selectedVessel} coronary artery anatomical reference`} onError={(event) => { event.currentTarget.style.display = "none"; }} />
              <p>{selectedVessel} — {selectedVesselLabel}. The 3D highlight is an anatomical aid; this dataset does not provide direct lesion coordinates.</p>
              {prediction && <p className="selected-vessel-risk">Model estimate: <strong>{values[selectedVessel.toLowerCase()] === null ? "—" : `${values[selectedVessel.toLowerCase()]}%`}</strong></p>}
            </div>
          </article>

          <article className="panel patient-panel">
            <div className="panel-heading"><div><span className="eyebrow">PATIENT PROFILE</span><h2>Clinical assessment inputs</h2></div><span className="panel-symbol">＋</span></div>
            <p className="panel-description">Enter the available demographic, clinical, laboratory, ECG, and echocardiographic features. Fields are loaded directly from the backend's saved feature schema.</p>
            {schemaLoading ? (
              <div className="explanation-empty">Loading clinical input fields from the backend…</div>
            ) : featureSchema.length === 0 ? (
              <div className="error-message" role="alert">No feature schema is available. Check that FastAPI is running and that the source dataset/schema is available.</div>
            ) : (
              <form className="patient-form" onSubmit={handleAssessment}>
                {groupedFeatures.map(([category, items]) => (
                  <fieldset className="clinical-fieldset" key={category}>
                    <legend>{String(category).replaceAll("_", " ")}</legend>
                    <div className="clinical-fields-grid">
                      {items.map((item) => <FeatureInput key={item.feature_name} item={item} value={patient[item.feature_name]} onChange={updatePatient} />)}
                    </div>
                  </fieldset>
                ))}
                {error && <div className="error-message" role="alert">{error}</div>}
                <button className="submit-button" type="submit" disabled={loading || schemaLoading || featureSchema.length === 0}>
                  {loading ? "Analysing…" : "Generate assessment"} {!loading && <span aria-hidden="true">↗</span>}
                </button>
              </form>
            )}
            <div className="privacy-note"><span>ⓘ</span><p>Educational prototype only. Results are model estimates, not a medical diagnosis or a substitute for formal diagnostic imaging or clinician assessment.</p></div>
          </article>
        </section>

        <section className="section-block">
          <ExplanationPanel prediction={prediction} />
        </section>

        <footer className="footer">
          <span className="footer-brand">♥ CardioAI</span>
          <span>Visual insights for cardiovascular risk exploration</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
