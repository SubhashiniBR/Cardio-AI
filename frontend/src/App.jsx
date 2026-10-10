
import { useState, Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Center, useGLTF } from "@react-three/drei";
import "./App.css";

const DEMO_RISKS = {
  CAD: 68,
  LAD: 72,
  LCX: 41,
  RCA: 29,
};

function HeartModel() {
  const { scene } = useGLTF("/models/heart.glb");

  const model = useMemo(() => {
    const copy = scene.clone(true);

    copy.traverse((child) => {
      if (child.isMesh) {
        child.material = child.material.clone();
        if (child.material.color) {
          child.material.color.set("#c98288");
        }
        child.material.roughness = 0.7;
      }
    });

    return copy;
  }, [scene]);

  return (
    <Center>
      <primitive object={model} scale={1.5} />
    </Center>
  );
}

function RiskCard({ title, value, description }) {
  const level = value >= 65 ? "High" : value >= 35 ? "Moderate" : "Lower";

  return (
    <article className="risk-card">
      <div className="risk-card-top">
        <span>{title}</span>
        <span className={`risk-dot ${level.toLowerCase()}`} />
      </div>

      <div className="risk-value">
        {value}<span>%</span>
      </div>

      <div className="risk-track">
        <div
          className={`risk-fill ${level.toLowerCase()}`}
          style={{ width: `${value}%` }}
        />
      </div>

      <div className="risk-card-bottom">
        <span>{description}</span>
        <span className={`risk-label ${level.toLowerCase()}`}>
          {level} demo
        </span>
      </div>
    </article>
  );
}

function App() {
  const [patient, setPatient] = useState({
    age: 55,
    sex: "Male",
    systolicBP: 130,
    cholesterol: 210,
  });

  const [selectedVessel, setSelectedVessel] = useState("LAD");

  function updatePatient(event) {
    const { name, value } = event.target;

    setPatient((previous) => ({
      ...previous,
      [name]: name === "sex" ? value : Number(value),
    }));
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">C</div>
          <div>
            <h1>CardioAI</h1>
            <p>Cardiovascular Risk Intelligence</p>
          </div>
        </div>

        <span className="demo-status">
          <span className="status-dot" />
          Demo mode
        </span>
      </header>

      <main className="dashboard">
        <section className="welcome">
          <div>
            <p className="eyebrow">CARDIOVASCULAR OVERVIEW</p>
            <h2>Understand the heart.<br />Explore the risk.</h2>
            <p className="welcome-copy">
              Explore a 3D heart model and a sample cardiovascular
              risk dashboard.
            </p>
          </div>

          <div className="overview-badge">
            <span>DEMO PATIENT</span>
            <strong>Patient #001</strong>
            <small>Illustrative values only</small>
          </div>
        </section>

        <section className="risk-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">RISK SUMMARY</p>
              <h3>Prediction overview</h3>
            </div>
            <span className="demo-note">Sample values · Not a diagnosis</span>
          </div>

          <div className="risk-grid">
            <RiskCard
              title="Overall CAD"
              value={DEMO_RISKS.CAD}
              description="Overall risk score"
            />
            <RiskCard
              title="LAD"
              value={DEMO_RISKS.LAD}
              description="Left anterior descending"
            />
            <RiskCard
              title="LCX"
              value={DEMO_RISKS.LCX}
              description="Left circumflex"
            />
            <RiskCard
              title="RCA"
              value={DEMO_RISKS.RCA}
              description="Right coronary artery"
            />
          </div>
        </section>

        <section className="main-grid">
          <article className="panel heart-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">INTERACTIVE ANATOMY</p>
                <h3>3D heart explorer</h3>
              </div>
              <span className="live-tag">3D MODEL</span>
            </div>

            <p className="panel-description">
              Drag to rotate and scroll to zoom.
            </p>

            <div className="heart-canvas">
              <Canvas
                camera={{ position: [0, 0, 3], fov: 40 }}
                style={{ width: "100%", height: "430px" }}
              >
                <color attach="background" args={["#101827"]} />
                <ambientLight intensity={1.8} />
                <directionalLight
                  position={[5, 5, 5]}
                  intensity={2.5}
                />
                <directionalLight
                  position={[-5, -2, -5]}
                  intensity={1}
                />

                <Suspense fallback={null}>
                  <HeartModel />
                </Suspense>

                <OrbitControls
                  enableDamping
                  minDistance={1.5}
                  maxDistance={7}
                />
              </Canvas>
            </div>

            <div className="vessel-selector">
              <span>Explore vessel</span>
              {["LAD", "LCX", "RCA"].map((vessel) => (
                <button
                  key={vessel}
                  className={
                    selectedVessel === vessel ? "vessel-button active" : "vessel-button"
                  }
                  onClick={() => setSelectedVessel(vessel)}
                >
                  {vessel}
                </button>
              ))}
            </div>

            <p className="selection-note">
              Selected: <strong>{selectedVessel}</strong>. Vessel selection
              is a dashboard interaction only; this model does not yet
              highlight individual coronary arteries.
            </p>
          </article>

          <div className="side-column">
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">PATIENT PROFILE</p>
                  <h3>Sample measurements</h3>
                </div>
              </div>

              <p className="panel-description">
                Edit these demo fields to explore the form.
              </p>

              <div className="patient-form">
                <label>
                  Age
                  <input
                    name="age"
                    type="number"
                    min="18"
                    max="110"
                    value={patient.age}
                    onChange={updatePatient}
                  />
                </label>

                <label>
                  Sex
                  <select
                    name="sex"
                    value={patient.sex}
                    onChange={updatePatient}
                  >
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </label>

                <label>
                  Systolic blood pressure (mmHg)
                  <input
                    name="systolicBP"
                    type="number"
                    value={patient.systolicBP}
                    onChange={updatePatient}
                  />
                </label>

                <label>
                  Total cholesterol (mg/dL)
                  <input
                    name="cholesterol"
                    type="number"
                    value={patient.cholesterol}
                    onChange={updatePatient}
                  />
                </label>

                <button
                  className="primary-button"
                  onClick={() =>
                    window.alert(
                      "These are demo fields only. Real predictions will be connected to the backend next."
                    )
                  }
                >
                  Run demo assessment
                </button>
              </div>
            </article>

            <article className="panel explanation-panel">
              <p className="eyebrow">MODEL INTERPRETATION</p>
              <h3>Why this prediction?</h3>
              <p className="panel-description">
                The explanation panel will show feature contributions from
                the trained model once the backend is connected.
              </p>

              <div className="explanation-placeholder">
                <span className="explanation-icon">✳</span>
                <div>
                  <strong>SHAP explanations</strong>
                  <p>Waiting for model output</p>
                </div>
              </div>
            </article>
          </div>
        </section>

        <footer className="medical-disclaimer">
          <strong>Educational prototype only.</strong> Demo values are
          illustrative and are not real patient predictions. This tool is
          not a substitute for professional medical advice or formal
          diagnostic imaging.
        </footer>
      </main>
    </div>
  );
}

export default App;
