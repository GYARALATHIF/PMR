const form = document.getElementById("predict-form");
const button = document.getElementById("predict-btn");
const mortalityBar = document.getElementById("mortality-bar");
const readmissionBar = document.getElementById("readmission-bar");
const mortalityText = document.getElementById("mortality-text");
const readmissionText = document.getElementById("readmission-text");
const embeddingMode = document.getElementById("embedding-mode");
const historyList = document.getElementById("history-list");
const liveDot = document.getElementById("live-dot");
const liveText = document.getElementById("live-text");

function toPayload(formData) {
  return {
    los_days: Number(formData.get("los_days")),
    num_diagnoses: Number(formData.get("num_diagnoses")),
    num_procedures: Number(formData.get("num_procedures")),
    has_sepsis: formData.get("has_sepsis") === "on",
    has_diabetes: formData.get("has_diabetes") === "on",
    has_vent: formData.get("has_vent") === "on",
    insurance: formData.get("insurance"),
    discharge_group: formData.get("discharge_group"),
    admission_type: formData.get("admission_type"),
    admission_location: formData.get("admission_location"),
    ethnicity: formData.get("ethnicity"),
    clinical_note: formData.get("clinical_note").trim(),
  };
}

function pct(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function paintResult(data) {
  mortalityBar.style.width = pct(data.mortality_probability);
  readmissionBar.style.width = pct(data.readmission_probability);

  mortalityText.textContent = `${pct(data.mortality_probability)} (${data.mortality_risk_tier} risk)`;
  readmissionText.textContent = `${pct(data.readmission_probability)} (${data.readmission_risk_tier} risk)`;
  embeddingMode.textContent = `Embedding mode: ${data.embedding_mode}`;
}

async function refreshHistory() {
  const res = await fetch("/api/history?limit=10");
  const data = await res.json();
  historyList.innerHTML = "";
  if (!data.items || data.items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No predictions yet";
    historyList.appendChild(li);
    return;
  }

  for (const item of data.items) {
    const li = document.createElement("li");
    li.textContent = `${new Date(item.created_at).toLocaleTimeString()} | Mortality ${pct(item.mortality_probability)} | Readmission ${pct(item.readmission_probability)}`;
    historyList.appendChild(li);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  button.disabled = true;
  button.textContent = "Predicting...";

  try {
    const payload = toPayload(new FormData(form));
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || "Prediction failed");
    }

    const data = await res.json();
    paintResult(data);
    await refreshHistory();
  } catch (err) {
    alert(err.message || "Prediction failed");
  } finally {
    button.disabled = false;
    button.textContent = "Predict Risk";
  }
});

function connectLive() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);

  ws.onopen = () => {
    liveDot.classList.add("live");
    liveText.textContent = "Live status connected";
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    liveText.textContent = `Live OK | ${new Date(data.timestamp).toLocaleTimeString()} | Predictions: ${data.history_size}`;
    embeddingMode.textContent = `Embedding mode: ${data.embedding_mode}`;
  };

  ws.onclose = () => {
    liveDot.classList.remove("live");
    liveText.textContent = "Live status disconnected, reconnecting...";
    setTimeout(connectLive, 1500);
  };
}

connectLive();
refreshHistory();
