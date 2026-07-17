const API_URL = "http://localhost:5000/predict";

const CLASS_INFO = {
    glioma: { label: "Glioma", desc: "Tumor en células gliales del cerebro", color: "tumor", icon: "tumor" },
    meningioma: { label: "Meningioma", desc: "Tumor en las meninges del cerebro", color: "tumor", icon: "tumor" },
    pituitary: { label: "Pituitary", desc: "Tumor en la glándula pituitaria", color: "tumor", icon: "tumor" },
    notumor: { label: "No Tumor", desc: "MRI sin evidencia de tumor", color: "notumor", icon: "notumor" }
};

const COLOR_MAP = { glioma: "#8b5cf6", meningioma: "#3b82f6", pituitary: "#f97316", notumor: "#22c55e" };

const $ = (id) => document.getElementById(id);

const dropZone = $("dropZone");
const dropZoneContent = $("dropZoneContent");
const previewContainer = $("previewContainer");
const previewImage = $("previewImage");
const fileInput = $("fileInput");
const btnAnalyze = $("btnAnalyze");
const loading = $("loading");
const resultSection = $("resultSection");
const errorSection = $("errorSection");
const radarCanvas = $("radarChart");

let selectedFile = null;

function resetUI() {
    resultSection.style.display = "none";
    errorSection.style.display = "none";
    loading.style.display = "none";
    btnAnalyze.disabled = true;
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => { if (e.target.files.length > 0) handleFile(e.target.files[0]); });
$("btnChange").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) { showError("Formato no soportado. Usa JPG o PNG."); return; }
    if (file.size > 10 * 1024 * 1024) { showError("La imagen excede 10 MB."); return; }
    selectedFile = file;
    resetUI();
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZoneContent.style.display = "none";
        previewContainer.style.display = "flex";
        btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
}

btnAnalyze.addEventListener("click", analyzeImage);

async function analyzeImage() {
    if (!selectedFile) return;
    resetUI();
    loading.style.display = "flex";
    btnAnalyze.disabled = true;
    const formData = new FormData();
    formData.append("image", selectedFile);
    try {
        let response;
        try { response = await fetch(API_URL, { method: "POST", body: formData }); }
        catch (e) { loading.style.display = "none"; showError("No se pudo conectar a la API en localhost:5000."); btnAnalyze.disabled = false; return; }
        loading.style.display = "none";
        if (!response.ok) {
            let msg = "Error del servidor (" + response.status + ")";
            try { const d = await response.json(); if (d.error) msg = d.error; } catch(e) {}
            showError(msg); btnAnalyze.disabled = false; return;
        }
        let data = await response.json();
        if (!data.tipo_tumor || data.probabilidad === undefined) { showError("Respuesta incompleta."); btnAnalyze.disabled = false; return; }
        showResult(data);
    } catch (err) { loading.style.display = "none"; showError("Error inesperado."); btnAnalyze.disabled = false; }
}

function showResult(data) {
    const tipo = data.tipo_tumor.toLowerCase();
    const prob = data.probabilidad;
    const info = CLASS_INFO[tipo] || CLASS_INFO.notumor;
    const isTumor = info.color === "tumor";

    $("resultBadge").textContent = isTumor ? "Tumor Detectado" : "Sin Tumor";
    $("resultBadge").className = "result-badge " + info.color;
    $("resultConfidence").textContent = (prob * 100).toFixed(1) + "%";
    $("resultConfidence").className = "result-confidence " + info.color;
    $("resultIcon").className = "result-icon " + info.icon;
    $("resultTitle").textContent = info.label;
    $("resultDesc").textContent = info.desc;
    const resultBar = $("resultBar");
    resultBar.className = "result-bar " + info.color;
    setTimeout(function() { resultBar.style.width = (prob * 100).toFixed(1) + "%"; }, 50);

    const probList = $("probList");
    if (data.probabilidades) {
        probList.innerHTML = "";
        var sorted = Object.entries(data.probabilidades).sort(function(a, b) { return b[1] - a[1]; });
        sorted.forEach(function(item) {
            var cls = item[0], p = item[1];
            var pct = (p * 100).toFixed(1);
            var el = document.createElement("div");
            el.className = "prob-item";
            el.innerHTML = '<span class="prob-label">' + (CLASS_INFO[cls] ? CLASS_INFO[cls].label : cls) + '</span>' +
                '<div class="prob-track"><div class="prob-fill ' + cls + '" style="width:0%"></div></div>' +
                '<span class="prob-value">' + pct + '%</span>';
            probList.appendChild(el);
            setTimeout(function() { el.querySelector(".prob-fill").style.width = pct + "%"; }, 100);
        });
    }

    drawRadarChart(data.probabilidades);
    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
    btnAnalyze.disabled = false;
}

function drawRadarChart(probs) {
    if (!radarCanvas || !probs) return;
    var ctx = radarCanvas.getContext("2d");
    var w = radarCanvas.width, h = radarCanvas.height;
    var cx = w / 2, cy = h / 2, r = Math.min(cx, cy) - 20;
    ctx.clearRect(0, 0, w, h);

    var labels = Object.keys(probs);
    var values = labels.map(function(l) { return probs[l]; });
    var n = labels.length;
    var angleStep = (2 * Math.PI) / n;

    ctx.save();
    ctx.translate(cx, cy);

    for (var ring = 1; ring <= 4; ring++) {
        var rr = (r / 4) * ring;
        ctx.beginPath();
        for (var i = 0; i <= n; i++) {
            var idx = i % n;
            var angle = idx * angleStep - Math.PI / 2;
            var x = Math.cos(angle) * rr;
            var y = Math.sin(angle) * rr;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    for (var i = 0; i < n; i++) {
        var angle = i * angleStep - Math.PI / 2;
        var x = Math.cos(angle) * r;
        var y = Math.sin(angle) * r;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(x, y);
        ctx.strokeStyle = "#e2e8f0";
        ctx.stroke();
    }

    ctx.beginPath();
    for (var i = 0; i <= n; i++) {
        var idx = i % n;
        var angle = idx * angleStep - Math.PI / 2;
        var val = Math.max(values[idx], 0.01);
        var dist = r * Math.min(val, 1);
        var x = Math.cos(angle) * dist;
        var y = Math.sin(angle) * dist;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(14, 165, 233, 0.15)";
    ctx.fill();
    ctx.strokeStyle = "#0ea5e9";
    ctx.lineWidth = 2;
    ctx.stroke();

    for (var i = 0; i < n; i++) {
        var angle = i * angleStep - Math.PI / 2;
        var val = Math.max(values[i], 0.01);
        var dist = r * Math.min(val, 1);
        var x = Math.cos(angle) * dist;
        var y = Math.sin(angle) * dist;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = COLOR_MAP[labels[i]] || "#0ea5e9";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    ctx.restore();
    ctx.save();
    for (var i = 0; i < n; i++) {
        var angle = i * angleStep - Math.PI / 2;
        var labelR = r + 14;
        var x = Math.cos(angle) * labelR + cx;
        var y = Math.sin(angle) * labelR + cy;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "11px -apple-system, sans-serif";
        ctx.fillStyle = "#64748b";
        var displayLabel = CLASS_INFO[labels[i]] ? CLASS_INFO[labels[i]].label : labels[i];
        ctx.fillText(displayLabel, x, y);
    }
    ctx.restore();
}

function showError(message) {
    $("errorMessage").textContent = message;
    errorSection.style.display = "block";
    errorSection.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(function() { errorSection.style.display = "none"; }, 6000);
}
