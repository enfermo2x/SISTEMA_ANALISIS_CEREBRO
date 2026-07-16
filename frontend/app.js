const API_URL = "http://localhost:5000/predict";

const CLASS_INFO = {
    glioma: {
        label: "Glioma",
        desc: "Tumor en células gliales del cerebro o médula espinal",
        color: "tumor",
        icon: "tumor"
    },
    meningioma: {
        label: "Meningioma",
        desc: "Tumor en las meninges (membranas que recubren el cerebro)",
        color: "tumor",
        icon: "tumor"
    },
    pituitary: {
        label: "Pituitary",
        desc: "Tumor en la glándula pituitaria (hipófisis)",
        color: "tumor",
        icon: "tumor"
    },
    notumor: {
        label: "No Tumor",
        desc: "MRI sin evidencia de tumor",
        color: "notumor",
        icon: "notumor"
    }
};

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

let selectedFile = null;

function resetUI() {
    resultSection.style.display = "none";
    errorSection.style.display = "none";
    loading.style.display = "none";
    btnAnalyze.disabled = true;
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

$("btnChange").addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
        showError("Formato no soportado. Usa JPG o PNG.");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError("La imagen excede 10 MB. Elige un archivo más pequeño.");
        return;
    }

    selectedFile = file;
    resetUI();

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZoneContent.style.display = "none";
        previewContainer.style.display = "flex";
        btnAnalyze.disabled = false;
        dropZone.classList.add("has-image");
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
        try {
            response = await fetch(API_URL, {
                method: "POST",
                body: formData
            });
        } catch (networkError) {
            loading.style.display = "none";
            showError("No se pudo conectar al servidor. Verifica que la API esté corriendo en localhost:5000.");
            btnAnalyze.disabled = false;
            return;
        }

        loading.style.display = "none";

        if (!response.ok) {
            let errorMsg = `Error del servidor (${response.status})`;
            try {
                const errData = await response.json();
                if (errData.error) errorMsg = errData.error;
            } catch (parseError) {}
            showError(errorMsg);
            btnAnalyze.disabled = false;
            return;
        }

        let data;
        try {
            data = await response.json();
        } catch (parseError) {
            showError("Respuesta malformada del servidor.");
            btnAnalyze.disabled = false;
            return;
        }

        if (!data.tipo_tumor || data.probabilidad === undefined) {
            showError("Respuesta incompleta del servidor.");
            btnAnalyze.disabled = false;
            return;
        }

        showResult(data);

    } catch (err) {
        loading.style.display = "none";
        showError("Error inesperado al analizar la imagen.");
        btnAnalyze.disabled = false;
    }
}

function showResult(data) {
    const tipo = data.tipo_tumor.toLowerCase();
    const prob = data.probabilidad;
    const info = CLASS_INFO[tipo] || CLASS_INFO.notumor;
    const isTumor = info.color === "tumor";

    const resultCard = $("resultCard");
    const resultBadge = $("resultBadge");
    const resultConfidence = $("resultConfidence");
    const resultIcon = $("resultIcon");
    const resultTitle = $("resultTitle");
    const resultDesc = $("resultDesc");
    const resultBar = $("resultBar");
    const probList = $("probList");

    resultCard.className = "result-card";
    resultBadge.textContent = isTumor ? "Tumor Detectado" : "Sin Tumor";
    resultBadge.className = `result-badge ${info.color}`;
    resultConfidence.textContent = `${(prob * 100).toFixed(1)}%`;
    resultConfidence.className = `result-confidence ${info.color}`;
    resultIcon.className = `result-icon ${info.icon}`;
    resultTitle.textContent = info.label;
    resultDesc.textContent = info.desc;
    resultBar.className = `result-bar ${info.color}`;

    setTimeout(() => {
        resultBar.style.width = `${(prob * 100).toFixed(1)}%`;
    }, 50);

    if (data.probabilidades) {
        probList.innerHTML = "";
        const sortedClasses = Object.entries(data.probabilidades)
            .sort(([, a], [, b]) => b - a);

        sortedClasses.forEach(([cls, p]) => {
            const pct = (p * 100).toFixed(1);
            const item = document.createElement("div");
            item.className = "prob-item";
            item.innerHTML = `
                <span class="prob-label">${CLASS_INFO[cls]?.label || cls}</span>
                <div class="prob-track">
                    <div class="prob-fill ${cls}" style="width:${pct}%"></div>
                </div>
                <span class="prob-value">${pct}%</span>
            `;
            probList.appendChild(item);
        });
    }

    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
    btnAnalyze.disabled = false;
}

function showError(message) {
    $("errorMessage").textContent = message;
    errorSection.style.display = "block";
    errorSection.scrollIntoView({ behavior: "smooth", block: "center" });

    setTimeout(() => {
        errorSection.style.display = "none";
    }, 6000);
}
