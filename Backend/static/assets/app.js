/**
 * ============================================================================
 * CONTROLADOR PRINCIPAL DEL FRONTEND (DETECCIÓN DE ANOMALÍAS)
 * ============================================================================
 * Este script gestiona la interacción del usuario, la subida de imágenes,
 * la comunicación con el API (backend) y el renderizado de resultados.
 */

// ============================================================================
// 1. REFERENCIAS AL DOM (ELEMENTOS HTML)
// ============================================================================

// --- Entradas de archivos ---
const filesInput = document.getElementById('files');      // Input tipo file (multiple)
const folderInput = document.getElementById('folder');    // Input para subir carpetas completas
const dropzone = document.getElementById('dropzone');     // Área de "arrastrar y soltar"
const fileList = document.getElementById('fileList');     // Contenedor visual de la lista de archivos

// --- Botones de acción principales ---
const analyzeBtn = document.getElementById('analyzeBtn'); // Botón para iniciar inferencia batch
const clearBtn = document.getElementById('clearBtn');     // Botón para limpiar todo

// --- Configuración de parámetros ---
const thrAuto = document.getElementById('thrAuto');       // Radio button: Umbral automático
const thrManual = document.getElementById('thrManual');   // Radio button: Umbral manual
const thrValue = document.getElementById('thrValue');     // Input numérico para valor manual
const modeSel = document.getElementById('mode');          // Selector de modo (ej. sensibilidad)

// --- Cajas de información ---
const healthBox = document.getElementById('health');      // Muestra config del backend al inicio
const jsonBox = document.getElementById('jsonBox');       // Muestra el JSON crudo (debug)

// --- KPIs (Indicadores Clave de Rendimiento) ---
const kpiTotal = document.getElementById('kpiTotal');     // Total imágenes procesadas
const kpiNorm  = document.getElementById('kpiNorm');      // Cantidad Normales
const kpiAnom  = document.getElementById('kpiAnom');      // Cantidad Anomalías
const kpiRate  = document.getElementById('kpiRate');      // % de defecto
const kpiArea  = document.getElementById('kpiArea');      // Área promedio de defectos
const kpiRecall = document.getElementById('kpiRecall');


// --- Vistas y Tablas ---
const gridView = document.getElementById('gridView');     // Contenedor para vista de cuadrícula (imágenes)
const tableView = document.getElementById('tableView');   // Contenedor para vista de tabla
const resultsTable = document.getElementById('resultsTable'); // Tbody de la tabla

// --- Controles de Vistas ---
const tabGrid = document.getElementById('tabGrid');
const tabTable = document.getElementById('tabTable');
const toggleViewBtn = document.getElementById('toggleViewBtn');
const downloadCsvBtn = document.getElementById('downloadCsvBtn'); // Botón descargar reporte

// --- Elementos del Módulo de Cámara ---
const cameraToggleBtn = document.getElementById('cameraToggleBtn');
const cameraPanel = document.getElementById('cameraPanel');       // Contenedor del video
const cameraVideo = document.getElementById('cameraVideo');       // Elemento <video>
const cameraOverlayImg = document.getElementById('cameraOverlay');// Imagen <img> superpuesta para mostrar defectos
const camScore = document.getElementById('camScore');             // Puntuación en tiempo real
const camAreaTotal = document.getElementById('camAreaTotal');
const camState = document.getElementById('camState');             // Texto NORMAL/ANOMALÍA
const camIoU = document.getElementById('camIoU');                 // Intersection over Union (si aplica)


// ============================================================================
// 2. ESTADO DE LA APLICACIÓN
// ============================================================================

/** Almacena los objetos File seleccionados por el usuario antes de enviarlos. */
let selectedFiles = []; 

/** Almacena la última respuesta completa del servidor para poder cambiar vistas o descargar CSV sin re-procesar. */
let lastBatch = null;   

// --- Estado interno de la cámara ---
let cameraStream = null;      // Objeto MediaStream (el flujo de video)
let cameraIntervalId = null;  // ID del setInterval para detenerlo después
let cameraBusy = false;       // "Semáforo": true si se está procesando una foto, evita colapsar el servidor
const CAMERA_INTERVAL_MS = 1500; // Frecuencia de captura (1.5 segundos)


// ============================================================================
// 3. FUNCIONES AUXILIARES (HELPERS)
// ============================================================================

/**
 * Verifica si un archivo es una imagen basándose en su tipo MIME.
 * @param {File} f - Archivo a verificar.
 */
function isImage(f){
  return f && f.type && f.type.startsWith('image/');
}

/**
 * Agrega archivos a la lista de seleccionados, filtrando no-imágenes y duplicados.
 * @param {FileList} fileListLike - Lista de archivos del input o drop event.
 */
function addFiles(fileListLike){
  // 1. Filtrar solo imágenes
  const arr = Array.from(fileListLike).filter(isImage);
  selectedFiles.push(...arr);

  // 2. Eliminar duplicados usando un Map
  // Se usa 'nombre_tamaño' como clave única heurística.
  const map = new Map();
  selectedFiles.forEach(f=>{
    const key = `${f.name}_${f.size}`;
    if(!map.has(key)) map.set(key, f);
  });
  
  // 3. Actualizar estado y UI
  selectedFiles = Array.from(map.values());
  renderFileList();
  updateButtons();
}

/**
 * Renderiza la lista visual de archivos pendientes de análisis.
 */
function renderFileList(){
  if(selectedFiles.length === 0){
    fileList.innerHTML = '<div class="muted">No hay archivos seleccionados</div>';
    return;
  }
  // Muestra nombre y tamaño en KB
  fileList.innerHTML = selectedFiles.map(f => `
    <div class="item">
        <span>${f.webkitRelativePath || f.name}</span>
        <span class="muted">${(f.size/1024).toFixed(0)} KB</span>
    </div>
  `).join('');
}

/**
 * Habilita o deshabilita botones según el estado actual.
 */
function updateButtons(){
  const has = selectedFiles.length > 0;
  analyzeBtn.disabled = !has;         // Solo activar si hay archivos
  toggleViewBtn.disabled = !lastBatch;// Solo activar vistas si hay resultados
  downloadCsvBtn.disabled = !lastBatch;
}

/**
 * Actualiza los números del dashboard (KPIs) con el resumen del backend.
 */
function setKPIs(summary){
  const total = summary?.total_images || 0;
  const anom = summary?.anomalies || 0;
  const norm = summary?.normals || 0;

  const rate =
    summary?.defect_rate != null
      ? (summary.defect_rate * 100).toFixed(1)
      : "0.0";

  const area =
    summary?.avg_defect_area_px != null
      ? Math.round(summary.avg_defect_area_px)
      : 0;

  const recall = summary?.recall != null
  ? (summary.recall * 100).toFixed(1) + "%"
  : "—";

  kpiTotal.textContent = total;
  kpiAnom.textContent  = anom;
  kpiNorm.textContent  = norm;
  kpiRate.textContent  = `${rate}%`;
  kpiArea.textContent  = area;

  kpiRecall.textContent = recall;
}

/**
 * Resetea toda la interfaz y variables de estado (botón Limpiar).
 */
function clearResults(){
  lastBatch = null;
  gridView.innerHTML = '';
  resultsTable.innerHTML = '';
  jsonBox.textContent = '';
  setKPIs({});
  updateButtons();
}

/**
 * Genera el HTML para una celda de estado (ANOMALÍA/NORMAL) con colores.
 */
function formatStateCell(is_anomaly){
  return is_anomaly
    ? `<span class="state bad">ANOMALÍA</span>`
    : `<span class="state ok">NORMAL</span>`;
}

/**
 * Calcula la suma total de un array de áreas.
 */
function totalArea(areas){
  if(!areas || !areas.length) return 0;
  return Math.round(areas.reduce((a,b)=>a+b,0));
}

/**
 * Crea una URL temporal para previsualizar la imagen local antes de subirla.
 */
function fileThumbURL(file){
  return URL.createObjectURL(file);
}

/**
 * Convierte el objeto de datos JSON a formato CSV para descarga.
 * Maneja el escape de comillas dobles para evitar romper el formato.
 */
function asCsv(data){
  const rows = [
    // Cabecera del CSV
    ["filename","score","threshold","is_anomaly","num_polygons","area_total_px","area_max_px","iou","overlay_url"]
  ];
  
  // Procesar cada fila
  data.results.forEach(r=>{
    const polyAreas = r.polygon_areas_px || [];
    // Prioridad: dato del servidor > cálculo local
    const areaTotal = (r.total_defect_area_px != null) 
      ? Math.round(r.total_defect_area_px) 
      : totalArea(polyAreas);
      
    const areaMax = (r.max_defect_area_px != null)
      ? Math.round(r.max_defect_area_px)
      : (polyAreas.length ? Math.round(Math.max(...polyAreas)) : 0);
      
    const iouStr = (r.iou != null) ? r.iou : '';
    
    rows.push([
      r.filename,
      r.score,
      r.threshold,
      r.is_anomaly ? 1 : 0,
      (r.polygons || []).length,
      areaTotal,
      areaMax,
      iouStr,
      r.overlay_url || ""
    ]);
  });

  // Unir columnas con comas y filas con saltos de línea
  return rows.map(r=>r.map(v => {
    const s = String(v ?? '');
    // Si contiene comas o comillas, envolver en comillas y escapar las internas
    return /[",;\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
  }).join(',' )).join('\n');
}


// ============================================================================
// 4. EVENT LISTENERS (INTERACCIÓN UI)
// ============================================================================

// --- Inputs de archivos ---
filesInput.addEventListener('change', e=>{
  if(e.target.files) addFiles(e.target.files);
});
folderInput.addEventListener('change', e=>{
  if(e.target.files) addFiles(e.target.files);
});

// --- Botón Limpiar ---
clearBtn.addEventListener('click', ()=>{
  selectedFiles = [];
  filesInput.value = '';
  folderInput.value = '';
  renderFileList();
  clearResults();
  updateButtons(); // Deshabilita el botón analizar
});

// --- Drag & Drop (Arrastrar y soltar) ---
dropzone.addEventListener('dragover', e=>{
  e.preventDefault(); // Necesario para permitir el drop
  dropzone.classList.add('drag'); // Efecto visual
});
dropzone.addEventListener('dragleave', ()=>{
  dropzone.classList.remove('drag');
});
dropzone.addEventListener('drop', e=>{
  e.preventDefault(); // Evita que el navegador abra la imagen
  dropzone.classList.remove('drag');
  if(e.dataTransfer.files) addFiles(e.dataTransfer.files);
});

// --- Configuración de Umbral ---
// Si se marca Manual, se habilita el input numérico.
thrManual.addEventListener('change', ()=> thrValue.disabled = !thrManual.checked);
thrAuto.addEventListener('change', ()=> thrValue.disabled = !thrManual.checked);

// --- Gestión de Pestañas (Grid vs Table) ---
function setView(mode){ // mode: 'grid' | 'table'
  if(mode === 'grid'){
    tabGrid.classList.add('active');
    tabTable.classList.remove('active');
    gridView.classList.remove('hidden');
    tableView.classList.add('hidden');
  }else{
    tabTable.classList.add('active');
    tabGrid.classList.remove('active');
    tableView.classList.remove('hidden');
    gridView.classList.add('hidden');
  }
}
tabGrid.addEventListener('click', ()=> setView('grid'));
tabTable.addEventListener('click', ()=> setView('table'));
toggleViewBtn.addEventListener('click', ()=>{
  // Alternar basado en el estado actual
  const isGrid = !gridView.classList.contains('hidden');
  setView(isGrid ? 'table' : 'grid');
});

// --- Descarga de CSV ---
downloadCsvBtn.addEventListener('click', ()=>{
  if(!lastBatch) return;
  const csv = asCsv(lastBatch);
  
  // Crear un Blob y un link temporal para forzar la descarga
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'resultados_patchcore.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url); // Liberar memoria
});


// ============================================================================
// 5. LÓGICA DE ANÁLISIS BATCH (Botón "Analizar")
// ============================================================================

// Flag para saber si estamos en fase de GT/recall
let awaitingGT = false;
// Guardamos la última respuesta para poder leer idx y filenames
let lastResultsForGT = null;

analyzeBtn.addEventListener("click", async () => {
  // FASE 1: no tenemos aún GT → solo analizamos y mostramos resultados
  if (!awaitingGT) {
    if (selectedFiles.length === 0) {
      alert("Selecciona imágenes o carpeta.");
      return;
    }

    clearResults();              // limpiamos UI anterior
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Procesando...";

    const fd = new FormData();
    selectedFiles.forEach(f => fd.append("files", f));
    if (thrManual.checked && thrValue.value) fd.append("thr", thrValue.value);
    if (modeSel.value) fd.append("mode", modeSel.value);

    try {
      const r = await fetch("/predict_batch", {
        method: "POST",
        body: fd
      });
      if (!r.ok) throw new Error("Error servidor");

      const data = await r.json();

      // Guardamos resultados sin GT
      lastBatch = data;
      lastResultsForGT = data;

      renderResults(data, selectedFiles);
      jsonBox.textContent = JSON.stringify(data, null, 2);

      // Ahora sí: pasamos a FASE 2 (esperar GT)
      awaitingGT = true;
      analyzeBtn.textContent = "Calcular recall";
      analyzeBtn.disabled = false;

      // Aún no activamos CSV ni toggle, porque las métricas "finales" serán tras recall
      updateButtons();

    } catch (err) {
      jsonBox.textContent = "⚠️ ERROR: " + err.message;
      console.error(err);
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analizar";
    }
    return;
  }

  // FASE 2: ya tenemos radios GT en el DOM → calculamos recall
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Calculando recall...";

  try {
    if (!lastResultsForGT || !lastResultsForGT.results) {
      throw new Error("No hay resultados previos para leer GT.");
    }

    // 1️⃣ Leer GT desde los radios actuales
    const gtLabels = [];
    lastResultsForGT.results.forEach(r => {
      const radios = document.querySelectorAll(`input[name="gt_${r.idx}"]`);
      let val = 0;
      radios.forEach(rb => { if (rb.checked) val = Number(rb.value); });
      gtLabels.push(val);
    });
    console.log("GT ENVIADAS:", gtLabels);

    // 2️⃣ Preparar nuevo FormData con archivos + GT
    const fd2 = new FormData();
    selectedFiles.forEach(f => fd2.append("files", f));
    if (thrManual.checked && thrValue.value) fd2.append("thr", thrValue.value);
    if (modeSel.value) fd2.append("mode", modeSel.value);
    gtLabels.forEach(v => fd2.append("gt_labels", v));

    // 3️⃣ Segunda llamada a /predict_batch
    const r2 = await fetch("/predict_batch", {
      method: "POST",
      body: fd2
    });
    if (!r2.ok) throw new Error("Error servidor en segunda pasada");

    const data2 = await r2.json();

    // 4️⃣ Ahora sí: resultados finales con recall
    lastBatch = data2;                  // importante para CSV y toggle
    renderResults(data2, selectedFiles);
    jsonBox.textContent = JSON.stringify(data2, null, 2);

    awaitingGT = false;                // volvemos a modo normal
    analyzeBtn.textContent = "Analizar";
    analyzeBtn.disabled = false;

    // Ahora que ya tenemos métricas globales (incluyendo recall), activamos CSV/Vista tabla
    updateButtons();

  } catch (err) {
    jsonBox.textContent = "⚠️ ERROR: " + err.message;
    console.error(err);
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analizar";
    awaitingGT = false;
  }
});




/**
 * Renderiza los resultados en el DOM (tanto Grid como Tabla).
 * @param {Object} data - Respuesta JSON del servidor.
 * @param {File[]} files - Archivos originales para generar thumbnails locales.
 */

function normalizeName(name) {
  // Mantener letras, números, guiones, guion bajo, puntos.
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function renderResults(data, files) {

  // --- A. KPIs ---
  if (data.summary) {
    setKPIs(data.summary);
  } else {
    setKPIs({
      total_images: data.results.length,
      anomalies: data.results.filter(r => r.is_anomaly).length,
      normals: data.results.filter(r => !r.is_anomaly).length,
      defect_rate: data.results.filter(r => r.is_anomaly).length / data.results.length,
      avg_defect_area_px:
        data.results.reduce((acc, r) => acc + (r.total_defect_area_px || 0), 0) / data.results.length,
      recall: null
    });
  }

  // --- B. TABLA ---
  resultsTable.innerHTML = data.results.map(r => {
    const polyAreas = r.polygon_areas_px || [];
    const areaTotal = (r.total_defect_area_px ?? totalArea(polyAreas));
    const areaMax = (r.max_defect_area_px ?? (polyAreas.length ? Math.max(...polyAreas) : 0));
    const iouStr = (r.iou != null) ? r.iou.toFixed(3) : "-";

    return `
      <tr>
        <td>${r.filename}</td>
        <td>${r.score.toFixed(6)}</td>
        <td>${Number(r.threshold).toFixed(6)}</td>
        <td>${formatStateCell(r.is_anomaly)}</td>
        <td>${(r.polygons || []).length}</td>
        <td>${Math.round(areaTotal)}</td>
        <td>${Math.round(areaMax)}</td>
        <td>${iouStr}</td>
        <td>${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Ver</a>` : '-'}</td>
      </tr>
    `;
  }).join("");

  // --- C. GRID (Galería) ---
  const fileMap = new Map();
  files.forEach(f => {
    fileMap.set(f.name, f);
    if (f.webkitRelativePath) fileMap.set(f.webkitRelativePath, f);
  });

  gridView.innerHTML = data.results.map(r => {

    //SIEMPRE LA MISMA NORMALIZACIÓN
    const safeName = normalizeName(r.filename);

    let f = fileMap.get(r.filename);

    if (!f) {
      const baseName = r.filename.split(/[\\/]/).pop();
      f = fileMap.get(baseName);
    }


    const thumb = f ? fileThumbURL(f) : "";
    const polyAreas = r.polygon_areas_px || [];
    const polyCount = (r.polygons || []).length;
    const areaTotal = Math.round(r.total_defect_area_px ?? totalArea(polyAreas));
    const areaMax = Math.round(r.max_defect_area_px ?? (polyAreas.length ? Math.max(...polyAreas) : 0));
    const iouStr = (r.iou != null) ? r.iou.toFixed(3) : "-";

    return `
      <article class="card-img">

        ${thumb ? `<img class="thumb" src="${thumb}" alt="${r.filename}"/>` : ''}

        <div class="pad">
          <div class="row">
            <div class="state ${r.is_anomaly ? "bad" : "ok"}">
              ${r.is_anomaly ? "ANOMALÍA" : "NORMAL"}
            </div>
            ${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Overlay</a>` : ""}
          </div>

          <div class="meta">score=<b>${r.score.toFixed(6)}</b> • thr=${Number(r.threshold).toFixed(6)}</div>
          <div class="meta">polígonos=${polyCount} • área total=${areaTotal}px • área máx=${areaMax}px</div>
          <div class="meta">IoU=${iouStr}</div>
          <div class="meta">${r.filename}</div>

         <!-- Ground Truth --> 
          <div class="meta">
            <label class="gtlabel">
              <input type="radio" name="gt_${r.idx}" value="0" checked>
              Normal
            </label>
            <label class="gtlabel">
              <input type="radio" name="gt_${r.idx}" value="1">
              Defecto
            </label>
          </div>
        </div>
      </article>
    `;
  }).join("");

}
 



// ============================================================================
// 6. MÓDULO DE CÁMARA EN TIEMPO REAL
// ============================================================================

/**
 * Inicia el stream de video y el ciclo de predicción.
 */
async function startCamera(){
  if(cameraStream) return; // Ya está iniciada
  
  // Validar soporte del navegador
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    alert('La cámara no está soportada en este navegador.');
    return;
  }
  
  try{
    // Solicitar acceso a la cámara
    const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
    cameraStream = stream;
    cameraVideo.srcObject = stream; // Mostrar video en el elemento HTML
    
    // Actualizar UI
    cameraPanel.classList.remove('hidden');
    cameraToggleBtn.textContent = 'Detener cámara';

    // Iniciar el bucle de captura (polling cada X ms)
    cameraIntervalId = setInterval(captureFrameAndPredict, CAMERA_INTERVAL_MS);
  }catch(err){
    alert('No se pudo acceder a la cámara: ' + err.message);
  }
}

/**
 * Detiene el video, libera recursos y detiene el ciclo de predicción.
 */
function stopCamera(){
  // 1. Detener intervalo
  if(cameraIntervalId){
    clearInterval(cameraIntervalId);
    cameraIntervalId = null;
  }
  // 2. Detener tracks de video (apagar hardware)
  if(cameraStream){
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  // 3. Limpiar UI
  cameraVideo.srcObject = null;
  cameraToggleBtn.textContent = 'Activar cámara';
  cameraPanel.classList.add('hidden');

  // Resetear valores visuales
  camScore.textContent = '0.000';
  camAreaTotal.textContent = '0';
  camState.textContent = '-';
  camState.style.color = '';
  camIoU.textContent = '-';
  cameraOverlayImg.removeAttribute('src');
}

/**
 * Genera un nombre de archivo único basado en la fecha actual.
 * Formato: camera_yyyyMMdd_HHmmssfff.jpg
 */
function makeCameraFilename(){
  const now = new Date();
  const pad = (n, l=2) => n.toString().padStart(l,'0');
  const y = now.getFullYear();
  const M = pad(now.getMonth()+1);
  const d = pad(now.getDate());
  const h = pad(now.getHours());
  const m = pad(now.getMinutes());
  const s = pad(now.getSeconds());
  const ms = pad(now.getMilliseconds(),3);
  return `camera_${y}${M}${d}_${h}${m}${s}${ms}.jpg`;
}

/**
 * Captura un frame del video, lo envía al servidor y muestra el resultado.
 * Se ejecuta periódicamente por setInterval.
 */
async function captureFrameAndPredict(){
  // Validaciones: si no hay stream, si ya estamos ocupados procesando una foto, o si el video no tiene tamaño
  if(!cameraStream || cameraBusy || !cameraVideo.videoWidth) return;
  
  cameraBusy = true; // Bloquear nuevas peticiones (Semáforo)

  try{
    // 1. Dibujar frame actual en un Canvas en memoria
    const canvas = document.createElement('canvas');
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

    // 2. Convertir Canvas a Blob (JPG)
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
    if(!blob) return;

    // 3. Preparar FormData para enviar
    const filename = makeCameraFilename();
    const fd = new FormData();
    fd.append('file', new File([blob], filename, { type:'image/jpeg' }));

    // Configurar params query string
    const params = new URLSearchParams();
    params.append('source', 'camera');
    if(thrManual.checked && thrValue.value) params.append('thr', thrValue.value);
    if(modeSel.value) params.append('mode', modeSel.value);

    // 4. Enviar a endpoint de imagen única (/predict)
    const resp = await fetch(`/predict?${params.toString()}`, {
      method:'POST',
      body: fd
    });
    if(!resp.ok) throw new Error('Error del servidor en cámara');
    const data = await resp.json();

    jsonBox.textContent = JSON.stringify(data, null, 2);

    // 5. Actualizar UI de la cámara
    const score = data.score ?? 0;
    const polyAreas = data.polygon_areas_px || [];
    const areaTotal = (data.total_defect_area_px != null) ? Math.round(data.total_defect_area_px) : totalArea(polyAreas);
    const iouVal = data.iou;

    camScore.textContent = score.toFixed(3);
    camAreaTotal.textContent = areaTotal;
    camState.textContent = data.is_anomaly ? 'ANOMALÍA' : 'NORMAL';
    camState.style.color = data.is_anomaly ? 'var(--danger)' : 'var(--ok)';
    camIoU.textContent = (iouVal != null) ? iouVal.toFixed(3) : '-';

    // 6. Mostrar Overlay (Mascara de calor/defecto)
    if(data.overlay_url){
      // TRUCO: cache-busting (?t=...)
      // Agregamos un timestamp para forzar al navegador a recargar la imagen y no usar caché.
      cameraOverlayImg.src = `${data.overlay_url}?t=${Date.now()}`;
    }else{
      cameraOverlayImg.removeAttribute('src');
    }
  }catch(err){
    console.error(err);
  }finally{
    cameraBusy = false; // Liberar semáforo
  }
}

// Listener del botón de cámara
if(cameraToggleBtn){
  cameraToggleBtn.addEventListener('click', ()=>{
    if(cameraStream){
      stopCamera();
    }else{
      startCamera();
    }
  });
}


// ============================================================================
// 7. INICIALIZACIÓN (IIFE)
// ============================================================================
// Se ejecuta automáticamente al cargar el script.

(function init(){
  // 1. Obtener estado de salud y configuración del backend
  fetch('/health').then(r=>r.json()).then(d=>{
    // Mostrar configuración en badges
    healthBox.innerHTML = `
      <span class="badge">img_size=${d.img_size}</span>
      <span class="badge">knn_k=${d.knn_k}</span>
      <span class="badge">thr=${d.threshold}</span>
      ${d.ignore_border_pct ? `<span class="badge">ignore_border=${d.ignore_border_pct}%</span>` : ''}
    `;
  }).catch(()=>{});

  // 2. Render inicial
  renderFileList();
  updateButtons();
  setView('grid'); // Vista por defecto

  // 3. Sincronizar estado inicial de UI
  thrValue.disabled = !thrManual.checked;

  // 4. Verificar soporte de cámara al inicio
  if(cameraToggleBtn && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)){
    cameraToggleBtn.disabled = true;
    cameraToggleBtn.textContent = 'Cámara no soportada';
  }
})();