// --- elementos ---
const filesInput = document.getElementById('files');
const folderInput = document.getElementById('folder');
const dropzone = document.getElementById('dropzone');
const fileList = document.getElementById('fileList');

const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');

const thrAuto = document.getElementById('thrAuto');
const thrManual = document.getElementById('thrManual');
const thrValue = document.getElementById('thrValue');
const modeSel = document.getElementById('mode');

const healthBox = document.getElementById('health');
const jsonBox = document.getElementById('jsonBox');

const kpiTotal = document.getElementById('kpiTotal');
const kpiNorm  = document.getElementById('kpiNorm');
const kpiAnom  = document.getElementById('kpiAnom');
const kpiRate  = document.getElementById('kpiRate');
const kpiArea  = document.getElementById('kpiArea');

const gridView = document.getElementById('gridView');
const tableView = document.getElementById('tableView');
const resultsTable = document.getElementById('resultsTable');

const tabGrid = document.getElementById('tabGrid');
const tabTable = document.getElementById('tabTable');
const toggleViewBtn = document.getElementById('toggleViewBtn');
const downloadCsvBtn = document.getElementById('downloadCsvBtn');

// Cámara
const cameraToggleBtn = document.getElementById('cameraToggleBtn');
const cameraPanel = document.getElementById('cameraPanel');
const cameraVideo = document.getElementById('cameraVideo');
const cameraOverlayImg = document.getElementById('cameraOverlay');
const camScore = document.getElementById('camScore');
const camAreaTotal = document.getElementById('camAreaTotal');
const camState = document.getElementById('camState');
const camIoU = document.getElementById('camIoU');

// --- estado ---
let selectedFiles = []; // File[]
let lastBatch = null;   // respuesta JSON del backend

// Estado de cámara
let cameraStream = null;
let cameraIntervalId = null;
let cameraBusy = false;
const CAMERA_INTERVAL_MS = 1500; // 1.5 segundos

// --- helpers ---
function isImage(f){
  return f && f.type && f.type.startsWith('image/');
}
function addFiles(fileListLike){
  const arr = Array.from(fileListLike).filter(isImage);
  selectedFiles.push(...arr);
  // eliminar duplicados por nombre + size (heurística simple)
  const map = new Map();
  selectedFiles.forEach(f=>{
    const key = `${f.name}_${f.size}`;
    if(!map.has(key)) map.set(key, f);
  });
  selectedFiles = Array.from(map.values());
  renderFileList();
  updateButtons();
}
function renderFileList(){
  if(selectedFiles.length === 0){
    fileList.innerHTML = '<div class="muted">No hay archivos seleccionados</div>';
    return;
  }
  fileList.innerHTML = selectedFiles.map(f => `
    <div class="item"><span>${f.webkitRelativePath || f.name}</span><span class="muted">${(f.size/1024).toFixed(0)} KB</span></div>
  `).join('');
}
function updateButtons(){
  const has = selectedFiles.length > 0;
  analyzeBtn.disabled = !has;
  toggleViewBtn.disabled = !lastBatch;
  downloadCsvBtn.disabled = !lastBatch;
}
function setKPIs(summary){
  const total = summary?.total_images || 0;
  const anom = summary?.anomalies || 0;
  const norm = summary?.normals || 0;
  const rate = summary?.defect_rate ? (summary.defect_rate*100).toFixed(1) : '0.0';
  const area = summary?.avg_defect_area_px ? Math.round(summary.avg_defect_area_px) : 0;

  kpiTotal.textContent = total;
  kpiAnom.textContent  = anom;
  kpiNorm.textContent  = norm;
  kpiRate.textContent  = `${rate}%`;
  kpiArea.textContent  = area;
}
function clearResults(){
  lastBatch = null;
  gridView.innerHTML = '';
  resultsTable.innerHTML = '';
  jsonBox.textContent = '';
  setKPIs({});
  updateButtons();
}
function formatStateCell(is_anomaly){
  return is_anomaly
    ? `<span class="state bad">ANOMALÍA</span>`
    : `<span class="state ok">NORMAL</span>`;
}
function totalArea(areas){
  if(!areas || !areas.length) return 0;
  return Math.round(areas.reduce((a,b)=>a+b,0));
}
function fileThumbURL(file){
  return URL.createObjectURL(file);
}
function asCsv(data){
  const rows = [
    ["filename","score","threshold","is_anomaly","num_polygons","area_total_px","area_max_px","iou","overlay_url"]
  ];
  data.results.forEach(r=>{
    const polyAreas = r.polygon_areas_px || [];
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
  return rows.map(r=>r.map(v => {
    const s = String(v ?? '');
    return /[",;\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
  }).join(',' )).join('\n');
}

// --- UI eventos ---
filesInput.addEventListener('change', e=>{
  if(e.target.files) addFiles(e.target.files);
});
folderInput.addEventListener('change', e=>{
  if(e.target.files) addFiles(e.target.files);
});
clearBtn.addEventListener('click', ()=>{
  selectedFiles = [];
  filesInput.value = '';
  folderInput.value = '';
  renderFileList();
  clearResults();
  updateButtons();
});

dropzone.addEventListener('dragover', e=>{
  e.preventDefault();
  dropzone.classList.add('drag');
});
dropzone.addEventListener('dragleave', ()=>{
  dropzone.classList.remove('drag');
});
dropzone.addEventListener('drop', e=>{
  e.preventDefault();
  dropzone.classList.remove('drag');
  if(e.dataTransfer.files) addFiles(e.dataTransfer.files);
});

thrManual.addEventListener('change', ()=> thrValue.disabled = !thrManual.checked);
thrAuto.addEventListener('change', ()=> thrValue.disabled = !thrManual.checked);

// tabs & toggles
function setView(mode){ // 'grid' | 'table'
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
  const isGrid = !gridView.classList.contains('hidden');
  setView(isGrid ? 'table' : 'grid');
});

// descargar CSV
downloadCsvBtn.addEventListener('click', ()=>{
  if(!lastBatch) return;
  const csv = asCsv(lastBatch);
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'resultados_patchcore.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// analizar lote
analyzeBtn.addEventListener('click', async ()=>{
  if(selectedFiles.length === 0){
    alert('Selecciona imágenes o carpeta.');
    return;
  }
  clearResults();

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f));

  if(thrManual.checked && thrValue.value) fd.append('thr', thrValue.value);
  if(modeSel.value) fd.append('mode', modeSel.value);

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Procesando...';

  try{
    const r = await fetch('/predict_batch', { method:'POST', body: fd });
    if(!r.ok) throw new Error('Error del servidor');
    const data = await r.json();
    lastBatch = data;

    renderResults(data, selectedFiles);
    jsonBox.textContent = JSON.stringify(data, null, 2);
    updateButtons();
  }catch(err){
    jsonBox.textContent = `Error: ${err.message}`;
  }finally{
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analizar';
  }
});

function renderResults(data, files){
  // KPIs
  setKPIs(data.summary);

  // Tabla
  resultsTable.innerHTML = data.results.map(r => {
    const polyAreas = r.polygon_areas_px || [];
    const areaTotal = (r.total_defect_area_px != null)
      ? Math.round(r.total_defect_area_px)
      : totalArea(polyAreas);
    const areaMax = (r.max_defect_area_px != null)
      ? Math.round(r.max_defect_area_px)
      : (polyAreas.length ? Math.round(Math.max(...polyAreas)) : 0);
    const iouStr = (r.iou != null) ? r.iou.toFixed(3) : '-';

    return `
      <tr>
        <td>${r.filename}</td>
        <td>${r.score.toFixed(6)}</td>
        <td>${Number(r.threshold).toFixed(6)}</td>
        <td>${formatStateCell(r.is_anomaly)}</td>
        <td>${(r.polygons || []).length}</td>
        <td>${areaTotal}</td>
        <td>${areaMax}</td>
        <td>${iouStr}</td>
        <td>${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Ver</a>` : '-'}</td>
      </tr>
    `;
  }).join('');

  // Grid (galería)
  const fileMap = new Map();
  files.forEach(f => fileMap.set(f.name, f));
  gridView.innerHTML = data.results.map(r=>{
    const f = fileMap.get(r.filename);
    const thumb = f ? fileThumbURL(f) : '';
    const polyCount = (r.polygons || []).length;
    const polyAreas = r.polygon_areas_px || [];
    const areaTotal = (r.total_defect_area_px != null)
      ? Math.round(r.total_defect_area_px)
      : totalArea(polyAreas);
    const areaMax = (r.max_defect_area_px != null)
      ? Math.round(r.max_defect_area_px)
      : (polyAreas.length ? Math.round(Math.max(...polyAreas)) : 0);
    const iouStr = (r.iou != null) ? r.iou.toFixed(3) : '-';

    return `
      <article class="card-img">
        ${thumb ? `<img class="thumb" src="${thumb}" alt="${r.filename}"/>` : ''}
        <div class="pad">
          <div class="row">
            <div class="state ${r.is_anomaly ? 'bad' : 'ok'}">${r.is_anomaly ? 'ANOMALÍA' : 'NORMAL'}</div>
            ${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Overlay</a>` : ''}
          </div>
          <div class="meta">score=<b>${r.score.toFixed(6)}</b> • thr=${Number(r.threshold).toFixed(6)}</div>
          <div class="meta">polígonos=${polyCount} • área total=${areaTotal}px • área máx=${areaMax}px</div>
          <div class="meta">IoU=${iouStr}</div>
          <div class="meta">${r.filename}</div>
        </div>
      </article>
    `;
  }).join('');
}

// --- Cámara en tiempo real --- //
async function startCamera(){
  if(cameraStream) return;
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    alert('La cámara no está soportada en este navegador.');
    return;
  }
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
    cameraStream = stream;
    cameraVideo.srcObject = stream;
    cameraPanel.classList.remove('hidden');
    cameraToggleBtn.textContent = 'Detener cámara';

    cameraIntervalId = setInterval(captureFrameAndPredict, CAMERA_INTERVAL_MS);
  }catch(err){
    alert('No se pudo acceder a la cámara: ' + err.message);
  }
}

function stopCamera(){
  if(cameraIntervalId){
    clearInterval(cameraIntervalId);
    cameraIntervalId = null;
  }
  if(cameraStream){
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraToggleBtn.textContent = 'Activar cámara';
  cameraPanel.classList.add('hidden');

  camScore.textContent = '0.000';
  camAreaTotal.textContent = '0';
  camState.textContent = '-';
  camState.style.color = '';
  camIoU.textContent = '-';
  cameraOverlayImg.removeAttribute('src');
}

function makeCameraFilename(){
  const now = new Date();
  // yyyyMMdd_HHmmssfff
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

async function captureFrameAndPredict(){
  if(!cameraStream || cameraBusy || !cameraVideo.videoWidth) return;
  cameraBusy = true;

  try{
    const canvas = document.createElement('canvas');
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
    if(!blob) return;

    const filename = makeCameraFilename();
    const fd = new FormData();
    fd.append('file', new File([blob], filename, { type:'image/jpeg' }));

    const params = new URLSearchParams();
    params.append('source', 'camera');
    if(thrManual.checked && thrValue.value) params.append('thr', thrValue.value);
    if(modeSel.value) params.append('mode', modeSel.value);

    const resp = await fetch(`/predict?${params.toString()}`, {
      method:'POST',
      body: fd
    });
    if(!resp.ok) throw new Error('Error del servidor en cámara');
    const data = await resp.json();

    jsonBox.textContent = JSON.stringify(data, null, 2);

    const score = data.score ?? 0;
    const polyAreas = data.polygon_areas_px || [];
    const areaTotal = (data.total_defect_area_px != null)
      ? Math.round(data.total_defect_area_px)
      : totalArea(polyAreas);
    const iouVal = data.iou;

    camScore.textContent = score.toFixed(3);
    camAreaTotal.textContent = areaTotal;
    camState.textContent = data.is_anomaly ? 'ANOMALÍA' : 'NORMAL';
    camState.style.color = data.is_anomaly ? 'var(--danger)' : 'var(--ok)';
    camIoU.textContent = (iouVal != null) ? iouVal.toFixed(3) : '-';

    if(data.overlay_url){
      // cache-busting para ver siempre el último overlay
      cameraOverlayImg.src = `${data.overlay_url}?t=${Date.now()}`;
    }else{
      cameraOverlayImg.removeAttribute('src');
    }
  }catch(err){
    console.error(err);
  }finally{
    cameraBusy = false;
  }
}

// botón de la cámara
if(cameraToggleBtn){
  cameraToggleBtn.addEventListener('click', ()=>{
    if(cameraStream){
      stopCamera();
    }else{
      startCamera();
    }
  });
}

// --- health/config ---
(function init(){
  fetch('/health').then(r=>r.json()).then(d=>{
    healthBox.innerHTML = `
      <span class="badge">img_size=${d.img_size}</span>
      <span class="badge">knn_k=${d.knn_k}</span>
      <span class="badge">thr=${d.threshold}</span>
      ${d.ignore_border_pct ? `<span class="badge">ignore_border=${d.ignore_border_pct}%</span>` : ''}
    `;
  }).catch(()=>{});
  renderFileList();
  updateButtons();
  setView('grid');

  thrValue.disabled = !thrManual.checked;

  if(cameraToggleBtn && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)){
    cameraToggleBtn.disabled = true;
    cameraToggleBtn.textContent = 'Cámara no soportada';
  }
})();
