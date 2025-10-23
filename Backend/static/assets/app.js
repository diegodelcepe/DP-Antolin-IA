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

// --- estado ---
let selectedFiles = []; // File[]
let lastBatch = null;   // respuesta JSON del backend

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
    ["filename","score","threshold","is_anomaly","num_polygons","area_total_px","overlay_url"]
  ];
  data.results.forEach(r=>{
    rows.push([
      r.filename,
      r.score,
      r.threshold,
      r.is_anomaly ? 1 : 0,
      (r.polygons || []).length,
      totalArea(r.polygon_areas_px || []),
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

// analizar
analyzeBtn.addEventListener('click', async ()=>{
  if(selectedFiles.length === 0){
    alert('Selecciona imágenes o carpeta.');
    return;
  }
  clearResults();

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f));

  // umbral y modo
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
  resultsTable.innerHTML = data.results.map(r => `
    <tr>
      <td>${r.filename}</td>
      <td>${r.score.toFixed(6)}</td>
      <td>${Number(r.threshold).toFixed(6)}</td>
      <td>${formatStateCell(r.is_anomaly)}</td>
      <td>${(r.polygons || []).length}</td>
      <td>${totalArea(r.polygon_areas_px || [])}</td>
      <td>${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Ver</a>` : '-'}</td>
    </tr>
  `).join('');

  // Grid (galería)
  // Mapeo rápido filename->File para previews
  const fileMap = new Map();
  files.forEach(f => fileMap.set(f.name, f));
  gridView.innerHTML = data.results.map(r=>{
    const f = fileMap.get(r.filename);
    const thumb = f ? fileThumbURL(f) : '';
    const polyCount = (r.polygons || []).length;
    const areaSum = totalArea(r.polygon_areas_px || []);
    return `
      <article class="card-img">
        ${thumb ? `<img class="thumb" src="${thumb}" alt="${r.filename}"/>` : ''}
        <div class="pad">
          <div class="row">
            <div class="state ${r.is_anomaly ? 'bad' : 'ok'}">${r.is_anomaly ? 'ANOMALÍA' : 'NORMAL'}</div>
            ${r.overlay_url ? `<a class="link" href="${r.overlay_url}" target="_blank">Overlay</a>` : ''}
          </div>
          <div class="meta">score=<b>${r.score.toFixed(6)}</b> • thr=${Number(r.threshold).toFixed(6)}</div>
          <div class="meta">polígonos=${polyCount} • área total=${areaSum}px</div>
          <div class="meta">${r.filename}</div>
        </div>
      </article>
    `;
  }).join('');
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

  // habilitar número al pasar a "manual"
  thrValue.disabled = !thrManual.checked;
})();
