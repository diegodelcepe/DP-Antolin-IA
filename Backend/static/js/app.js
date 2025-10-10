document.addEventListener('DOMContentLoaded', () => {
  const fileEl   = document.getElementById('file');
  const preview  = document.getElementById('preview');
  const btn      = document.getElementById('btn');
  const thrAuto  = document.getElementById('thrAuto');
  const thrManual= document.getElementById('thrManual');
  const thrValue = document.getElementById('thrValue');
  const modeSel  = document.getElementById('mode');
  const result   = document.getElementById('result');
  const jsonBox  = document.getElementById('json');
  const overlayBox = document.getElementById('overlayBox');
  const cfgBox     = document.getElementById('cfg');

  // Habilitar/deshabilitar el input del umbral manual
  const updateThrInput = () => {
    thrValue.disabled = !thrManual.checked;
    if (!thrManual.checked) thrValue.value = '';
  };
  thrAuto.addEventListener('change', updateThrInput);
  thrManual.addEventListener('change', updateThrInput);
  updateThrInput();

  // Preview de la imagen
  fileEl.addEventListener('change', () => {
    const f = fileEl.files?.[0];
    if (!f) return;
    preview.src = URL.createObjectURL(f);
  });

  // Leer /health para mostrar la config activa
  fetch('/health')
    .then(r => r.json())
    .then(d => {
      cfgBox.innerHTML = `
        <span class="chip">img_size=${d.img_size}</span>
        <span class="chip">knn_k=${d.knn_k}</span>
        <span class="chip">threshold_base=${d.threshold}</span>
      `;
    })
    .catch(() => { /* opcional: silenciar */ });

  // Click "Analizar"
  btn.addEventListener('click', async () => {
  if (!fileEl.files.length) {
    alert('Selecciona una o más imágenes.');
    return;
  }

  // Construimos la URL con los parámetros de query (thr/mode)
  let url = '/predict';
  const qs = new URLSearchParams();
  if (thrManual.checked && thrValue.value) {
    qs.set('thr', thrValue.value);
  }
  if (modeSel.value) {
    qs.set('mode', modeSel.value);
  }
  const qsStr = qs.toString();
  if (qsStr) url += `?${qsStr}`;

  // Reset UI
  result.innerHTML = '';
  overlayBox.innerHTML = '';
  jsonBox.textContent = '';

  // Procesar cada archivo
  for (const f of fileEl.files) {
    const fd = new FormData();
    fd.append('file', f);

    try {
      const r = await fetch(url, { method: 'POST', body: fd });
      if (!r.ok) throw new Error('Error en el servidor');
      const data = await r.json();

      const badge = data.is_anomaly
        ? `<span class="badge warn">ANOMALÍA</span>`
        : `<span class="badge ok">NORMAL</span>`;

      const thrText = (typeof data.threshold === 'number')
        ? data.threshold.toFixed(6)
        : String(data.threshold);

      //Creamos un bloque para esta imagen
      const item = document.createElement('div');
      item.classList.add('result-item');
      item.innerHTML = `
        <h4>${f.name}</h4>
        ${badge}
        <div>score=<b>${Number(data.score).toFixed(6)}</b>,
        threshold=<b>${thrText}</b></div>
        ${data.overlay_url ? `<img src="${data.overlay_url}" alt="overlay" />` : ''}
      `;

      result.appendChild(item);

      // También añadimos su JSON
      jsonBox.textContent += `${f.name}:\n${JSON.stringify(data, null, 2)}\n\n`;

    } catch (err) {
      const item = document.createElement('div');
      item.classList.add('result-item');
      item.innerHTML = `
        <h4>${f.name}</h4>
        <span class="badge warn">Error</span> ${err.message}
      `;
      result.appendChild(item);
    }
  }

});
});