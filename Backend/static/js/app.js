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
    const f = fileEl.files?.[0];
    if (!f) { alert('Selecciona una imagen.'); return; }

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

    // Cuerpo (multipart/form-data) solo con el archivo
    const fd = new FormData();
    fd.append('file', f);

    // Reset UI
    result.innerHTML = 'Procesando...';
    overlayBox.innerHTML = '';
    jsonBox.textContent = '';

    try {
      const r = await fetch(url, { method: 'POST', body: fd });
      if (!r.ok) throw new Error('Error en el servidor');
      const data = await r.json();

      const badge = data.is_anomaly
        ? `<span class="badge warn">ANOMALÍA</span>`
        : `<span class="badge ok">NORMAL</span>`;

      // Mostrar resultado
      const thrText = (typeof data.threshold === 'number')
        ? data.threshold.toFixed(6)
        : String(data.threshold);

      result.innerHTML = `
        ${badge}
        <div>score=<b>${Number(data.score).toFixed(6)}</b>,
        threshold=<b>${thrText}</b></div>
      `;

      // Overlay (si el backend lo guardó)
      if (data.overlay_url) {
        overlayBox.innerHTML = `
          <label>Overlay:</label><br/>
          <img src="${data.overlay_url}" alt="overlay" />
        `;
      }

      // JSON crudo
      jsonBox.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
      result.innerHTML = `<span class="badge warn">Error</span> ${err.message}`;
    }
  });
});
