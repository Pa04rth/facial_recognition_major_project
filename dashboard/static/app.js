const $ = (id) => document.getElementById(id);
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString();

let eventsSocket = null;

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    const el = $("status");
    if (s.connected) {
      el.textContent = `connected — ${s.server_info?.server_id || ""}`;
      el.className = "status connected";
      $("video-placeholder").hidden = true;
      const img = $("stream");
      if (!img.src || !img.src.includes("/api/stream")) {
        img.src = "/api/stream?t=" + Date.now();
      }
      img.hidden = false;
    } else {
      el.textContent = "disconnected";
      el.className = "status disconnected";
      $("stream").hidden = true;
      $("stream").removeAttribute("src");
      $("video-placeholder").hidden = false;
    }
  } catch (e) {
    console.error(e);
  }
}

async function connect() {
  const pi_url = $("pi-url").value.trim();
  const token = $("pi-token").value.trim();
  if (!pi_url || !token) return;
  $("status").textContent = "connecting…";
  $("status").className = "status reconnecting";
  try {
    const r = await fetch("/api/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pi_url, token }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
      $("status").textContent = `error: ${detail}`;
      $("status").className = "status disconnected";
      return;
    }
    await refreshStatus();
    openEventsWS();
    await loadEnrolled();
    await loadGallery();
  } catch (e) {
    $("status").textContent = `error: ${e.message}`;
    $("status").className = "status disconnected";
  }
}

async function disconnect() {
  await fetch("/api/disconnect", { method: "POST" });
  if (eventsSocket) eventsSocket.close();
  await refreshStatus();
}

function openEventsWS() {
  if (eventsSocket) eventsSocket.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  eventsSocket = new WebSocket(`${proto}://${location.host}/api/events`);
  eventsSocket.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    prependMatch(ev);
    addGalleryCard(ev);
  };
  eventsSocket.onclose = () => {
    setTimeout(openEventsWS, 2000);
  };
}

function prependMatch(ev) {
  const list = $("match-list");
  const row = document.createElement("div");
  row.className = "match-row";
  const img = document.createElement("img");
  img.src = "data:image/jpeg;base64," + ev.crop_b64;
  const meta = document.createElement("div");
  meta.innerHTML = `<div class="match-name">${escapeHtml(ev.name)}</div>
                    <div class="match-meta">${fmtTime(ev.ts)} &middot; score ${ev.score.toFixed(2)}</div>`;
  row.appendChild(img);
  row.appendChild(meta);
  list.prepend(row);
  while (list.childElementCount > 50) list.removeChild(list.lastChild);
}

function addGalleryCard(ev) {
  const grid = $("gallery");
  const card = document.createElement("div");
  card.className = "gallery-card";
  card.dataset.name = ev.name;
  card.innerHTML = `
    <img src="data:image/jpeg;base64,${ev.crop_b64}" alt="">
    <div class="meta">
      <div class="name">${escapeHtml(ev.name)}</div>
      <div class="ts">${fmtTime(ev.ts)} &middot; ${ev.score.toFixed(2)}</div>
    </div>`;
  grid.prepend(card);
  applyGalleryFilter();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function loadEnrolled() {
  const filterSel = $("gallery-filter");
  try {
    const r = await fetch("/api/enrolled");
    if (!r.ok) {
      $("enrolled-list").innerHTML = "<em>not connected</em>";
      return;
    }
    const people = await r.json();
    $("enrolled-list").innerHTML = people.length
      ? people.map(p => `
          <span class="enrolled-chip">
            ${escapeHtml(p.name)} <span class="photos">${p.n_photos} photos</span>
            <button class="danger" data-name="${escapeHtml(p.name)}">remove</button>
          </span>`).join("")
      : "<em>no one enrolled yet — add someone below</em>";
    document.querySelectorAll(".enrolled-chip .danger").forEach(b => {
      b.onclick = () => removePerson(b.dataset.name);
    });

    const seen = new Set();
    filterSel.innerHTML = '<option value="">All people</option>';
    people.forEach(p => {
      if (!seen.has(p.name)) {
        seen.add(p.name);
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.name;
        filterSel.appendChild(opt);
      }
    });
  } catch (e) {
    $("enrolled-list").innerHTML = "<em>not connected</em>";
  }
}

async function removePerson(name) {
  if (!confirm(`Remove "${name}" from the search list?`)) return;
  const r = await fetch(`/api/enrolled/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (r.ok) {
    await loadEnrolled();
  } else {
    alert("failed to remove");
  }
}

async function enrollPerson(ev) {
  ev.preventDefault();
  const name = $("enroll-name").value.trim();
  const files = $("enroll-photos").files;
  if (!name || !files.length) return;

  const result = $("enroll-result");
  result.className = "result";
  result.textContent = "uploading…";

  const fd = new FormData();
  fd.append("name", name);
  for (const f of files) fd.append("photos", f);

  try {
    const r = await fetch("/api/enroll", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) {
      result.className = "result err";
      result.textContent = "error: " + (data.detail || r.statusText);
      return;
    }
    result.className = "result ok";
    result.textContent = `added ${data.added}, skipped ${data.skipped.length}` +
      (data.skipped.length ? " (" + data.skipped.join("; ") + ")" : "");
    $("enroll-name").value = "";
    $("enroll-photos").value = "";
    await loadEnrolled();
  } catch (e) {
    result.className = "result err";
    result.textContent = "error: " + e.message;
  }
}

async function loadGallery() {
  const r = await fetch("/api/captures");
  const data = await r.json();
  const grid = $("gallery");
  grid.innerHTML = "";
  const all = [];
  for (const [name, items] of Object.entries(data)) {
    for (const it of items) all.push({ ...it, name });
  }
  all.sort((a, b) => b.ts - a.ts);
  for (const it of all) {
    const card = document.createElement("div");
    card.className = "gallery-card";
    card.dataset.name = it.name;
    card.innerHTML = `
      <img src="${it.url}" alt="">
      <div class="meta">
        <div class="name">${escapeHtml(it.name)}</div>
        <div class="ts">${fmtTime(it.ts)}</div>
      </div>`;
    grid.appendChild(card);
  }
  applyGalleryFilter();
}

function applyGalleryFilter() {
  const f = $("gallery-filter").value;
  document.querySelectorAll("#gallery .gallery-card").forEach(c => {
    c.style.display = (!f || c.dataset.name === f) ? "" : "none";
  });
}

$("connect-btn").onclick = connect;
$("disconnect-btn").onclick = disconnect;
$("enroll-form").onsubmit = enrollPerson;
$("gallery-filter").onchange = applyGalleryFilter;

refreshStatus().then(() => {
  loadGallery();
  loadEnrolled();
  fetch("/api/status").then(r => r.json()).then(s => {
    if (s.connected) openEventsWS();
  });
});
