"""
Interactive Label Editor - View and edit YOLO bounding box annotations in browser.

Features:
  - Click to select box, drag to move
  - Drag corner handles to resize
  - Delete key to remove selected box
  - Right-click drag to draw new box
  - Number keys (0-9) to change class of selected box
  - Ctrl+S to save current image labels back to .txt file
  - Auto-save on image switch
  - A/D or arrow keys to navigate images

Usage: py scripts/preview_labels.py <dataset_dir>
"""
import http.server
import json
import os
import sys
import urllib.parse
import webbrowser
from pathlib import Path

DATASET_DIR = None  # set in main()

def build_index(dataset_dir: Path):
    """Scan dataset and return (image_list, class_names)."""
    image_files = sorted(dataset_dir.glob("*.jpg"))
    classes_file = dataset_dir / "classes.txt"
    if classes_file.exists():
        class_names = classes_file.read_text(encoding="utf-8").strip().split("\n")
    else:
        class_names = [f"class_{i}" for i in range(20)]

    items = []
    for img_path in image_files:
        label_path = img_path.with_suffix(".txt")
        labels = []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                xc, yc, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
                labels.append({"cls": cls_id, "xc": xc, "yc": yc, "w": w, "h": h})
        items.append({"img": img_path.name, "labels": labels})
    return items, class_names


EDITOR_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Label Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #111827; color: #e5e7eb; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
.toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: #1f2937; border-bottom: 1px solid #374151; flex-wrap: wrap; min-height: 42px; }
.toolbar button { padding: 4px 12px; background: #374151; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 4px; cursor: pointer; font-size: 13px; }
.toolbar button:hover { background: #4b5563; }
.toolbar button.danger { background: #991b1b; border-color: #dc2626; }
.toolbar button.danger:hover { background: #dc2626; }
.toolbar button.active { background: #2563eb; border-color: #3b82f6; }
.toolbar .sep { width: 1px; height: 24px; background: #4b5563; }
.toolbar .info { font-size: 13px; color: #9ca3af; }
.toolbar .save-ok { color: #34d399; font-weight: bold; }
.main { flex: 1; display: flex; overflow: hidden; }
.sidebar { width: 220px; background: #1f2937; border-right: 1px solid #374151; overflow-y: auto; padding: 8px; }
.sidebar h3 { font-size: 12px; color: #9ca3af; margin: 8px 0 4px; text-transform: uppercase; }
.class-btn { display: flex; align-items: center; gap: 6px; padding: 4px 8px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; border: 1px solid transparent; }
.class-btn:hover { background: #374151; }
.class-btn.selected { border-color: #3b82f6; background: #1e3a5f; }
.class-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.box-list-item { display: flex; align-items: center; gap: 4px; padding: 3px 6px; margin: 1px 0; border-radius: 3px; cursor: pointer; font-size: 11px; }
.box-list-item:hover { background: #374151; }
.box-list-item.sel { background: #1e3a5f; border: 1px solid #3b82f6; }
.canvas-area { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
canvas { cursor: crosshair; }
.help { font-size: 11px; color: #6b7280; padding: 8px; line-height: 1.6; }
.help kbd { background: #374151; padding: 1px 4px; border-radius: 2px; font-size: 10px; }
.add-class-row { display: flex; gap: 4px; margin: 6px 0; }
.add-class-row input { flex: 1; padding: 4px 6px; background: #374151; color: #e5e7eb; border: 1px solid #4b5563; border-radius: 3px; font-size: 12px; outline: none; }
.add-class-row input:focus { border-color: #3b82f6; }
.add-class-row button { padding: 4px 8px; background: #059669; color: #fff; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; }
.add-class-row button:hover { background: #047857; }
</style>
</head>
<body>

<div class="toolbar">
  <button onclick="prevImg()">◀ Prev</button>
  <span class="info" id="counter">-</span>
  <button onclick="nextImg()">Next ▶</button>
  <div class="sep"></div>
  <button onclick="saveLabels()" id="saveBtn">💾 Save (Ctrl+S)</button>
  <span id="saveStatus"></span>
  <div class="sep"></div>
  <button onclick="jumpEmpty()">Skip to empty</button>
  <button onclick="jumpLabeled()">Skip to labeled</button>
  <div class="sep"></div>
  <button class="danger" onclick="deleteImage()">🗑 Delete Image (Shift+Del)</button>
  <div class="sep"></div>
  <span class="info" id="fileInfo">-</span>
</div>

<div class="main">
  <div class="sidebar">
    <h3>Classes (click to set draw class)</h3>
    <div class="add-class-row">
      <input type="text" id="newClassName" placeholder="new class name..." onkeydown="if(event.key==='Enter')addClass()">
      <button onclick="addClass()">+ Add</button>
    </div>
    <div id="classList"></div>
    <h3>Boxes <span id="boxCount"></span></h3>
    <div id="boxList"></div>
    <div class="help">
      <b>Controls:</b><br>
      <kbd>A</kbd>/<kbd>D</kbd> or <kbd>←</kbd>/<kbd>→</kbd> navigate<br>
      <kbd>Click</kbd> select box<br>
      <kbd>Drag</kbd> move selected box<br>
      <kbd>Drag corner</kbd> resize<br>
      <kbd>Right-drag</kbd> draw new box<br>
      <kbd>Delete</kbd>/<kbd>Backspace</kbd> remove box<br>
      <kbd>0-9</kbd> change class of selected<br>
      <kbd>Ctrl+S</kbd> save<br>
      Auto-saves on image switch.
    </div>
  </div>
  <div class="canvas-area">
    <canvas id="canvas"></canvas>
  </div>
</div>

<script>
/* ── data injected by Python ── */
const DATA    = __DATA__;
const CLASSES = __CLASSES__;
const COLORS  = [
  "#ef4444","#22c55e","#3b82f6","#eab308","#a855f7","#06b6d4",
  "#f97316","#8b5cf6","#10b981","#ec4899","#84cc16","#0ea5e9",
  "#f43f5e","#4ade80","#818cf8","#facc15","#c084fc","#2dd4bf",
  "#fb923c","#a3e635"
];

let idx = 0;
let selIdx = -1;      // selected box index
let drawClass = 0;    // class for newly drawn boxes
let dirty = false;    // unsaved changes?
let imgW = 1, imgH = 1;
let bgImg = null;

/* ── interaction state ── */
let dragMode = null;  // 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br' | 'draw'
let dragStart = null;
let dragOrig = null;  // original box snapshot
const HANDLE = 7;     // corner handle radius in px

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

/* ── class list sidebar ── */
function buildClassList() {
  const el = document.getElementById('classList');
  el.innerHTML = '';
  CLASSES.forEach((name, i) => {
    const d = document.createElement('div');
    d.className = 'class-btn' + (i === drawClass ? ' selected' : '');
    d.innerHTML = `<div class="class-dot" style="background:${COLORS[i%COLORS.length]}"></div>${i}: ${name}`;
    d.onclick = () => { drawClass = i; buildClassList(); };
    el.appendChild(d);
  });
}
buildClassList();

async function addClass() {
  const input = document.getElementById('newClassName');
  const name = input.value.trim();
  if (!name) return;
  // check duplicate
  if (CLASSES.includes(name)) { alert('Class already exists: ' + name); return; }
  CLASSES.push(name);
  drawClass = CLASSES.length - 1;
  input.value = '';
  buildClassList();
  // persist to server
  try {
    await fetch('/api/add_class', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name }),
    });
  } catch (err) { console.error('Failed to save class:', err); }
}

/* ── box list sidebar ── */
function buildBoxList() {
  const el = document.getElementById('boxList');
  el.innerHTML = '';
  const labels = DATA[idx].labels;
  document.getElementById('boxCount').textContent = `(${labels.length})`;
  labels.forEach((lb, i) => {
    const d = document.createElement('div');
    d.className = 'box-list-item' + (i === selIdx ? ' sel' : '');
    const cname = CLASSES[lb.cls] || `cls_${lb.cls}`;
    d.innerHTML = `<div class="class-dot" style="background:${COLORS[lb.cls%COLORS.length]}"></div>${cname}`;
    d.onclick = () => { selIdx = i; draw(); buildBoxList(); };
    el.appendChild(d);
  });
}

/* ── drawing ── */
function draw() {
  if (!bgImg) return;
  canvas.width = bgImg.width;
  canvas.height = bgImg.height;
  imgW = bgImg.width;
  imgH = bgImg.height;
  ctx.drawImage(bgImg, 0, 0);

  const labels = DATA[idx].labels;
  labels.forEach((lb, i) => {
    const x1 = (lb.xc - lb.w/2) * imgW;
    const y1 = (lb.yc - lb.h/2) * imgH;
    const bw = lb.w * imgW;
    const bh = lb.h * imgH;
    const color = COLORS[lb.cls % COLORS.length];
    const isSel = (i === selIdx);

    // box fill
    ctx.fillStyle = color;
    ctx.globalAlpha = isSel ? 0.25 : 0.1;
    ctx.fillRect(x1, y1, bw, bh);
    ctx.globalAlpha = 1;

    // box border
    ctx.strokeStyle = color;
    ctx.lineWidth = isSel ? 3 : 1.5;
    if (isSel) ctx.setLineDash([]);
    else ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, bw, bh);

    // label tag
    ctx.font = '12px sans-serif';
    const cname = CLASSES[lb.cls] || `cls_${lb.cls}`;
    const tm = ctx.measureText(cname);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x1, y1 - 16, tm.width + 6, 16);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.fillText(cname, x1 + 3, y1 - 3);

    // corner handles for selected
    if (isSel) {
      const corners = [[x1, y1], [x1+bw, y1], [x1, y1+bh], [x1+bw, y1+bh]];
      corners.forEach(([cx, cy]) => {
        ctx.fillStyle = '#fff';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, HANDLE, 0, Math.PI*2);
        ctx.fill();
        ctx.stroke();
      });
    }
  });

  // draw-in-progress box
  if (dragMode === 'draw' && dragStart && dragOrig) {
    const x1 = Math.min(dragStart.x, dragOrig.x);
    const y1 = Math.min(dragStart.y, dragOrig.y);
    const bw = Math.abs(dragStart.x - dragOrig.x);
    const bh = Math.abs(dragStart.y - dragOrig.y);
    const color = COLORS[drawClass % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x1, y1, bw, bh);
    ctx.setLineDash([]);
  }

  updateInfo();
  buildBoxList();
}

function updateInfo() {
  document.getElementById('counter').textContent = `${idx+1} / ${DATA.length}`;
  document.getElementById('fileInfo').textContent = `${DATA[idx].img} | ${DATA[idx].labels.length} boxes` + (dirty ? ' [unsaved]' : '');
}

/* ── load image ── */
function loadImage() {
  selIdx = -1;
  bgImg = new Image();
  bgImg.onload = draw;
  bgImg.src = '/img/' + DATA[idx].img;
}

/* ── navigation ── */
function autoSaveAndGo(fn) {
  if (dirty) saveLabels();
  fn();
  loadImage();
}
function nextImg() { autoSaveAndGo(() => { idx = (idx+1) % DATA.length; }); }
function prevImg() { autoSaveAndGo(() => { idx = (idx-1+DATA.length) % DATA.length; }); }
function jumpEmpty() {
  autoSaveAndGo(() => {
    for (let i = 1; i <= DATA.length; i++) {
      const j = (idx+i) % DATA.length;
      if (DATA[j].labels.length === 0) { idx = j; return; }
    }
  });
}
function jumpLabeled() {
  autoSaveAndGo(() => {
    for (let i = 1; i <= DATA.length; i++) {
      const j = (idx+i) % DATA.length;
      if (DATA[j].labels.length > 0) { idx = j; return; }
    }
  });
}

/* ── hit testing ── */
function getMousePos(e) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width;
  const sy = canvas.height / r.height;
  return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
}

function hitTestHandle(mx, my) {
  // returns 'resize-tl' etc if mouse is on a corner handle of selected box
  if (selIdx < 0) return null;
  const lb = DATA[idx].labels[selIdx];
  const x1 = (lb.xc - lb.w/2) * imgW;
  const y1 = (lb.yc - lb.h/2) * imgH;
  const x2 = x1 + lb.w * imgW;
  const y2 = y1 + lb.h * imgH;
  const corners = [
    { x: x1, y: y1, tag: 'resize-tl' },
    { x: x2, y: y1, tag: 'resize-tr' },
    { x: x1, y: y2, tag: 'resize-bl' },
    { x: x2, y: y2, tag: 'resize-br' },
  ];
  for (const c of corners) {
    if (Math.hypot(mx - c.x, my - c.y) <= HANDLE + 3) return c.tag;
  }
  return null;
}

function hitTestBox(mx, my) {
  // returns index of topmost box under mouse, or -1
  const labels = DATA[idx].labels;
  for (let i = labels.length - 1; i >= 0; i--) {
    const lb = labels[i];
    const x1 = (lb.xc - lb.w/2) * imgW;
    const y1 = (lb.yc - lb.h/2) * imgH;
    const x2 = x1 + lb.w * imgW;
    const y2 = y1 + lb.h * imgH;
    if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return i;
  }
  return -1;
}

/* ── mouse handlers ── */
canvas.addEventListener('mousedown', e => {
  const pos = getMousePos(e);

  // right-click: draw new box
  if (e.button === 2) {
    e.preventDefault();
    dragMode = 'draw';
    dragStart = pos;
    dragOrig = pos;
    return;
  }

  // left-click: check handle first
  const handle = hitTestHandle(pos.x, pos.y);
  if (handle) {
    dragMode = handle;
    dragStart = pos;
    const lb = DATA[idx].labels[selIdx];
    dragOrig = { xc: lb.xc, yc: lb.yc, w: lb.w, h: lb.h };
    return;
  }

  // left-click: select or move
  const hit = hitTestBox(pos.x, pos.y);
  selIdx = hit;
  draw();

  if (hit >= 0) {
    dragMode = 'move';
    dragStart = pos;
    const lb = DATA[idx].labels[hit];
    dragOrig = { xc: lb.xc, yc: lb.yc, w: lb.w, h: lb.h };
  }
});

canvas.addEventListener('mousemove', e => {
  if (!dragMode) return;
  const pos = getMousePos(e);
  const dx = pos.x - dragStart.x;
  const dy = pos.y - dragStart.y;

  if (dragMode === 'draw') {
    dragOrig = pos;
    draw();
    return;
  }

  if (selIdx < 0 || !dragOrig) return;
  const lb = DATA[idx].labels[selIdx];

  if (dragMode === 'move') {
    lb.xc = dragOrig.xc + dx / imgW;
    lb.yc = dragOrig.yc + dy / imgH;
    dirty = true;
    draw();
    return;
  }

  // resize modes
  let ox1 = (dragOrig.xc - dragOrig.w/2) * imgW;
  let oy1 = (dragOrig.yc - dragOrig.h/2) * imgH;
  let ox2 = ox1 + dragOrig.w * imgW;
  let oy2 = oy1 + dragOrig.h * imgH;

  if (dragMode === 'resize-tl') { ox1 += dx; oy1 += dy; }
  if (dragMode === 'resize-tr') { ox2 += dx; oy1 += dy; }
  if (dragMode === 'resize-bl') { ox1 += dx; oy2 += dy; }
  if (dragMode === 'resize-br') { ox2 += dx; oy2 += dy; }

  const nx1 = Math.min(ox1, ox2);
  const ny1 = Math.min(oy1, oy2);
  const nx2 = Math.max(ox1, ox2);
  const ny2 = Math.max(oy1, oy2);

  lb.w = (nx2 - nx1) / imgW;
  lb.h = (ny2 - ny1) / imgH;
  lb.xc = (nx1 + nx2) / 2 / imgW;
  lb.yc = (ny1 + ny2) / 2 / imgH;
  dirty = true;
  draw();
});

canvas.addEventListener('mouseup', e => {
  if (dragMode === 'draw' && dragStart && dragOrig) {
    const x1 = Math.min(dragStart.x, dragOrig.x);
    const y1 = Math.min(dragStart.y, dragOrig.y);
    const bw = Math.abs(dragStart.x - dragOrig.x);
    const bh = Math.abs(dragStart.y - dragOrig.y);
    if (bw > 5 && bh > 5) {
      const newBox = {
        cls: drawClass,
        xc: (x1 + bw/2) / imgW,
        yc: (y1 + bh/2) / imgH,
        w: bw / imgW,
        h: bh / imgH,
      };
      DATA[idx].labels.push(newBox);
      selIdx = DATA[idx].labels.length - 1;
      dirty = true;
    }
  }
  dragMode = null;
  dragStart = null;
  dragOrig = null;
  draw();
});

canvas.addEventListener('contextmenu', e => e.preventDefault());

/* ── keyboard ── */
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveLabels(); return; }
  if (e.key === 'd' || e.key === 'ArrowRight') { nextImg(); return; }
  if (e.key === 'a' || e.key === 'ArrowLeft') { prevImg(); return; }
  if (e.shiftKey && (e.key === 'Delete')) { e.preventDefault(); deleteImage(); return; }
  if ((e.key === 'Delete' || e.key === 'Backspace') && selIdx >= 0) {
    DATA[idx].labels.splice(selIdx, 1);
    selIdx = -1;
    dirty = true;
    draw();
    return;
  }
  // number keys change class
  if (/^[0-9]$/.test(e.key) && selIdx >= 0) {
    DATA[idx].labels[selIdx].cls = parseInt(e.key);
    dirty = true;
    draw();
    return;
  }
});

/* ── save ── */
async function saveLabels() {
  const item = DATA[idx];
  const lines = item.labels.map(lb =>
    `${lb.cls} ${lb.xc.toFixed(6)} ${lb.yc.toFixed(6)} ${lb.w.toFixed(6)} ${lb.h.toFixed(6)}`
  );
  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ img: item.img, labels: lines.join('\\n') }),
    });
    if (resp.ok) {
      dirty = false;
      const el = document.getElementById('saveStatus');
      el.className = 'save-ok';
      el.textContent = 'Saved!';
      setTimeout(() => { el.textContent = ''; }, 1500);
      updateInfo();
    }
  } catch (err) { console.error('Save failed:', err); }
}

/* ── delete image ── */
async function deleteImage() {
  if (DATA.length === 0) return;
  const item = DATA[idx];
  if (!confirm(`Delete ${item.img} and its labels?`)) return;
  try {
    const resp = await fetch('/api/delete_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ img: item.img }),
    });
    if (resp.ok) {
      DATA.splice(idx, 1);
      if (DATA.length === 0) { alert('No images left.'); return; }
      if (idx >= DATA.length) idx = DATA.length - 1;
      dirty = false;
      loadImage();
      const el = document.getElementById('saveStatus');
      el.className = 'save-ok';
      el.textContent = 'Deleted!';
      setTimeout(() => { el.textContent = ''; }, 1500);
    }
  } catch (err) { console.error('Delete failed:', err); }
}

/* ── init ── */
loadImage();
</script>
</body>
</html>"""


class EditorHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with save API endpoint."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/editor.html':
            items, class_names = build_index(DATASET_DIR)
            html = EDITOR_HTML.replace('__DATA__', json.dumps(items))
            html = html.replace('__CLASSES__', json.dumps(class_names))
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            return

        if path.startswith('/img/'):
            fname = urllib.parse.unquote(path[5:])
            fpath = DATASET_DIR / fname
            if fpath.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)
            return

        super().do_GET()

    def do_POST(self):
        if self.path == '/api/save':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            img_name = body['img']
            label_text = body['labels']
            label_path = DATASET_DIR / Path(img_name).with_suffix('.txt')
            label_path.write_text(label_text, encoding='utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return

        if self.path == '/api/delete_image':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            img_name = body['img']
            img_path = DATASET_DIR / img_name
            label_path = DATASET_DIR / Path(img_name).with_suffix('.txt')
            deleted = []
            if img_path.exists():
                img_path.unlink()
                deleted.append(str(img_path.name))
            if label_path.exists():
                label_path.unlink()
                deleted.append(str(label_path.name))
            print(f"  [DELETE] {', '.join(deleted)}")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'deleted': deleted}).encode())
            return

        if self.path == '/api/add_class':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            new_name = body['name'].strip()
            classes_file = DATASET_DIR / 'classes.txt'
            existing = []
            if classes_file.exists():
                existing = classes_file.read_text(encoding='utf-8').strip().split('\n')
                existing = [c for c in existing if c.strip()]
            if new_name not in existing:
                existing.append(new_name)
                classes_file.write_text('\n'.join(existing) + '\n', encoding='utf-8')
                print(f"  [CLASS] Added '{new_name}' (id={len(existing)-1})")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'id': len(existing)-1}).encode())
            return

        self.send_error(404)

    def log_message(self, format, *args):
        # quieter logging
        if '/api/save' in str(args):
            print(f"  [SAVED] {args}")


def main():
    global DATASET_DIR
    if len(sys.argv) < 2:
        print("Usage: py scripts/preview_labels.py <dataset_dir>")
        sys.exit(1)

    DATASET_DIR = Path(sys.argv[1]).resolve()
    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} not found")
        sys.exit(1)

    PORT = 8765
    server = http.server.HTTPServer(("localhost", PORT), EditorHandler)
    url = f"http://localhost:{PORT}/"
    print(f"Label Editor serving at {url}")
    print(f"Dataset: {DATASET_DIR}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
