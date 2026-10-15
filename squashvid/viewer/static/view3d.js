// Three.js court. World frame = court frame (x across, y back->front wall, z up); camera.up is +z.
'use strict';
const View3D = (() => {
  const host = document.getElementById('three');
  const hud = document.getElementById('three-hud');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.up.set(0, 0, 1);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.addEventListener('change', () => { dirty = true; });
  let dirty = true, courtGroup = null, W = 6.4, L = 9.75;
  const dyn = new THREE.Group(); scene.add(dyn);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  const players = {};
  const SKEL_EDGES = COCO_EDGES;
  for (const p of ['A', 'B']) {
    const col = new THREE.Color(V.COLORS[p]);
    const g = new THREE.Group();
    const body = new THREE.Group();
    const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 1.4, 16), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.55 }));
    cyl.rotation.x = Math.PI / 2; cyl.position.z = 0.7;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 12), new THREE.MeshBasicMaterial({ color: col }));
    head.position.z = 1.62;
    body.add(cyl, head);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.32, 24), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.5 }));
    disc.position.z = 0.01;
    g.add(body, disc);
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SKEL_EDGES.length * 6), 3));
    const skel = new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ color: col }));
    skel.frustumCulled = false;
    dyn.add(g, skel);
    players[p] = { g, body, skel, sg };
  }

  // Ball: solid sphere when 3D known, else a floor ring + the camera ray ("height unknown").
  const ballMesh = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd54a }));
  const ghost = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.16, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
  const rayGeo = new THREE.BufferGeometry(); rayGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const rayLine = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.45 }));
  rayLine.frustumCulled = false;
  const TRAIL = 24;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
  trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true }));
  trail.frustumCulled = false;
  dyn.add(ballMesh, ghost, rayLine, trail);
  const impacts = [];
  for (let i = 0; i < 8; i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff5d5d, transparent: true }));
    m.visible = false; dyn.add(m); impacts.push(m);
  }

  function poly(pts, mat) {
    const g = new THREE.BufferGeometry(); const a = [];
    pts.forEach(p => a.push(...p));
    g.setAttribute('position', new THREE.Float32BufferAttribute(a, 3));
    return new THREE.Mesh(g, mat);
  }
  function quad(p0, p1, p2, p3, mat) {
    const m = poly([p0, p1, p2, p0, p2, p3], mat); return m;
  }

  function buildCourt(D) {
    if (courtGroup) scene.remove(courtGroup);
    const s = D.court.spec, d = D.court.derived;
    W = s.width; L = s.length;
    const g = new THREE.Group();
    const floor = new THREE.MeshBasicMaterial({ color: 0x8a6a3a, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
    const wall = new THREE.MeshBasicMaterial({ color: 0x6fa8dc, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
    const front = new THREE.MeshBasicMaterial({ color: 0x6fa8dc, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false });
    g.add(quad([0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0], floor));
    g.add(quad([0, L, 0], [W, L, 0], [W, L, d.front_out_z], [0, L, d.front_out_z], front));
    g.add(quad([0, 0, 0], [W, 0, 0], [W, 0, d.back_out_z], [0, 0, d.back_out_z], wall));
    g.add(quad([0, 0, 0], [0, L, 0], [0, L, d.front_out_z], [0, 0, d.back_out_z], wall));
    g.add(quad([W, 0, 0], [W, L, 0], [W, L, d.front_out_z], [W, 0, d.back_out_z], wall));
    // tin (0..tin_top on the front wall)
    g.add(quad([0, L - 0.003, 0], [W, L - 0.003, 0], [W, L - 0.003, s.tin_top], [0, L - 0.003, s.tin_top],
      new THREE.MeshBasicMaterial({ color: 0xd9534f, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false })));
    for (const ln of D.court.lines) {
      if (ln.pts.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...ln.pts[0]), new THREE.Vector3(...ln.pts[ln.pts.length - 1])]);
      const col = ln.kind === 'edge' ? 0xffffff : (ln.surface === 'floor' ? 0xffe27a : 0x8fd0ff);
      g.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: col, transparent: ln.kind === 'edge', opacity: ln.kind === 'edge' ? 0.5 : 1 })));
    }
    courtGroup = g; scene.add(g);
    preset('behind');
  }

  function preset(name) {
    const c = controls.target;
    if (name === 'behind') { camera.position.set(W / 2, -3.4, 2.4); c.set(W / 2, L * 0.55, 0.9); }
    else if (name === 'side') { camera.position.set(-9.5, L / 2, 3.6); c.set(W / 2, L / 2, 1.0); }
    else { camera.position.set(W / 2, L / 2 - 4.5, 15); c.set(W / 2, L / 2, 0); }
    controls.update(); dirty = true;
  }
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => preset(b.dataset.view)));

  // A high ball's camera ray can meet the floor beyond the walls; such a projection says nothing, so it is not drawn.
  const inCourt = (x, y) => Number.isFinite(x) && x > -0.5 && x < W + 0.5 && y > -0.5 && y < L + 0.5;

  function update(f) {
    const D = V.D; if (!D) return;
    const cam = cameraAt(D, f);
    for (const p of ['A', 'B']) {
      const P = D[p], o = players[p];
      const x = P.pos ? P.pos[2 * f] : NaN, y = P.pos ? P.pos[2 * f + 1] : NaN;
      const have = Number.isFinite(x);
      o.g.visible = have;
      if (have) o.g.position.set(x, y, 0);
      let skel = false;
      if (have && cam && V.opts.skel3d && P.kp && P.status[f] === 2) {
        const pos = o.sg.attributes.position.array, ks = f * 17, kb = f * 34;
        const pts = [];
        for (let i = 0; i < 17; i++) pts.push(P.ks[ks + i] > 0.3 && Number.isFinite(P.kp[kb + 2 * i]) ? liftKeypoint(cam, [x, y], P.kp[kb + 2 * i], P.kp[kb + 2 * i + 1]) : null);
        let n = 0;
        SKEL_EDGES.forEach(([i, j], e) => {
          const a = pts[i], b = pts[j];
          if (a && b) { pos.set(a, e * 6); pos.set(b, e * 6 + 3); n++; } else pos.fill(0, e * 6, e * 6 + 6);
        });
        o.sg.attributes.position.needsUpdate = true;
        skel = n > 3;
      }
      o.skel.visible = skel;
      o.body.visible = have && !skel;
      // body is stylised: face the camera plane is irrelevant for a cylinder.
    }
    // ball
    const known = D.ball_xyz && Number.isFinite(D.ball_xyz[3 * f]);
    const uvOk = D.ball_uv && Number.isFinite(D.ball_uv[2 * f]);
    ballMesh.visible = !!known; ghost.visible = false; rayLine.visible = false;
    let mode = 'no ball';
    if (known) {
      ballMesh.position.set(D.ball_xyz[3 * f], D.ball_xyz[3 * f + 1], D.ball_xyz[3 * f + 2]);
      mode = D.ball_anchored && D.ball_anchored[f] ? 'ball: 3D (anchored on a wall/floor event)' : 'ball: 3D';
    } else if (uvOk && D.ball_floor && inCourt(D.ball_floor[2 * f], D.ball_floor[2 * f + 1]) && cam) {
      const gx = D.ball_floor[2 * f], gy = D.ball_floor[2 * f + 1];
      ghost.visible = true; ghost.position.set(gx, gy, 0.02);
      const arr = rayGeo.attributes.position.array;
      arr.set(cam.c, 0); arr.set([gx, gy, 0], 3);
      rayGeo.attributes.position.needsUpdate = true; rayLine.computeLineDistances(); rayLine.visible = true;
      mode = 'ball: ESTIMATED (floor projection of the camera ray, height unknown)';
    } else if (uvOk) mode = 'ball seen, no 3D position';
    // trail from whichever source is present
    const tp = trailGeo.attributes.position.array, tc = trailGeo.attributes.color.array;
    let n = 0;
    for (let g = Math.max(0, f - TRAIL + 1); g <= f; g++) {
      let q = null;
      if (D.ball_xyz && Number.isFinite(D.ball_xyz[3 * g])) q = [D.ball_xyz[3 * g], D.ball_xyz[3 * g + 1], D.ball_xyz[3 * g + 2]];
            if (!q) continue;
      const a = (g - (f - TRAIL + 1)) / TRAIL;
      tp.set(q, n * 3); tc.set([0.15 + 0.85 * a, 0.14 + 0.72 * a, 0.05 + 0.2 * a], n * 3); n++;
    }
    trailGeo.setDrawRange(0, n);
    trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
    // recent wall/floor impacts
    const t = D.t[f]; let k = 0;
    for (const e of D.bounces) {
      if (e.t > t || e.t < t - 1.5 || !Number.isFinite(e.x) || !Number.isFinite(e.z) || !Number.isFinite(e.y)) continue;
      if (k >= impacts.length) break;
      const m = impacts[k++]; m.visible = true; m.position.set(e.x, e.y, e.z);
      m.material.opacity = 1 - (t - e.t) / 1.5;
    }
    for (; k < impacts.length; k++) impacts[k].visible = false;
    hud.textContent = mode;
    dirty = true;
  }

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
  }
  new ResizeObserver(resize).observe(host);
  (function loop() { if (dirty) { renderer.setClearColor(getComputedStyle(document.body).backgroundColor); renderer.render(scene, camera); dirty = false; } requestAnimationFrame(loop); })();
  return { buildCourt, update, preset };
})();
