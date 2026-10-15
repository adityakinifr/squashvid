// Three.js court. World frame = court frame (x across, y back->front wall, z up); camera.up is +z.
'use strict';
// ---- Solid 3D figures driven by the lifted joints (bodies.parquet; H36M-17 order, court metres) ----------------------
// A rigged humanoid (Mixamo-rig Xbot.glb from the three.js examples, personal use) is posed bone by bone from the joints;
// if it cannot load, a shaped capsule figure with the same interface is used. Both are lit, tinted in the player colour.
const RIG_URL = '/static/vendor/models/Xbot.glb';
const FIG = { buffer: null, state: 'idle', waiters: [] };  // idle | loading | ready | failed
function loadRigBuffer(cb) {
  if (FIG.state === 'ready' || FIG.state === 'failed') return cb();
  FIG.waiters.push(cb);
  if (FIG.state === 'loading') return;
  FIG.state = 'loading';
  const done = (ok) => { FIG.state = ok ? 'ready' : 'failed'; FIG.waiters.splice(0).forEach(w => w()); };
  const s = document.createElement('script');
  s.src = '/static/vendor/GLTFLoader.js';  // classic-script r128 build; loaded here so index.html stays unchanged
  s.onload = () => fetch(RIG_URL).then(r => r.arrayBuffer()).then(b => { FIG.buffer = b; done(true); }).catch(e => { console.warn('figure model failed to load:', e); done(false); });
  s.onerror = () => { console.warn('GLTFLoader failed to load'); done(false); };
  document.head.appendChild(s);
}

const _v = [0, 1, 2, 3, 4, 5].map(() => new THREE.Vector3());
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _m = new THREE.Matrix4();
const UP = new THREE.Vector3(0, 0, 1);

// Rotation taking the rest pose's reference axes to a body part's measured axes: y = up, z = forward = left x up.
function frameQuat(up, left, out) {
  const y = _v[0].copy(up).normalize();
  const z = _v[1].crossVectors(left, y).normalize();
  const x = _v[2].crossVectors(y, z);
  return out.setFromRotationMatrix(_m.makeBasis(x, y, z));
}

const SHIRT = /^(Spine|Spine1|Spine2|Shoulder|Arm)$/, SHORTS = /^(Hips|UpLeg)$/, SHOE = /^(Foot|ToeBase|Toe_End)$/;
function boneGroup(name) {
  const n = name.replace(/^mixamorig:?/, '').replace(/^(Left|Right)/, '');
  if (SHIRT.test(n)) return 'shirt';
  if (SHORTS.test(n)) return 'shorts';
  if (SHOE.test(n)) return 'shoe';
  return 'skin';
}
const TONES = { skin: new THREE.Color('#c8946f'), shorts: new THREE.Color('#252a36'), shoe: new THREE.Color('#e9e9ee') };

class RigFigure {
  constructor(gltf, col) {
    this.root = new THREE.Group();
    this.root.rotation.x = Math.PI / 2;  // the model is y-up, the court z-up
    this.model = gltf.scene;
    this.root.add(this.model);
    this.bones = {};
    this.model.traverse(o => { if (o.isBone) this.bones[o.name.replace(/^mixamorig:?/, '')] = o; });
    this.model.traverse(o => { if (o.isSkinnedMesh) { o.frustumCulled = false; this.tint(o, col); } });
    this.root.updateMatrixWorld(true);
    this.rp = {}; this.rq = {};
    for (const [n, b] of Object.entries(this.bones)) { this.rp[n] = b.getWorldPosition(new THREE.Vector3()); this.rq[n] = b.getWorldQuaternion(new THREE.Quaternion()); }
    const P = this.rp, line = (a, b) => P[a].distanceTo(P[b]);
    this.restLeg = (line('RightUpLeg', 'RightLeg') + line('RightLeg', 'RightFoot') + line('LeftUpLeg', 'LeftLeg') + line('LeftLeg', 'LeftFoot')) / 2;
    this.restTrunk = line('Hips', 'Neck') + line('Neck', 'HeadTop_End');
    this.restAnkle = (P.RightFoot.z + P.LeftFoot.z) / 2;
    const L = P.LeftUpLeg.clone().sub(P.RightUpLeg), S = P.LeftArm.clone().sub(P.RightArm);
    this.restHip = L; this.restSho = S;
    this.restFwd = new THREE.Vector3().crossVectors(L, UP).normalize();
    this.frames = {};
    for (const [n, w] of [['Hips', 0], ['Spine', 0.33], ['Spine1', 0.66], ['Spine2', 1], ['Head', 0]]) {
      const left = n === 'Head' ? new THREE.Vector3().crossVectors(UP, this.restFwd) : L.clone().normalize().lerp(S.clone().normalize(), w).normalize();
      this.frames[n] = frameQuat(UP, left, new THREE.Quaternion()).invert();
    }
  }

  tint(mesh, col) {
    const names = mesh.skeleton.bones.map(b => b.name), g = mesh.geometry;
    const si = g.attributes.skinIndex, sw = g.attributes.skinWeight, n = si.count, out = new Float32Array(n * 3);
    const tone = { ...TONES, shirt: col }, c = new THREE.Color();
    const grp = names.map(boneGroup);
    for (let i = 0; i < n; i++) {
      let r = 0, gg = 0, b = 0, w = 0;
      for (let k = 0; k < 4; k++) {
        const wt = sw.array[i * 4 + k]; if (!wt) continue;
        c.copy(tone[grp[si.array[i * 4 + k]]]); r += c.r * wt; gg += c.g * wt; b += c.b * wt; w += wt;
      }
      out.set(w ? [r / w, gg / w, b / w] : [0.7, 0.7, 0.7], i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(out, 3));
    mesh.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.0, skinning: true });
  }

  set(b, qWorld) {
    const bone = this.bones[b]; if (!bone) return;
    bone.parent.getWorldQuaternion(_q2);
    bone.quaternion.copy(_q2.invert().multiply(qWorld));
    bone.updateMatrixWorld(true);
  }

  swing(b, child, from, to) {  // rest orientation, swung so the bone points from `from` to `to`
    const rest = _v[3].copy(this.rp[child]).sub(this.rp[b]).normalize();
    this.set(b, _q.setFromUnitVectors(rest, _v[4].copy(to).sub(from).normalize()).multiply(this.rq[b]));
  }

  // J: 17 Vector3 in court metres. Returns false when the pose is unusable.
  pose(J) {
    const d = (a, b) => J[a].distanceTo(J[b]);
    const leg = (d(1, 2) + d(2, 3) + d(4, 5) + d(5, 6)) / 2, trunk = d(0, 8) + d(8, 10);
    const s = 0.5 * (leg / this.restLeg + trunk / this.restTrunk);
    if (!(s > 0.2 && s < 3)) return false;
    this.root.scale.setScalar(s); this.root.position.set(0, 0, 0);
    this.root.updateMatrixWorld(true);
    const hips = this.bones.Hips;
    hips.position.copy(hips.parent.worldToLocal(J[0].clone()));
    const hipL = new THREE.Vector3().subVectors(J[4], J[1]), shoL = new THREE.Vector3().subVectors(J[11], J[14]);
    const upLow = new THREE.Vector3().subVectors(J[7], J[0]), upAll = new THREE.Vector3().subVectors(J[8], J[0]), upTop = new THREE.Vector3().subVectors(J[8], J[7]);
    hipL.normalize(); shoL.normalize();
    const mix = w => hipL.clone().lerp(shoL, w).normalize();
    this.set('Hips', frameQuat(upLow, hipL, new THREE.Quaternion()).multiply(this.frames.Hips).multiply(this.rq.Hips));
    this.set('Spine', frameQuat(upLow, mix(0.33), new THREE.Quaternion()).multiply(this.frames.Spine).multiply(this.rq.Spine));
    this.set('Spine1', frameQuat(upAll, mix(0.66), new THREE.Quaternion()).multiply(this.frames.Spine1).multiply(this.rq.Spine1));
    this.set('Spine2', frameQuat(upTop, shoL, new THREE.Quaternion()).multiply(this.frames.Spine2).multiply(this.rq.Spine2));
    const headUp = new THREE.Vector3().subVectors(J[10], J[8]);
    this.swing('Neck', 'Head', J[8], J[10]);
    const nose = new THREE.Vector3().subVectors(J[9], J[8]); nose.addScaledVector(headUp.clone().normalize(), -nose.dot(headUp.clone().normalize()));
    if (nose.lengthSq() > 1e-6) {
      const left = new THREE.Vector3().crossVectors(headUp, nose);
      this.set('Head', frameQuat(headUp, left, new THREE.Quaternion()).multiply(this.frames.Head).multiply(this.rq.Head));
    }
    for (const [side, h, k, a, sh, el, wr] of [['Right', 1, 2, 3, 14, 15, 16], ['Left', 4, 5, 6, 11, 12, 13]]) {
      this.swing(`${side}UpLeg`, `${side}Leg`, J[h], J[k]);
      this.swing(`${side}Leg`, `${side}Foot`, J[k], J[a]);
      this.swing(`${side}Arm`, `${side}ForeArm`, J[sh], J[el]);
      this.swing(`${side}ForeArm`, `${side}Hand`, J[el], J[wr]);
      this.swing(`${side}Hand`, `${side}HandMiddle1`, J[el], J[wr]);
      // The foot points along the pelvis heading, pitched down more the higher the heel is lifted.
      const fwd = _v[5].copy(this.restFwd).applyQuaternion(frameQuat(upLow, hipL, new THREE.Quaternion()));
      fwd.z = 0; fwd.normalize();
      const pitch = Math.min(1.0, 0.18 + 3 * Math.max(0, J[a].z - 0.08));
      const target = fwd.multiplyScalar(Math.cos(pitch)); target.z = -Math.sin(pitch);
      this.swing(`${side}Foot`, `${side}ToeBase`, new THREE.Vector3(), target);
    }
    // The ankle bone sits at the rest ankle height above the floor.
    const fz = Math.min(this.bones.LeftFoot.getWorldPosition(_v[0]).z, this.bones.RightFoot.getWorldPosition(_v[1]).z);
    this.root.position.z = this.restAnkle * s - fz;
    this.root.updateMatrixWorld(true);
    return true;
  }
}

// Shaped capsule figure: tapered limbs, ellipsoid chest and pelvis, neck, head, hands and shoes.
class CapsuleFigure {
  constructor(col) {
    this.root = new THREE.Group();
    const lit = (c, rough = 0.8) => new THREE.MeshStandardMaterial({ color: c, roughness: rough });
    this.mat = { shirt: lit(col), shorts: lit(TONES.shorts), skin: lit(TONES.skin), shoe: lit(TONES.shoe) };
    this.limbs = []; this.blobs = {};
    const limb = (a, b, r0, r1, m) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, 14, 1), this.mat[m]);  // r0 at joint a, r1 at b
      this.root.add(mesh); this.limbs.push({ a, b, mesh, r0, r1 });
    };
    // [from, to, radius at from, radius at to (metres at 1.7 m stature), material]
    limb(1, 2, 0.085, 0.060, 'shorts'); limb(2, 3, 0.058, 0.040, 'skin'); limb(4, 5, 0.085, 0.060, 'shorts'); limb(5, 6, 0.058, 0.040, 'skin');
    limb(14, 15, 0.048, 0.038, 'shirt'); limb(15, 16, 0.037, 0.030, 'skin'); limb(11, 12, 0.048, 0.038, 'shirt'); limb(12, 13, 0.037, 0.030, 'skin');
    limb(8, 9, 0.052, 0.046, 'skin');
    const blob = (name, m, r) => { const o = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), this.mat[m]); o.userData.r = r; this.root.add(o); this.blobs[name] = o; };
    blob('chest', 'shirt', [0.17, 0.11, 0.24]); blob('belly', 'shirt', [0.145, 0.10, 0.2]); blob('pelvis', 'shorts', [0.15, 0.10, 0.11]);
    blob('head', 'skin', [0.085, 0.10, 0.115]); blob('hair', 'shorts', [0.088, 0.10, 0.08]);
    blob('lhand', 'skin', [0.04, 0.04, 0.05]); blob('rhand', 'skin', [0.04, 0.04, 0.05]);
    blob('lshoe', 'shoe', [0.055, 0.12, 0.045]); blob('rshoe', 'shoe', [0.055, 0.12, 0.045]);
    blob('lshoulder', 'shirt', [0.055, 0.055, 0.055]); blob('rshoulder', 'shirt', [0.055, 0.055, 0.055]);
    for (const j of [2, 5, 3, 6, 12, 15]) { const o = new THREE.Mesh(new THREE.SphereGeometry(1, 10, 8), this.mat.skin); o.userData.joint = j; this.root.add(o); this.blobs['j' + j] = o; }
  }

  place(o, centre, up, fwd, left, r, k) {  // ellipsoid with semi-axes r = [left, fwd, up] (metres at 1.7 m)
    o.position.copy(centre);
    o.quaternion.setFromRotationMatrix(_m.makeBasis(_v[5].copy(left).negate(), fwd, up));  // left is -x of the figure: keep the basis right-handed
    o.scale.set(r[0] * k, r[1] * k, r[2] * k);
  }

  pose(J) {
    const k = (J[10].z - 0.08) / 1.62;  // scale from this player's measured stature
    if (!(k > 0.3 && k < 2)) return false;
    const Y = new THREE.Vector3(0, 1, 0);
    for (const { a, b, mesh, r0, r1 } of this.limbs) {
      const dir = _v[0].subVectors(J[b], J[a]), len = dir.length();
      mesh.position.copy(J[a]).addScaledVector(dir, 0.5);
      mesh.quaternion.setFromUnitVectors(Y, dir.normalize());
      mesh.scale.set(k, len, k);
    }
    const up = new THREE.Vector3().subVectors(J[8], J[0]).normalize();
    const left = new THREE.Vector3().subVectors(J[11], J[14]).normalize();
    left.addScaledVector(up, -left.dot(up)).normalize();
    const fwd = new THREE.Vector3().crossVectors(left, up).normalize();
    const mid = (a, b, t) => new THREE.Vector3().lerpVectors(J[a], J[b], t);
    const B = this.blobs;
    this.place(B.chest, mid(7, 8, 0.65).addScaledVector(fwd, 0.0), up, fwd, left, B.chest.userData.r, k);
    this.place(B.belly, mid(0, 7, 0.55), up, fwd, left, B.belly.userData.r, k);
    this.place(B.pelvis, J[0].clone().addScaledVector(up, 0.02), up, fwd, left, B.pelvis.userData.r, k);
    const hup = new THREE.Vector3().subVectors(J[10], J[8]).normalize();
    const hfwd = new THREE.Vector3().subVectors(J[9], J[8]); hfwd.addScaledVector(hup, -hfwd.dot(hup));
    if (hfwd.lengthSq() < 1e-6) hfwd.copy(fwd); hfwd.normalize();
    const hleft = new THREE.Vector3().crossVectors(hup, hfwd).normalize();
    const hc = J[10].clone().addScaledVector(hup, -0.105 * k);
    this.place(B.head, hc, hup, hfwd, hleft, B.head.userData.r, k);
    this.place(B.hair, hc.clone().addScaledVector(hup, 0.025 * k).addScaledVector(hfwd, -0.012 * k), hup, hfwd, hleft, B.hair.userData.r, k);
    for (const [n, w, s] of [['lhand', 13, 12], ['rhand', 16, 15]]) {
      const d = new THREE.Vector3().subVectors(J[w], J[s]).normalize();
      this.place(B[n], J[w].clone().addScaledVector(d, 0.05 * k), up, fwd, left, B[n].userData.r, k);
    }
    for (const [n, a, kn] of [['lshoe', 6, 5], ['rshoe', 3, 2]]) {
      const f = fwd.clone(); f.z = 0; f.normalize();
      const c = J[a].clone().addScaledVector(f, 0.06 * k); c.z = Math.max(J[a].z - 0.03 * k, 0.025);
      this.place(B[n], c, UP, f, new THREE.Vector3().crossVectors(UP, f).normalize(), B[n].userData.r, k);
    }
    this.place(B.lshoulder, J[11], up, fwd, left, B.lshoulder.userData.r, k);
    this.place(B.rshoulder, J[14], up, fwd, left, B.rshoulder.userData.r, k);
    for (const j of [2, 5, 3, 6, 12, 15]) this.place(B['j' + j], J[j], up, fwd, left, [0.045, 0.045, 0.045], k);
    return true;
  }
}

// One player's figure: the rig once it has loaded, the capsule figure until then or if it never does.
class Figure {
  constructor(col) {
    this.group = new THREE.Group(); this.group.visible = false;
    this.col = col; this.capsule = new CapsuleFigure(col); this.group.add(this.capsule.root); this.rig = null;
    this.J = Array.from({ length: 17 }, () => new THREE.Vector3());
    loadRigBuffer(() => {
      if (FIG.state !== 'ready') return;
      new THREE.GLTFLoader().parse(FIG.buffer.slice(0), '', gltf => {
        this.rig = new RigFigure(gltf, this.col); this.group.add(this.rig.root); this.capsule.root.visible = false;
        View3D.dirtyFlag();
      }, err => console.warn('figure model failed to parse:', err));
    });
  }

  // Draw frame f of player P if its 3D pose is usable; returns whether something was drawn.
  show(P, f) {
    const b = P.body; if (!b || !P.bconf || !(P.bconf[f] > 0.25)) return this.group.visible = false;
    for (let i = 0; i < 17; i++) { const o = f * 51 + 3 * i; if (!Number.isFinite(b[o])) return this.group.visible = false; this.J[i].set(b[o], b[o + 1], b[o + 2]); }
    const fig = this.rig || this.capsule;
    this.group.visible = fig.pose(this.J);
    if (this.rig) this.capsule.root.visible = false;
    if (this.rig) this.rig.root.visible = true;
    return this.group.visible;
  }
}

const View3D = (() => {
  const host = document.getElementById('three');
  const hud = document.getElementById('three-hud');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (err) {
    // Without WebGL only this pane is lost; the video, 2D views and stats keep working.
    console.warn('3D view disabled:', err);
    hud.textContent = '3D view needs WebGL, which this browser has turned off.';
    return { buildCourt() {}, update() {}, preset() {} };
  }
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
  // The court and ball use unlit MeshBasicMaterial; these lights shade only the player figures.
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));
  scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x4a3b2a, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(3, -5, 9); scene.add(sun);

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
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.26, 24), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.28 }));
    disc.position.z = 0.01;
    g.add(body, disc);
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SKEL_EDGES.length * 6), 3));
    const skel = new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ color: col }));
    skel.frustumCulled = false;
    const fig = new Figure(col);
    dyn.add(g, skel, fig.group);
    players[p] = { g, body, skel, sg, fig, disc };
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
    preset(new URLSearchParams(location.search).get('view') || 'behind');  // ?view=behind|side|top, handy for screenshots
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
      const solid = have && V.opts.skel3d && o.fig.show(P, f);
      if (have && cam && V.opts.skel3d && !solid && !P.body && P.kp && P.status[f] === 2) {  // no bodies.parquet: the flat keypoint lift
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
      if (!V.opts.skel3d || !have) o.fig.group.visible = false;
      o.skel.visible = skel;
      o.body.visible = have && !skel && !solid;
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
  // look(pos, target) and figureKinds() serve screenshot tooling: aim the camera, and say whether each player is drawn by the rig.
  return { buildCourt, update, preset, dirtyFlag() { dirty = true; },
    look(pos, target) { camera.position.set(...pos); controls.target.set(...target); controls.update(); dirty = true; },
    figureKinds() { return ['A', 'B'].map(p => players[p].fig.rig ? 'rig' : 'capsule'); } };
})();
