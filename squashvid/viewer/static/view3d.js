// Three.js court. World frame = court frame (x across, y back->front wall, z up); camera.up is +z.
'use strict';
// ---- Solid 3D figures driven by the lifted joints (bodies.parquet; H36M-17 order, court metres) ----------------------
// A rigged child (Rocketbox Male_Child_02, MIT; built by squashvid/viewer/figure_model.py) is posed bone by bone from
// the joints and painted in its player's colours as measured from the video; if it cannot load, a shaped capsule
// figure with the same interface is used.
const RIG_URL = '/static/vendor/models/kid.glb';
const MASK_URL = { kid_body: '/static/vendor/models/kid_mask_body.png', kid_head: '/static/vendor/models/kid_mask_head.png' };
const FIG = { buffer: null, masks: {}, state: 'idle', waiters: [] };  // idle | loading | ready | failed
const loadImage = url => new Promise((ok, no) => { const im = new Image(); im.onload = () => ok(im); im.onerror = no; im.src = url; });
function loadRigBuffer(cb) {
  if (FIG.state === 'ready' || FIG.state === 'failed') return cb();
  FIG.waiters.push(cb);
  if (FIG.state === 'loading') return;
  FIG.state = 'loading';
  const done = (ok) => { FIG.state = ok ? 'ready' : 'failed'; FIG.waiters.splice(0).forEach(w => w()); };
  const s = document.createElement('script');
  s.src = '/static/vendor/GLTFLoader.js';  // classic-script r128 build; loaded here so index.html stays unchanged
  s.onload = () => Promise.all([fetch(RIG_URL).then(r => r.arrayBuffer()),
    ...Object.entries(MASK_URL).map(([k, u]) => loadImage(u).then(im => { FIG.masks[k] = im; }).catch(() => console.warn('figure mask failed to load:', u)))])
    .then(([b]) => { FIG.buffer = b; done(true); }).catch(e => { console.warn('figure model failed to load:', e); done(false); });
  s.onerror = () => { console.warn('GLTFLoader failed to load'); done(false); };
  document.head.appendChild(s);
}

// Mask labels (figure_model.py): grey level = LABEL_STEP x part.
const LABEL_STEP = 60, PARTS = ['', 'shirt', 'shorts', 'skin'];

// The texture repainted: pixels the mask labels with a part take that part's colour, keeping the texture's light and
// shade (each pixel's brightness against the part's mean brightness). Parts without a colour keep the model's own.
function paintTexture(img, mask, look) {
  const w = img.width, h = img.height, cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d');
  g.drawImage(img, 0, 0);
  if (!mask) return cv;
  const px = g.getImageData(0, 0, w, h), d = px.data;
  const mc = document.createElement('canvas'); mc.width = w; mc.height = h;
  const mg = mc.getContext('2d'); mg.imageSmoothingEnabled = false; mg.drawImage(mask, 0, 0, w, h);
  const m = mg.getImageData(0, 0, w, h).data;
  const cols = PARTS.map(p => p && look[p]), sum = new Float64Array(PARTS.length), cnt = new Float64Array(PARTS.length);
  const lum = i => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  for (let i = 0; i < d.length; i += 4) { const l = Math.round(m[i] / LABEL_STEP); if (cols[l]) { sum[l] += lum(i); cnt[l]++; } }
  for (let i = 0; i < d.length; i += 4) {
    const l = Math.round(m[i] / LABEL_STEP), c = cols[l];
    if (!c) continue;
    const k = lum(i) / Math.max(1, sum[l] / cnt[l]);
    d[i] = Math.min(255, c[0] * k); d[i + 1] = Math.min(255, c[1] * k); d[i + 2] = Math.min(255, c[2] * k);
  }
  g.putImageData(px, 0, 0);
  return cv;
}

// A junior squash racket (0.62 m), along +y from the butt, strings in the xy plane.
function makeRacket() {
  const g = new THREE.Group();
  const frame = new THREE.MeshStandardMaterial({ color: 0x2c3340, roughness: 0.45, metalness: 0.25 });
  const grip = new THREE.MeshStandardMaterial({ color: 0x15161a, roughness: 0.9 });
  const part = (geo, mat, y) => { const o = new THREE.Mesh(geo, mat); o.position.y = y; o.castShadow = true; g.add(o); return o; };
  part(new THREE.CylinderGeometry(0.0135, 0.0155, 0.22, 10), grip, 0.11);
  part(new THREE.CylinderGeometry(0.0065, 0.0085, 0.15, 8), frame, 0.295);
  const head = part(new THREE.TorusGeometry(0.098, 0.0065, 8, 48), frame, 0.49);
  head.scale.set(1, 1.3, 1);
  const strings = part(new THREE.PlaneGeometry(0.19, 0.25), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }), 0.49);
  strings.castShadow = false;
  return g;
}
const RACKET_COCK = 1.0;  // radians the racket turns from the forearm line toward the thumb (a cocked wrist)

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

const TONES = { skin: new THREE.Color('#c8946f'), shorts: new THREE.Color('#252a36'), shoe: new THREE.Color('#e9e9ee') };

class RigFigure {
  constructor(gltf, col) {
    this.root = new THREE.Group();
    this.root.rotation.x = Math.PI / 2;  // the model is y-up, the court z-up
    this.model = gltf.scene;
    this.root.add(this.model);
    this.bones = {};
    this.model.traverse(o => { if (o.isBone) this.bones[o.name.replace(/^mixamorig:?/, '')] = o; });
    this.meshes = [];
    this.model.traverse(o => {
      if (!o.isMesh) return;
      o.frustumCulled = false; o.castShadow = true;
      if (o.material.map) this.meshes.push({ mesh: o, image: o.material.map.image, mask: FIG.masks[o.material.name] });
    });
    this.paint({ shirt: [col.r * 255, col.g * 255, col.b * 255] });
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
    this.racket = makeRacket();
    this.hold('Right');
  }

  // Repaint the clothes and skin: look = {shirt, shorts, skin: [r, g, b] sRGB 0-255}, any may be missing.
  paint(look) {
    for (const { mesh, image, mask } of this.meshes) {
      const tex = new THREE.CanvasTexture(paintTexture(image, mask, look));
      tex.flipY = false;  // glTF texture convention
      if (mesh.material.map) mesh.material.map.dispose();
      mesh.material.map = tex; mesh.material.needsUpdate = true;
    }
  }

  // Put the racket in this hand ('Right' | 'Left'): the handle in the palm, turned RACKET_COCK from the forearm line
  // toward the thumb, the strings facing the way the palm does.
  hold(side) {
    const hand = this.bones[`${side}Hand`], mid = this.rp[`${side}HandMiddle1`];
    const thumbName = Object.keys(this.bones).find(n => new RegExp(`${side[0]}_?Finger0$`).test(n));
    if (!hand || !mid || !thumbName) return;
    const inv = this.rq[`${side}Hand`].clone().invert(), at = this.rp[`${side}Hand`];
    const d = mid.clone().sub(at).applyQuaternion(inv).normalize();
    const t = this.rp[thumbName].clone().sub(at).applyQuaternion(inv);
    t.addScaledVector(d, -t.dot(d)).normalize();
    const y = d.clone().multiplyScalar(Math.cos(RACKET_COCK)).addScaledVector(t, Math.sin(RACKET_COCK)).normalize();
    const z = new THREE.Vector3().crossVectors(d, t).normalize(), x = new THREE.Vector3().crossVectors(y, z);
    const k = 1 / hand.getWorldScale(new THREE.Vector3()).x;  // racket metres at rest, whatever the bone's scale
    this.racket.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
    this.racket.scale.setScalar(k);
    this.racket.position.copy(d).multiplyScalar(0.06 * k).addScaledVector(y, -0.09 * k);
    hand.add(this.racket);
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
        if (this.look) this.setLook(this.look, this.hand);
        View3D.dirtyFlag();
      }, err => console.warn('figure model failed to parse:', err));
    });
  }

  // The player's measured colours and racket hand (export meta.players), applied now or once the rig loads.
  setLook(look, hand) {
    this.look = look; this.hand = hand;
    // Skin is not measured (too small and shaded in the video): every figure gets the capsule figure's neutral tone.
    const rgb = c => [c.r * 255, c.g * 255, c.b * 255];
    if (this.rig) { this.rig.paint({ shirt: rgb(this.col), ...look, skin: rgb(TONES.skin) }); this.rig.hold(hand === 'left' ? 'Left' : 'Right'); }
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
    return { buildCourt() {}, update() {}, ball() {}, preset() {} };
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
  // The figures and racket cast real shadows onto the floor (a shadow-only plane over the court floor).
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  sun.target.position.set(3.2, 4.9, 0); sun.position.set(6.2, -0.1, 9); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0005;
  Object.assign(sun.shadow.camera, { left: -7, right: 7, top: 7, bottom: -7, near: 1, far: 25 });
  scene.add(sun, sun.target);

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
  const ballMesh = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd54a, transparent: true }));
  // The ball's shadow straight below it on the floor, fainter the higher the ball: it is what makes the height readable.
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.075, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false }));
  const ghost = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.16, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
  const rayGeo = new THREE.BufferGeometry(); rayGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const rayLine = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.45 }));
  rayLine.frustumCulled = false;
  const TRAIL = 128, TRAIL_S = 0.8;  // points held; seconds of path behind the ball
  // The trail is a tube that tapers from the ball's size to nothing (WebGL lines are one pixel wide): SIDES vertices
  // around each path point, rebuilt in place each draw.
  const SIDES = 6, BALL_R = 0.07;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * SIDES * 3), 3));
  trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL * SIDES * 3), 3));
  const tubeIdx = [];
  for (let i = 0; i < TRAIL - 1; i++) for (let j = 0; j < SIDES; j++) {
    const a = i * SIDES + j, b = i * SIDES + (j + 1) % SIDES, c = a + SIDES, d = b + SIDES;
    tubeIdx.push(a, c, b, b, c, d);
  }
  trailGeo.setIndex(tubeIdx);
  const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
  trail.frustumCulled = false;
  dyn.add(ballMesh, shadow, ghost, rayLine, trail);
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
    const shade = quad([0, 0, 0.003], [W, 0, 0.003], [W, L, 0.003], [0, L, 0.003], new THREE.ShadowMaterial({ opacity: 0.32, side: THREE.DoubleSide }));
    shade.geometry.computeVertexNormals();  // the shadow lookup offsets along the normal; without one it is NaN
    shade.receiveShadow = true; g.add(shade);
    courtGroup = g; scene.add(g);
    const pl = D.meta.players || {};
    for (const p of ['A', 'B']) players[p].fig.setLook((pl.look || {})[p] || {}, (pl.racket_hand || {})[p] || 'right');
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
    if (!D.ball_path_xyz) legacyBall(D, f, cam);  // with a ball path, ball(ft) draws it, between frames too
    // recent wall/floor impacts
    const t = D.t[f]; let k = 0;
    for (const e of D.bounces) {
      if (e.t > t || e.t < t - 1.5 || !Number.isFinite(e.x) || !Number.isFinite(e.z) || !Number.isFinite(e.y)) continue;
      if (k >= impacts.length) break;
      const m = impacts[k++]; m.visible = true; m.position.set(e.x, e.y, e.z);
      m.material.opacity = 1 - (t - e.t) / 1.5;
    }
    for (; k < impacts.length; k++) impacts[k].visible = false;
    dirty = true;
  }

  const _t = new THREE.Vector3(), _n = new THREE.Vector3(), _b = new THREE.Vector3(), _X = new THREE.Vector3(1, 0, 0);
  function setTrail(pts, color) {
    const tp = trailGeo.attributes.position.array, tc = trailGeo.attributes.color.array, n = Math.min(pts.length, TRAIL);
    const P = pts.slice(pts.length - n);
    for (let i = 0; i < n; i++) {
      const a = n > 1 ? i / (n - 1) : 1, r = BALL_R * 0.9 * a, col = color(a);
      const p = P[Math.min(n - 1, i + 1)], q = P[Math.max(0, i - 1)];
      _t.set(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
      if (_t.lengthSq() < 1e-12) _t.set(0, 1, 0);
      _t.normalize();
      _n.crossVectors(_t, Math.abs(_t.z) > 0.9 ? _X : UP).normalize(); _b.crossVectors(_t, _n);
      for (let j = 0; j < SIDES; j++) {
        const th = (j / SIDES) * Math.PI * 2, c = Math.cos(th) * r, s = Math.sin(th) * r, o = (i * SIDES + j) * 3;
        tp[o] = P[i][0] + c * _n.x + s * _b.x; tp[o + 1] = P[i][1] + c * _n.y + s * _b.y; tp[o + 2] = P[i][2] + c * _n.z + s * _b.z;
        tc[o] = col[0]; tc[o + 1] = col[1]; tc[o + 2] = col[2];
      }
    }
    trailGeo.setDrawRange(0, Math.max(0, n - 1) * SIDES * 6);
    trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
  }

  // Without a ball path: the fitted 3D point, else the floor projection of the camera ray.
  function legacyBall(D, f, cam) {
    shadow.visible = false; ballMesh.material.opacity = 1;
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
    const pts = [];
    for (let g = Math.max(0, f - 23); g <= f; g++) if (D.ball_xyz && Number.isFinite(D.ball_xyz[3 * g])) pts.push([D.ball_xyz[3 * g], D.ball_xyz[3 * g + 1], D.ball_xyz[3 * g + 2]]);
    setTrail(pts, a => [0.15 + 0.85 * a, 0.14 + 0.72 * a, 0.05 + 0.2 * a]);
    hud.textContent = mode;
  }

  // The ball on the 3D path at fractional frame ft, with its shadow and the path over the last TRAIL_S behind it.
  function ball(ft) {
    const D = V.D; if (!D || !D.ball_path_xyz) return;
    const tr = ballTrail(D, ft, TRAIL_S, true), b = tr[tr.length - 1];
    ghost.visible = false; rayLine.visible = false;
    ballMesh.visible = shadow.visible = !!b;
    if (b) {
      ballMesh.position.set(b.p[0], b.p[1], b.p[2]);
      ballMesh.material.opacity = b.k === 2 ? 0.75 : 1;
      shadow.position.set(b.p[0], b.p[1], 0.012);
      shadow.material.opacity = 0.45 * Math.max(0.3, 1 - b.p[2] / 5);
    }
    setTrail(tr.map(q => q.p), a => [0.55 + 0.45 * a, 0.45 + 0.4 * a, 0.1 + 0.2 * a]);
    const uvOk = D.ball_uv && Number.isFinite(D.ball_uv[2 * Math.floor(ft)]);
    hud.textContent = b ? (b.k === 2 ? 'ball: 3D, filled between fitted points' : 'ball: 3D') : uvOk ? 'ball seen, no 3D position' : 'no ball';
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
  return { buildCourt, update, ball, preset, dirtyFlag() { dirty = true; },
    look(pos, target) { camera.position.set(...pos); controls.target.set(...target); controls.update(); dirty = true; },
    figureKinds() { return ['A', 'B'].map(p => players[p].fig.rig ? 'rig' : 'capsule'); } };
})();
