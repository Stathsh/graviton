// Graviton — Black Hole Astrophysics Simulator
// Zig/WASM + WebGPU Compute + TypeScript

import { loadKerrWasm, type KerrPhysics } from './wasm-bridge';
import { initWebGPU, type GPUContext } from './gpu';
import * as m from './math';

// @ts-ignore
import computeShader from './shaders/particles-compute.wgsl';
// @ts-ignore
import renderShader from './shaders/particles-render.wgsl';
// @ts-ignore
import lensingShader from './shaders/lensing.wgsl';
// @ts-ignore
import compositeShader from './shaders/composite.wgsl';
// @ts-ignore
import geodesicShader from './shaders/geodesic-render.wgsl';

const PARTICLE_COUNT = 300_000;
const PARTICLE_SIZE = 64;
const GEODESIC_POINTS = 400;
const NUM_GEODESICS = 8;
const STAR_PARTICLE_START = 250_000; // last 50K particles reserved for star
const STAR_PARTICLE_COUNT = 50_000;

interface Controls {
  mass: number; spin: number; accretion: number; temperature: number;
  camDist: number; camAngle: number; colorScheme: number;
  showTimeDilation: boolean; showGeodesics: boolean;
  bloomIntensity: number; jetPower: number;
}

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const noWebGPU = document.getElementById('no-webgpu') as HTMLElement;

  let gpu: GPUContext;
  let physics: KerrPhysics;
  try {
    [gpu, physics] = await Promise.all([initWebGPU(canvas), loadKerrWasm()]);
  } catch (e) {
    noWebGPU.style.display = 'flex';
    console.error(e);
    return;
  }

  const { device, context, format } = gpu;
  const hdrFormat: GPUTextureFormat = 'rgba16float';

  const resize = () => {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    context.configure({ device, format, alphaMode: 'opaque' });
  };
  resize();
  window.addEventListener('resize', resize);

  const controls: Controls = {
    mass: 3.0, spin: 0.5, accretion: 1.0, temperature: 6000,
    camDist: 20, camAngle: 30, colorScheme: 0,
    showTimeDilation: false, showGeodesics: false,
    bloomIntensity: 0.8, jetPower: 1.0,
  };

  // --- Bind UI ---
  function bind(id: string, key: keyof Controls, suffix = '') {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    const disp = document.getElementById(id + 'Val');
    el.addEventListener('input', () => {
      (controls as Record<string, number>)[key] = parseFloat(el.value);
      if (disp) disp.textContent = el.value + suffix;
    });
  }
  bind('mass', 'mass'); bind('spin', 'spin'); bind('accretion', 'accretion');
  bind('temperature', 'temperature', 'K'); bind('camDist', 'camDist');
  bind('camAngle', 'camAngle', '\u00B0'); bind('bloomIntensity', 'bloomIntensity');
  bind('jetPower', 'jetPower');

  const csEl = document.getElementById('colorScheme') as HTMLSelectElement | null;
  csEl?.addEventListener('change', () => { controls.colorScheme = parseInt(csEl.value); });
  const tdEl = document.getElementById('timeDilation') as HTMLInputElement | null;
  tdEl?.addEventListener('change', () => { controls.showTimeDilation = tdEl.checked; });
  const geoEl = document.getElementById('geodesics') as HTMLInputElement | null;
  geoEl?.addEventListener('change', () => { controls.showGeodesics = geoEl.checked; });

  // --- Camera state ---
  let orbitH = 0;           // horizontal orbit angle
  let orbitV = controls.camAngle; // vertical angle (degrees)
  let panX = 0, panY = 0;  // pan offset (look-at target offset)
  let autoRot = true;

  // --- Mouse controls ---
  // Left click + drag = orbit
  // Right click + drag = pan
  // Scroll = zoom
  // Double-click = reset auto-rotate
  let leftDrag = false, rightDrag = false;
  let prevMX = 0, prevMY = 0;

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  canvas.addEventListener('mousedown', (e) => {
    prevMX = e.clientX; prevMY = e.clientY;
    if (e.button === 0) { leftDrag = true; autoRot = false; canvas.style.cursor = 'grabbing'; }
    if (e.button === 2) { rightDrag = true; canvas.style.cursor = 'move'; }
  });

  canvas.addEventListener('mousemove', (e) => {
    const dx = e.clientX - prevMX;
    const dy = e.clientY - prevMY;
    prevMX = e.clientX; prevMY = e.clientY;

    if (leftDrag) {
      orbitH += dx * 0.005;
      orbitV = Math.max(5, Math.min(85, orbitV - dy * 0.3));
      controls.camAngle = orbitV;
      const ad = document.getElementById('camAngleVal');
      const as2 = document.getElementById('camAngle') as HTMLInputElement | null;
      if (ad) ad.textContent = Math.round(orbitV) + '\u00B0';
      if (as2) as2.value = String(Math.round(orbitV));
    }

    if (rightDrag) {
      // Pan: move the look-at target
      const panSpeed = controls.camDist * 0.002;
      const ca = orbitH + (autoRot ? performance.now() / 1000 * 0.04 : 0);
      // Pan relative to camera orientation
      panX += (-dx * Math.cos(ca) - dy * Math.sin(ca) * 0.3) * panSpeed;
      panY += dy * panSpeed * 0.5;
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (e.button === 0) leftDrag = false;
    if (e.button === 2) rightDrag = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('mouseleave', () => {
    leftDrag = false; rightDrag = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    controls.camDist = Math.max(5, Math.min(80, controls.camDist * (1 + e.deltaY * 0.001)));
    const dd = document.getElementById('camDistVal');
    const ds = document.getElementById('camDist') as HTMLInputElement | null;
    if (dd) dd.textContent = controls.camDist.toFixed(1);
    if (ds) ds.value = controls.camDist.toFixed(1);
  }, { passive: false });

  canvas.addEventListener('dblclick', () => {
    autoRot = true;
    panX = 0; panY = 0; // reset pan
  });

  // Touch
  let touchDist = 0;
  canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) { leftDrag = true; autoRot = false; prevMX = e.touches[0].clientX; prevMY = e.touches[0].clientY; }
    else if (e.touches.length === 2) { touchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }
  });
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && leftDrag) {
      orbitH += (e.touches[0].clientX - prevMX) * 0.005;
      orbitV = Math.max(5, Math.min(85, orbitV - (e.touches[0].clientY - prevMY) * 0.3));
      controls.camAngle = orbitV; prevMX = e.touches[0].clientX; prevMY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
      controls.camDist = Math.max(5, Math.min(80, controls.camDist * (1 - (d - touchDist) * 0.003)));
      touchDist = d;
    }
  }, { passive: false });
  canvas.addEventListener('touchend', () => { leftDrag = false; });
  canvas.style.cursor = 'grab';

  // Screenshot
  document.getElementById('screenshot')?.addEventListener('click', () => {
    canvas.toBlob((b) => {
      if (!b) return;
      const u = URL.createObjectURL(b);
      const a = document.createElement('a');
      a.href = u; a.download = `graviton-${Date.now()}.png`; a.click();
      URL.revokeObjectURL(u);
    });
  });

  // --- Tidal Disruption Event ---
  // star_trigger: 0 = no star, >0 = time when star was launched
  let starTriggerTime = 0;
  let starActive = false;

  function launchStar() {
    starActive = true;
    starTriggerTime = performance.now() / 1000;

    // Spawn a star: cluster of particles in a sphere at distance ~30M
    const starData = new Float32Array(STAR_PARTICLE_COUNT * 16);
    const starRadius = 1.5; // radius of the star
    const starDist = 35 * controls.mass; // starting distance
    const startAngle = orbitH + Math.PI * 0.7; // come from side

    for (let i = 0; i < STAR_PARTICLE_COUNT; i++) {
      const o = i * 16;
      // Random position in a sphere
      const theta = Math.acos(2 * Math.random() - 1);
      const phi = Math.random() * Math.PI * 2;
      const r = starRadius * Math.cbrt(Math.random()); // uniform volume distribution
      const lx = r * Math.sin(theta) * Math.cos(phi);
      const ly = r * Math.sin(theta) * Math.sin(phi);
      const lz = r * Math.cos(theta);

      // Position the star at distance from BH
      starData[o + 0] = Math.cos(startAngle) * starDist + lx;
      starData[o + 1] = ly * 0.5; // slightly flattened
      starData[o + 2] = Math.sin(startAngle) * starDist + lz;
      starData[o + 3] = 0; // age

      // Initial velocity: tangential + slight inward
      const orbV = Math.sqrt(controls.mass / starDist) * 0.6;
      starData[o + 4] = -Math.sin(startAngle) * orbV - Math.cos(startAngle) * 0.03;
      starData[o + 5] = 0;
      starData[o + 6] = Math.cos(startAngle) * orbV - Math.sin(startAngle) * 0.03;
      starData[o + 7] = 5500 + (Math.random() - 0.5) * 1000; // solar-like temperature

      // meta
      starData[o + 8] = starDist; // r
      starData[o + 9] = 1;  // redshift
      starData[o + 10] = 1; // doppler
      starData[o + 11] = 1; // alpha

      // extra: type=2 (star), time_dilation, initial_r, turbulence
      starData[o + 12] = 2.0; // type = star
      starData[o + 13] = 1.0;
      starData[o + 14] = r / starRadius; // normalized distance from star center
      starData[o + 15] = Math.random();
    }

    // Write star particles at offset
    device.queue.writeBuffer(particleBuf, STAR_PARTICLE_START * PARTICLE_SIZE, starData);

    // Update button state
    const btn = document.getElementById('launchStar');
    if (btn) {
      btn.textContent = 'Star Launched!';
      btn.setAttribute('disabled', 'true');
      setTimeout(() => {
        btn.textContent = 'Launch Star';
        btn.removeAttribute('disabled');
      }, 8000);
    }
  }

  document.getElementById('launchStar')?.addEventListener('click', launchStar);

  // --- WASM physics ---
  physics.setParams(controls.mass, controls.spin);

  // --- GPU Buffers ---
  const particleBuf = device.createBuffer({ size: PARTICLE_COUNT * PARTICLE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const computeParamsBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lensingUBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const renderUBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const compositeUBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const geodesicUBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const geodesicBuf = device.createBuffer({ size: NUM_GEODESICS * GEODESIC_POINTS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // Init disk + jet particles (first 250K)
  const isco = physics.computeISCO();
  const init = new Float32Array(PARTICLE_COUNT * 16);
  const diskEnd = STAR_PARTICLE_START;
  const jetStart = Math.floor(diskEnd * 7 / 8);
  for (let i = 0; i < diskEnd; i++) {
    const o = i * 16;
    const isJet = i >= jetStart;
    const ang = Math.random() * Math.PI * 2;
    if (isJet) {
      const ps = Math.random() > 0.5 ? 1 : -1;
      const sp = Math.random() * 0.2;
      init[o] = Math.cos(ang) * sp * 2; init[o+1] = ps * (0.5 + Math.random() * 10); init[o+2] = Math.sin(ang) * sp * 2;
      init[o+3] = Math.random() * 30; init[o+5] = ps * (0.3 + Math.random() * 0.5);
      init[o+7] = 50000 + Math.random() * 100000; init[o+12] = 1.0;
    } else {
      const rad = isco + Math.pow(Math.random(), 0.5) * 20;
      const h = (Math.random() - 0.5) * 0.2;
      const vo = Math.sqrt(controls.mass / rad);
      init[o] = Math.cos(ang) * rad; init[o+1] = h; init[o+2] = Math.sin(ang) * rad;
      init[o+3] = Math.random() * 40; init[o+4] = -Math.sin(ang) * vo; init[o+6] = Math.cos(ang) * vo;
      init[o+7] = 3000 + Math.random() * 17000; init[o+12] = 0;
    }
    init[o+9] = 1; init[o+10] = 1; init[o+11] = 0.5; init[o+13] = 1; init[o+15] = Math.random();
  }
  // Star particles start inactive (alpha=0, far away)
  for (let i = STAR_PARTICLE_START; i < PARTICLE_COUNT; i++) {
    const o = i * 16;
    init[o] = 200; init[o+1] = 200; init[o+2] = 200; // far away
    init[o+3] = 999; // very old = invisible
    init[o+12] = 2.0; // type = star
    init[o+11] = 0; // alpha = 0
  }
  device.queue.writeBuffer(particleBuf, 0, init);

  // --- Pipelines ---
  const computeMod = device.createShaderModule({ code: computeShader });
  const computePipe = device.createComputePipeline({ layout: 'auto', compute: { module: computeMod, entryPoint: 'main' } });
  const computeBG = device.createBindGroup({ layout: computePipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: computeParamsBuf } }, { binding: 1, resource: { buffer: particleBuf } }] });

  const lensingMod = device.createShaderModule({ code: lensingShader });
  const lensingPipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: lensingMod, entryPoint: 'vs_main' },
    fragment: { module: lensingMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'always' },
  });
  const lensingBG = device.createBindGroup({ layout: lensingPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: lensingUBuf } }] });

  const renderMod = device.createShaderModule({ code: renderShader });
  const renderPipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: renderMod, entryPoint: 'vs_main' },
    fragment: { module: renderMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat,
      blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
               alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });
  const renderBG = device.createBindGroup({ layout: renderPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: renderUBuf } }, { binding: 1, resource: { buffer: particleBuf } }] });

  const geoMod = device.createShaderModule({ code: geodesicShader });
  const geoPipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: geoMod, entryPoint: 'vs_main' },
    fragment: { module: geoMod, entryPoint: 'fs_main', targets: [{ format: hdrFormat,
      blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
               alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
    primitive: { topology: 'line-strip' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });
  const geoBG = device.createBindGroup({ layout: geoPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: geodesicUBuf } }, { binding: 1, resource: { buffer: geodesicBuf } }] });

  const compositeMod = device.createShaderModule({ code: compositeShader });
  const compositeSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const compositePipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: compositeMod, entryPoint: 'vs_main' },
    fragment: { module: compositeMod, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  // --- Offscreen textures ---
  function makeTextures(w: number, h: number) {
    return {
      hdr: device.createTexture({ size: [w, h], format: hdrFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }),
      depth: device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT }),
    };
  }
  let tex = makeTextures(canvas.width, canvas.height);
  let tw = canvas.width, th = canvas.height;

  // Geodesic data
  let lastGeoUpdate = 0;
  const geoData = new Float32Array(NUM_GEODESICS * GEODESIC_POINTS * 4);
  function traceGeodesics(camR: number, camTheta: number) {
    const impacts = [-6, -4, -2.5, -1.5, 1.5, 2.5, 4, 6];
    for (let g = 0; g < NUM_GEODESICS; g++) {
      const b = impacts[g] * controls.mass;
      const bo = g * GEODESIC_POINTS * 4;
      let px = camR * Math.sin(camTheta), py = camR * Math.cos(camTheta) * 0.1, pz = 0;
      let dx = -Math.sin(camTheta), dy = -Math.cos(camTheta) * 0.1, dz = b / camR;
      const dl = Math.sqrt(dx*dx+dy*dy+dz*dz); dx/=dl; dy/=dl; dz/=dl;
      const rs = 2 * controls.mass;
      for (let i = 0; i < GEODESIC_POINTS; i++) {
        const oi = bo + i * 4; geoData[oi] = px; geoData[oi+1] = py; geoData[oi+2] = pz;
        const r = Math.sqrt(px*px+py*py+pz*pz);
        geoData[oi+3] = Math.max(0, 1-i/GEODESIC_POINTS) * Math.max(0, 1-rs/r) * 0.5;
        if (r < rs*0.6 || r > 80) break;
        const def = rs/(r*r)*1.2;
        dx += -px/r*def*0.3; dy += -py/r*def*0.3; dz += -pz/r*def*0.3;
        const nl = Math.sqrt(dx*dx+dy*dy+dz*dz); dx/=nl; dy/=nl; dz/=nl;
        const st = r < rs*3 ? 0.15 : 0.4; px+=dx*st; py+=dy*st; pz+=dz*st;
      }
    }
    device.queue.writeBuffer(geodesicBuf, 0, geoData);
  }

  // --- Frame loop ---
  let lt = performance.now(), fc = 0, fa = 0;

  function frame(now: number) {
    const dt = Math.min((now - lt) / 1000, 0.05); lt = now; fc++; fa += dt;
    if (fa >= 0.5) {
      document.getElementById('fps')!.textContent = `${Math.round(fc/fa)} FPS`;
      document.getElementById('particles')!.textContent = `${(PARTICLE_COUNT/1000).toFixed(0)}K particles`;
      fc = 0; fa = 0;
    }

    physics.setParams(controls.mass, controls.spin);
    const ci = physics.computeISCO();
    const cpSphere = physics.computePhotonSphere();
    const rh = controls.mass + Math.sqrt(Math.max(controls.mass*controls.mass - controls.spin*controls.spin, 0));
    const ht = 1.227e23 / (8 * Math.PI * controls.mass);

    const info = document.getElementById('physics-info');
    if (info) info.innerHTML = [
      `<div>Event Horizon <b>${rh.toFixed(2)}M</b></div>`,
      `<div>Photon Sphere <b>${cpSphere.toFixed(2)}M</b></div>`,
      `<div>ISCO <b>${ci.toFixed(2)}M</b></div>`,
      `<div>Schwarzschild <b>${(2*controls.mass).toFixed(2)}M</b></div>`,
      `<div>Hawking Temp <b>${ht.toExponential(1)}K</b></div>`,
      `<div>Ergosphere <b>${(2*controls.mass).toFixed(2)}M</b></div>`,
    ].join('');

    if (canvas.width !== tw || canvas.height !== th) {
      tex.hdr.destroy(); tex.depth.destroy();
      tex = makeTextures(canvas.width, canvas.height); tw = canvas.width; th = canvas.height;
    }

    const time = now / 1000;
    const oa = (autoRot ? time * 0.04 : 0) + orbitH;
    const ca = controls.camAngle * Math.PI / 180;

    // Camera with pan offset
    const target: m.Vec3 = [panX, panY, 0];
    const camPos: m.Vec3 = [
      target[0] + Math.cos(oa) * Math.cos(ca) * controls.camDist,
      target[1] + Math.sin(ca) * controls.camDist,
      target[2] + Math.sin(oa) * Math.cos(ca) * controls.camDist,
    ];
    const view = m.lookAt(camPos, target, [0, 1, 0]);
    const proj = m.perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 300);
    const vp = m.multiply(proj, view);
    const ivp = m.invert(vp);

    if (controls.showGeodesics && now - lastGeoUpdate > 200) { traceGeodesics(controls.camDist, ca); lastGeoUpdate = now; }

    // Star trigger time for compute shader
    const starTime = starActive ? time - starTriggerTime : -1;

    // Uniforms — compute params expanded to 48 bytes (12 floats)
    device.queue.writeBuffer(computeParamsBuf, 0, new Float32Array([
      controls.mass, controls.spin, dt * 0.4, controls.accretion,
      time, PARTICLE_COUNT, ci, cpSphere,
      starTime, STAR_PARTICLE_START, STAR_PARTICLE_COUNT, 0,
    ]));

    const lu = new Float32Array(32);
    lu.set(ivp, 0); lu.set([camPos[0], camPos[1], camPos[2], 1], 16);
    lu.set([controls.mass, controls.spin, time, controls.colorScheme], 20);
    lu.set([rh, cpSphere, ci, 0], 24);
    device.queue.writeBuffer(lensingUBuf, 0, lu);

    const ru = new Float32Array(32);
    ru.set(vp, 0); ru.set([camPos[0], camPos[1], camPos[2], 1], 16);
    ru.set([controls.mass, time, controls.colorScheme, ci], 20);
    ru.set([controls.showTimeDilation ? 1 : 0, 0, 0, 0], 24);
    device.queue.writeBuffer(renderUBuf, 0, ru);

    device.queue.writeBuffer(compositeUBuf, 0, new Float32Array([1.3, 0.45, controls.bloomIntensity, time]));

    if (controls.showGeodesics) {
      const gu = new Float32Array(20); gu.set(vp, 0); gu.set([0.3, 0.8, 1.0, 0.5], 16);
      device.queue.writeBuffer(geodesicUBuf, 0, gu);
    }

    const enc = device.createCommandEncoder();

    const cp3 = enc.beginComputePass();
    cp3.setPipeline(computePipe); cp3.setBindGroup(0, computeBG);
    cp3.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256)); cp3.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: tex.hdr.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: tex.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    rp.setPipeline(lensingPipe); rp.setBindGroup(0, lensingBG); rp.draw(3);
    rp.setPipeline(renderPipe); rp.setBindGroup(0, renderBG); rp.draw(6, PARTICLE_COUNT);
    if (controls.showGeodesics) {
      rp.setPipeline(geoPipe); rp.setBindGroup(0, geoBG);
      for (let g = 0; g < NUM_GEODESICS; g++) rp.draw(GEODESIC_POINTS, 1, g * GEODESIC_POINTS, 0);
    }
    rp.end();

    const compBG = device.createBindGroup({ layout: compositePipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: tex.hdr.createView() }, { binding: 1, resource: compositeSampler },
      { binding: 2, resource: { buffer: compositeUBuf } }] });

    const fp = enc.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    });
    fp.setPipeline(compositePipe); fp.setBindGroup(0, compBG); fp.draw(3); fp.end();

    device.queue.submit([enc.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
