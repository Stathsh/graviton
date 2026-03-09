// Graviton v2 — Black Hole Astrophysics Simulator
// Zig/WASM + WebGPU Compute + TypeScript
// Features: volumetric accretion disk, relativistic jets, bloom,
// mouse orbit, time dilation viz, photon geodesics, screenshot

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
import bloomShader from './shaders/bloom.wgsl';
// @ts-ignore
import compositeShader from './shaders/composite.wgsl';
// @ts-ignore
import geodesicShader from './shaders/geodesic-render.wgsl';

const PARTICLE_COUNT = 200_000;
const PARTICLE_SIZE = 64; // 4 x vec4<f32> = 16 floats = 64 bytes
const GEODESIC_POINTS = 512; // points per geodesic path
const NUM_GEODESICS = 8;

interface Controls {
  mass: number;
  spin: number;
  accretion: number;
  temperature: number;
  camDist: number;
  camAngle: number;
  camOrbit: number;
  colorScheme: number;
  showTimeDilation: boolean;
  showGeodesics: boolean;
  bloomIntensity: number;
  jetPower: number;
}

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const noWebGPU = document.getElementById('no-webgpu') as HTMLElement;

  let gpu: GPUContext;
  let physics: KerrPhysics;

  try {
    [gpu, physics] = await Promise.all([
      initWebGPU(canvas),
      loadKerrWasm(),
    ]);
  } catch (e) {
    noWebGPU.style.display = 'flex';
    console.error(e);
    return;
  }

  const { device, context, format } = gpu;

  const resize = () => {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    context.configure({ device, format, alphaMode: 'opaque' });
  };
  resize();
  window.addEventListener('resize', resize);

  // --- Controls ---
  const controls: Controls = {
    mass: 3.0, spin: 0.5, accretion: 1.0, temperature: 6000,
    camDist: 20, camAngle: 30, camOrbit: 0,
    colorScheme: 0, showTimeDilation: false, showGeodesics: false,
    bloomIntensity: 0.8, jetPower: 1.0,
  };

  function bindControl(id: string, key: keyof Controls, suffix = '') {
    const el = document.getElementById(id) as HTMLInputElement;
    if (!el) return;
    const display = document.getElementById(id + 'Val');
    el.addEventListener('input', () => {
      (controls as Record<string, number>)[key] = parseFloat(el.value);
      if (display) display.textContent = el.value + suffix;
    });
  }

  bindControl('mass', 'mass');
  bindControl('spin', 'spin');
  bindControl('accretion', 'accretion');
  bindControl('temperature', 'temperature', 'K');
  bindControl('camDist', 'camDist');
  bindControl('camAngle', 'camAngle', '\u00B0');
  bindControl('bloomIntensity', 'bloomIntensity');
  bindControl('jetPower', 'jetPower');

  const colorSelect = document.getElementById('colorScheme') as HTMLSelectElement;
  if (colorSelect) colorSelect.addEventListener('change', () => {
    controls.colorScheme = parseInt(colorSelect.value);
  });

  const tdToggle = document.getElementById('timeDilation') as HTMLInputElement;
  if (tdToggle) tdToggle.addEventListener('change', () => {
    controls.showTimeDilation = tdToggle.checked;
  });

  const geoToggle = document.getElementById('geodesics') as HTMLInputElement;
  if (geoToggle) geoToggle.addEventListener('change', () => {
    controls.showGeodesics = geoToggle.checked;
  });

  // --- Mouse / Touch camera control ---
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  let manualOrbit = 0;
  let manualAngle = controls.camAngle;
  let autoRotate = true;

  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    autoRotate = false;
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouseX;
    const dy = e.clientY - lastMouseY;
    manualOrbit += dx * 0.005;
    manualAngle = Math.max(5, Math.min(85, manualAngle - dy * 0.3));
    controls.camAngle = manualAngle;
    const angleDisplay = document.getElementById('camAngleVal');
    const angleSlider = document.getElementById('camAngle') as HTMLInputElement;
    if (angleDisplay) angleDisplay.textContent = Math.round(manualAngle) + '\u00B0';
    if (angleSlider) angleSlider.value = String(Math.round(manualAngle));
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  canvas.addEventListener('mouseup', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    canvas.style.cursor = 'grab';
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    controls.camDist = Math.max(8, Math.min(60, controls.camDist + e.deltaY * 0.02));
    const distDisplay = document.getElementById('camDistVal');
    const distSlider = document.getElementById('camDist') as HTMLInputElement;
    if (distDisplay) distDisplay.textContent = controls.camDist.toFixed(1);
    if (distSlider) distSlider.value = String(controls.camDist.toFixed(1));
  }, { passive: false });

  // Touch support
  let lastTouchDist = 0;
  canvas.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDragging = true;
      autoRotate = false;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      lastTouchDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY,
      );
    }
  });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && isDragging) {
      const dx = e.touches[0].clientX - lastMouseX;
      const dy = e.touches[0].clientY - lastMouseY;
      manualOrbit += dx * 0.005;
      manualAngle = Math.max(5, Math.min(85, manualAngle - dy * 0.3));
      controls.camAngle = manualAngle;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      const dist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY,
      );
      controls.camDist = Math.max(8, Math.min(60, controls.camDist - (dist - lastTouchDist) * 0.05));
      lastTouchDist = dist;
    }
  }, { passive: false });

  canvas.addEventListener('touchend', () => { isDragging = false; });
  canvas.style.cursor = 'grab';

  // Screenshot button
  const screenshotBtn = document.getElementById('screenshot');
  if (screenshotBtn) {
    screenshotBtn.addEventListener('click', () => {
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `graviton-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
      });
    });
  }

  // Double-click to re-enable auto rotate
  canvas.addEventListener('dblclick', () => {
    autoRotate = true;
  });

  // --- Zig/WASM physics ---
  physics.setParams(controls.mass, controls.spin);

  // --- Particle buffer (bigger stride now: 4 vec4s) ---
  const particleBuffer = device.createBuffer({
    size: PARTICLE_COUNT * PARTICLE_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
  });

  // Initialize particles
  const isco = physics.computeISCO();
  const initData = new Float32Array(PARTICLE_COUNT * 16);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const off = i * 16;
    const isJet = i >= PARTICLE_COUNT * 0.9;
    const angle = Math.random() * Math.PI * 2;

    if (isJet) {
      const poleSign = Math.random() > 0.5 ? 1 : -1;
      const spread = Math.random() * 0.3;
      initData[off + 0] = Math.cos(angle) * spread * 2;
      initData[off + 1] = poleSign * (1 + Math.random() * 15);
      initData[off + 2] = Math.sin(angle) * spread * 2;
      initData[off + 3] = Math.random() * 30;
      initData[off + 4] = 0;
      initData[off + 5] = poleSign * (0.3 + Math.random() * 0.5);
      initData[off + 6] = 0;
      initData[off + 7] = 50000 + Math.random() * 100000;
      initData[off + 12] = 1.0; // type = jet
    } else {
      const radius = isco + Math.random() * 20;
      const height = (Math.random() - 0.5) * 0.3;
      const vOrbit = Math.sqrt(controls.mass / radius);
      initData[off + 0] = Math.cos(angle) * radius;
      initData[off + 1] = height;
      initData[off + 2] = Math.sin(angle) * radius;
      initData[off + 3] = Math.random() * 30;
      initData[off + 4] = -Math.sin(angle) * vOrbit;
      initData[off + 5] = 0;
      initData[off + 6] = Math.cos(angle) * vOrbit;
      initData[off + 7] = 3000 + Math.random() * 17000;
      initData[off + 12] = 0.0; // type = disk
    }
    initData[off + 8] = 0; // meta
    initData[off + 9] = 1.0;
    initData[off + 10] = 1.0;
    initData[off + 11] = 0.5;
    initData[off + 13] = 1.0; // time dilation
    initData[off + 14] = 0; // initial_r
    initData[off + 15] = Math.random(); // turbulence
  }
  device.queue.writeBuffer(particleBuffer, 0, initData);

  // --- Geodesic path buffer ---
  const geodesicBuffer = device.createBuffer({
    size: NUM_GEODESICS * GEODESIC_POINTS * 16, // vec4 per point
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // --- Compute pipeline ---
  const computeParamsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const computeModule = device.createShaderModule({ code: computeShader });
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'main' },
  });

  const computeBG = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: computeParamsBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  // --- Lensing pipeline ---
  const lensingUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lensingModule = device.createShaderModule({ code: lensingShader });

  // We'll render to an HDR offscreen texture for bloom
  const hdrFormat: GPUTextureFormat = 'rgba16float';

  const lensingPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: lensingModule, entryPoint: 'vs_main' },
    fragment: {
      module: lensingModule, entryPoint: 'fs_main',
      targets: [{ format: hdrFormat }],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'always' },
  });

  const lensingBG = device.createBindGroup({
    layout: lensingPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: lensingUniformBuffer } }],
  });

  // --- Particle render pipeline (renders to HDR) ---
  const renderUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const renderModule = device.createShaderModule({ code: renderShader });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: {
      module: renderModule, entryPoint: 'fs_main',
      targets: [{
        format: hdrFormat,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });

  const renderBG = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  // --- Geodesic render pipeline ---
  const geodesicUniformBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const geodesicModule = device.createShaderModule({ code: geodesicShader });
  const geodesicPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: geodesicModule, entryPoint: 'vs_main' },
    fragment: {
      module: geodesicModule, entryPoint: 'fs_main',
      targets: [{
        format: hdrFormat,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'line-strip' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });

  const geodesicBG = device.createBindGroup({
    layout: geodesicPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: geodesicUniformBuffer } },
      { binding: 1, resource: { buffer: geodesicBuffer } },
    ],
  });

  // --- Bloom compute pipelines ---
  const bloomModule = device.createShaderModule({ code: bloomShader });
  const bloomParamsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bloomThresholdPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: bloomModule, entryPoint: 'threshold_pass' },
  });

  const bloomBlurPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: bloomModule, entryPoint: 'blur_pass' },
  });

  // --- Composite pipeline (final output) ---
  const compositeModule = device.createShaderModule({ code: compositeShader });
  const compositeParamsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const compositeSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const compositePipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: compositeModule, entryPoint: 'vs_main' },
    fragment: {
      module: compositeModule, entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // --- Create offscreen textures (recreated on resize) ---
  function createRenderTextures(w: number, h: number) {
    const hdrTex = device.createTexture({
      size: [w, h],
      format: hdrFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const bloomTexA = device.createTexture({
      size: [w, h],
      format: hdrFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    const bloomTexB = device.createTexture({
      size: [w, h],
      format: hdrFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    const depthTex = device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return { hdrTex, bloomTexA, bloomTexB, depthTex };
  }

  let textures = createRenderTextures(canvas.width, canvas.height);
  let lastW = canvas.width;
  let lastH = canvas.height;

  // --- Geodesic tracing (uses Zig WASM) ---
  let lastGeodesicUpdate = 0;
  const geodesicData = new Float32Array(NUM_GEODESICS * GEODESIC_POINTS * 4);

  function traceGeodesics(camR: number, camTheta: number) {
    const impactParams = [-6, -4, -2.5, -1.5, 1.5, 2.5, 4, 6];

    for (let g = 0; g < NUM_GEODESICS; g++) {
      const b = impactParams[g] * controls.mass;
      const baseOff = g * GEODESIC_POINTS * 4;

      // Simple geodesic trace using the lensing ray marcher concept
      let px = camR * Math.sin(camTheta);
      let py = camR * Math.cos(camTheta) * 0.1;
      let pz = 0;
      let dx = -Math.sin(camTheta);
      let dy = -Math.cos(camTheta) * 0.1;
      let dz = b / camR;
      const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
      dx /= dl; dy /= dl; dz /= dl;

      const rs = 2 * controls.mass;

      for (let i = 0; i < GEODESIC_POINTS; i++) {
        const off = baseOff + i * 4;
        geodesicData[off] = px;
        geodesicData[off + 1] = py;
        geodesicData[off + 2] = pz;

        const r = Math.sqrt(px * px + py * py + pz * pz);
        const fade = Math.max(0, 1 - i / GEODESIC_POINTS) * Math.max(0, 1 - rs / r);
        geodesicData[off + 3] = fade * 0.5;

        if (r < rs * 0.6 || r > 80) break;

        // Deflection
        const deflect = rs / (r * r) * 1.2;
        const rx = -px / r, ry = -py / r, rz = -pz / r;
        dx += rx * deflect * 0.3;
        dy += ry * deflect * 0.3;
        dz += rz * deflect * 0.3;
        const nl = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx /= nl; dy /= nl; dz /= nl;

        const step = r < rs * 3 ? 0.15 : 0.4;
        px += dx * step;
        py += dy * step;
        pz += dz * step;
      }
    }
    device.queue.writeBuffer(geodesicBuffer, 0, geodesicData);
  }

  // --- Render loop ---
  let lastTime = performance.now();
  let frameCount = 0;
  let fpsAccum = 0;

  function frame(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    frameCount++;
    fpsAccum += dt;

    if (fpsAccum >= 0.5) {
      const fps = Math.round(frameCount / fpsAccum);
      document.getElementById('fps')!.textContent = `${fps} FPS`;
      document.getElementById('particles')!.textContent = `${(PARTICLE_COUNT / 1000).toFixed(0)}K particles`;
      frameCount = 0;
      fpsAccum = 0;
    }

    // Update physics info
    physics.setParams(controls.mass, controls.spin);
    const curIsco = physics.computeISCO();
    const curPhotonR = physics.computePhotonSphere();
    const rHorizon = controls.mass + Math.sqrt(Math.max(controls.mass * controls.mass - controls.spin * controls.spin, 0));
    const hawkingTemp = 1.227e23 / (8 * Math.PI * controls.mass);

    const infoEl = document.getElementById('physics-info');
    if (infoEl) {
      infoEl.innerHTML = [
        `<div>Event Horizon: <b>${rHorizon.toFixed(2)}M</b></div>`,
        `<div>Photon Sphere: <b>${curPhotonR.toFixed(2)}M</b></div>`,
        `<div>ISCO: <b>${curIsco.toFixed(2)}M</b></div>`,
        `<div>Schwarzschild r: <b>${(2 * controls.mass).toFixed(2)}M</b></div>`,
        `<div>Hawking Temp: <b>${hawkingTemp.toExponential(1)}K</b></div>`,
        `<div>Ergosphere: <b>${(2 * controls.mass).toFixed(2)}M</b> (equator)</div>`,
      ].join('');
    }

    // Resize check
    if (canvas.width !== lastW || canvas.height !== lastH) {
      textures.hdrTex.destroy();
      textures.bloomTexA.destroy();
      textures.bloomTexB.destroy();
      textures.depthTex.destroy();
      textures = createRenderTextures(canvas.width, canvas.height);
      lastW = canvas.width;
      lastH = canvas.height;
    }

    const time = now / 1000;

    // Camera
    const orbitAngle = (autoRotate ? time * 0.04 : 0) + manualOrbit;
    const camAngleRad = controls.camAngle * Math.PI / 180;
    const camPos: m.Vec3 = [
      Math.cos(orbitAngle) * Math.cos(camAngleRad) * controls.camDist,
      Math.sin(camAngleRad) * controls.camDist,
      Math.sin(orbitAngle) * Math.cos(camAngleRad) * controls.camDist,
    ];

    const view = m.lookAt(camPos, [0, 0, 0], [0, 1, 0]);
    const aspect = canvas.width / canvas.height;
    const proj = m.perspective(Math.PI / 4, aspect, 0.1, 300);
    const viewProj = m.multiply(proj, view);
    const invViewProj = m.invert(viewProj);

    // Trace geodesics periodically
    if (controls.showGeodesics && now - lastGeodesicUpdate > 200) {
      traceGeodesics(controls.camDist, camAngleRad);
      lastGeodesicUpdate = now;
    }

    // Update uniforms
    device.queue.writeBuffer(computeParamsBuffer, 0, new Float32Array([
      controls.mass, controls.spin, dt * 0.4, controls.accretion,
      time, PARTICLE_COUNT, curIsco, curPhotonR,
    ]));

    const lensingUniforms = new Float32Array(32);
    lensingUniforms.set(invViewProj, 0);
    lensingUniforms.set([camPos[0], camPos[1], camPos[2], 1], 16);
    lensingUniforms.set([controls.mass, controls.spin, time, controls.colorScheme], 20);
    lensingUniforms.set([rHorizon, curPhotonR, curIsco, 0], 24);
    device.queue.writeBuffer(lensingUniformBuffer, 0, lensingUniforms);

    const renderUniforms = new Float32Array(32);
    renderUniforms.set(viewProj, 0);
    renderUniforms.set([camPos[0], camPos[1], camPos[2], 1], 16);
    renderUniforms.set([controls.mass, time, controls.colorScheme, curIsco], 20);
    renderUniforms.set([controls.showTimeDilation ? 1 : 0, controls.bloomIntensity, 0, 0], 24);
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniforms);

    // Geodesic uniforms
    if (controls.showGeodesics) {
      const geoUniforms = new Float32Array(20);
      geoUniforms.set(viewProj, 0);
      geoUniforms.set([0.3, 0.8, 1.0, 0.6], 16); // cyan color
      device.queue.writeBuffer(geodesicUniformBuffer, 0, geoUniforms);
    }

    const encoder = device.createCommandEncoder();

    // 1. Compute pass
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBG);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    computePass.end();

    // 2. Render to HDR texture
    const hdrView = textures.hdrTex.createView();
    const depthView = textures.depthTex.createView();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: hdrView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    // Background lensing
    renderPass.setPipeline(lensingPipeline);
    renderPass.setBindGroup(0, lensingBG);
    renderPass.draw(3);

    // Particles (disk + jets)
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBG);
    renderPass.draw(6, PARTICLE_COUNT);

    // Geodesic paths
    if (controls.showGeodesics) {
      renderPass.setPipeline(geodesicPipeline);
      renderPass.setBindGroup(0, geodesicBG);
      for (let g = 0; g < NUM_GEODESICS; g++) {
        renderPass.draw(GEODESIC_POINTS, 1, g * GEODESIC_POINTS, 0);
      }
    }

    renderPass.end();

    // 3. Bloom: threshold extract
    const bloomBGA = device.createBindGroup({
      layout: bloomThresholdPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.hdrTex.createView() },
        { binding: 1, resource: textures.bloomTexA.createView() },
        { binding: 2, resource: { buffer: bloomParamsBuffer } },
      ],
    });

    device.queue.writeBuffer(bloomParamsBuffer, 0, new Float32Array([
      1.0 / canvas.width, 1.0 / canvas.height, controls.bloomIntensity, 0.4,
    ]));

    const bloomPass1 = encoder.beginComputePass();
    bloomPass1.setPipeline(bloomThresholdPipeline);
    bloomPass1.setBindGroup(0, bloomBGA);
    bloomPass1.dispatchWorkgroups(
      Math.ceil(canvas.width / 8),
      Math.ceil(canvas.height / 8),
    );
    bloomPass1.end();

    // Bloom: blur passes (ping-pong)
    for (let i = 0; i < 4; i++) {
      const srcTex = i % 2 === 0 ? textures.bloomTexA : textures.bloomTexB;
      const dstTex = i % 2 === 0 ? textures.bloomTexB : textures.bloomTexA;

      const blurBG = device.createBindGroup({
        layout: bloomBlurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcTex.createView() },
          { binding: 1, resource: dstTex.createView() },
          { binding: 2, resource: { buffer: bloomParamsBuffer } },
        ],
      });

      const blurPass = encoder.beginComputePass();
      blurPass.setPipeline(bloomBlurPipeline);
      blurPass.setBindGroup(0, blurBG);
      blurPass.dispatchWorkgroups(
        Math.ceil(canvas.width / 8),
        Math.ceil(canvas.height / 8),
      );
      blurPass.end();
    }

    // 4. Composite: HDR scene + bloom → final output
    device.queue.writeBuffer(compositeParamsBuffer, 0, new Float32Array([
      controls.bloomIntensity, 0.4, 1.2, 0,
    ]));

    const finalBloomTex = textures.bloomTexA; // after 4 passes, result is in A

    const compositeBG = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textures.hdrTex.createView() },
        { binding: 1, resource: finalBloomTex.createView() },
        { binding: 2, resource: compositeSampler },
        { binding: 3, resource: { buffer: compositeParamsBuffer } },
      ],
    });

    const compositePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    compositePass.setPipeline(compositePipeline);
    compositePass.setBindGroup(0, compositeBG);
    compositePass.draw(3);
    compositePass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
