// Graviton — Black Hole Astrophysics Simulator
// Zig/WASM (Kerr geodesics) + WebGPU (compute & render) + TypeScript

import { loadKerrWasm, type KerrPhysics } from './wasm-bridge';
import { initWebGPU, type GPUContext } from './gpu';
import * as m from './math';

// @ts-ignore — loaded via esbuild wgsl plugin
import computeShader from './shaders/particles-compute.wgsl';
// @ts-ignore
import renderShader from './shaders/particles-render.wgsl';
// @ts-ignore
import lensingShader from './shaders/lensing.wgsl';

const PARTICLE_COUNT = 200_000;
const PARTICLE_SIZE = 48; // 3 x vec4<f32> = 12 floats = 48 bytes

interface Controls {
  mass: number;
  spin: number;
  accretion: number;
  temperature: number;
  camDist: number;
  camAngle: number;
  colorScheme: number;
}

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const noWebGPU = document.getElementById('no-webgpu') as HTMLElement;

  // Initialize both Zig/WASM physics and WebGPU in parallel
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

  // Resize handler
  const resize = () => {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    context.configure({ device, format, alphaMode: 'opaque' });
  };
  resize();
  window.addEventListener('resize', resize);

  // Controls
  const controls: Controls = {
    mass: 3.0,
    spin: 0.5,
    accretion: 1.0,
    temperature: 6000,
    camDist: 20,
    camAngle: 30,
    colorScheme: 0,
  };

  function bindControl(id: string, key: keyof Controls, suffix = '') {
    const el = document.getElementById(id) as HTMLInputElement;
    const display = document.getElementById(id + 'Val')!;
    el.addEventListener('input', () => {
      (controls as Record<string, number>)[key] = parseFloat(el.value);
      display.textContent = el.value + suffix;
    });
  }

  bindControl('mass', 'mass');
  bindControl('spin', 'spin');
  bindControl('accretion', 'accretion');
  bindControl('temperature', 'temperature', 'K');
  bindControl('camDist', 'camDist');
  bindControl('camAngle', 'camAngle', '\u00B0');

  document.getElementById('colorScheme')!.addEventListener('change', (e) => {
    controls.colorScheme = parseInt((e.target as HTMLSelectElement).value);
  });

  // --- Set up Zig/WASM physics ---
  physics.setParams(controls.mass, controls.spin);
  const isco = physics.computeISCO();
  const photonR = physics.computePhotonSphere();

  const infoEl = document.getElementById('physics-info');
  if (infoEl) {
    infoEl.textContent = `ISCO: ${isco.toFixed(2)}M  Photon sphere: ${photonR.toFixed(2)}M`;
  }

  // --- Particle buffer ---
  const particleBuffer = device.createBuffer({
    size: PARTICLE_COUNT * PARTICLE_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
  });

  // Initialize particles
  const initData = new Float32Array(PARTICLE_COUNT * 12);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const off = i * 12;
    const angle = Math.random() * Math.PI * 2;
    const radius = isco + Math.random() * 20;
    const height = (Math.random() - 0.5) * 0.2;
    const vOrbit = Math.sqrt(controls.mass / radius);

    initData[off + 0] = Math.cos(angle) * radius;
    initData[off + 1] = height;
    initData[off + 2] = Math.sin(angle) * radius;
    initData[off + 3] = Math.random() * 30;
    initData[off + 4] = -Math.sin(angle) * vOrbit;
    initData[off + 5] = 0;
    initData[off + 6] = Math.cos(angle) * vOrbit;
    initData[off + 7] = 3000 + Math.random() * 17000;
    initData[off + 8] = radius;
    initData[off + 9] = 1.0;
    initData[off + 10] = 1.0;
    initData[off + 11] = 0.5;
  }
  device.queue.writeBuffer(particleBuffer, 0, initData);

  // --- Compute pipeline ---
  const computeParamsBuffer = device.createBuffer({
    size: 32, // 8 floats
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

  // --- Lensing (background) pipeline ---
  const lensingUniformBuffer = device.createBuffer({
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lensingModule = device.createShaderModule({ code: lensingShader });
  const lensingPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: lensingModule, entryPoint: 'vs_main' },
    fragment: {
      module: lensingModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'always',
    },
  });

  const lensingBG = device.createBindGroup({
    layout: lensingPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: lensingUniformBuffer } },
    ],
  });

  // --- Particle render pipeline ---
  const renderUniformBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const renderModule = device.createShaderModule({ code: renderShader });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: {
      module: renderModule,
      entryPoint: 'fs_main',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'always',
    },
  });

  const renderBG = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: renderUniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  // Depth texture
  let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // --- Render loop ---
  let lastTime = performance.now();
  let frameCount = 0;
  let fpsAccum = 0;
  let lastPhysicsUpdate = 0;

  function frame(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    frameCount++;
    fpsAccum += dt;

    if (fpsAccum >= 0.5) {
      document.getElementById('fps')!.textContent = `${Math.round(frameCount / fpsAccum)} FPS`;
      document.getElementById('particles')!.textContent = `${(PARTICLE_COUNT / 1000).toFixed(0)}K particles`;
      frameCount = 0;
      fpsAccum = 0;
    }

    // Update Zig/WASM physics params periodically (expensive, not every frame)
    if (now - lastPhysicsUpdate > 500) {
      physics.setParams(controls.mass, controls.spin);
      const newIsco = physics.computeISCO();
      const newPhotonR = physics.computePhotonSphere();
      if (infoEl) {
        infoEl.textContent = `ISCO: ${newIsco.toFixed(2)}M  Photon: ${newPhotonR.toFixed(2)}M  Horizon: ${(controls.mass + Math.sqrt(controls.mass * controls.mass - controls.spin * controls.spin)).toFixed(2)}M`;
      }
      lastPhysicsUpdate = now;
    }

    // Recompute ISCO/photon sphere for current params
    physics.setParams(controls.mass, controls.spin);
    const curIsco = physics.computeISCO();
    const curPhotonR = physics.computePhotonSphere();
    const rHorizon = controls.mass + Math.sqrt(Math.max(controls.mass * controls.mass - controls.spin * controls.spin, 0));

    // Resize depth texture if needed
    if (depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
      depthTexture.destroy();
      depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const time = now / 1000;

    // Camera
    const autoAngle = time * 0.04;
    const camAngleRad = controls.camAngle * Math.PI / 180;
    const camPos: m.Vec3 = [
      Math.cos(autoAngle) * Math.cos(camAngleRad) * controls.camDist,
      Math.sin(camAngleRad) * controls.camDist,
      Math.sin(autoAngle) * Math.cos(camAngleRad) * controls.camDist,
    ];

    const view = m.lookAt(camPos, [0, 0, 0], [0, 1, 0]);
    const aspect = canvas.width / canvas.height;
    const proj = m.perspective(Math.PI / 4, aspect, 0.1, 300);
    const viewProj = m.multiply(proj, view);
    const invViewProj = m.invert(viewProj);

    // Update compute params
    device.queue.writeBuffer(computeParamsBuffer, 0, new Float32Array([
      controls.mass, controls.spin, dt * 0.4, controls.accretion,
      time, PARTICLE_COUNT, curIsco, curPhotonR,
    ]));

    // Update lensing uniforms
    const lensingUniforms = new Float32Array(32);
    lensingUniforms.set(invViewProj, 0);
    lensingUniforms.set([camPos[0], camPos[1], camPos[2], 1], 16);
    lensingUniforms.set([controls.mass, controls.spin, time, controls.colorScheme], 20);
    lensingUniforms.set([rHorizon, curPhotonR, curIsco, 0], 24);
    device.queue.writeBuffer(lensingUniformBuffer, 0, lensingUniforms);

    // Update render uniforms
    const renderUniforms = new Float32Array(24);
    renderUniforms.set(viewProj, 0);
    renderUniforms.set([camPos[0], camPos[1], camPos[2], 1], 16);
    renderUniforms.set([controls.mass, time, controls.colorScheme, curIsco], 20);
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniforms);

    // Encode commands
    const encoder = device.createCommandEncoder();

    // 1. Compute pass: particle physics on GPU
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBG);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    computePass.end();

    // 2. Render pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    // 2a. Gravitational lensing background
    renderPass.setPipeline(lensingPipeline);
    renderPass.setBindGroup(0, lensingBG);
    renderPass.draw(3);

    // 2b. Accretion disk particles
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBG);
    renderPass.draw(6, PARTICLE_COUNT);

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
