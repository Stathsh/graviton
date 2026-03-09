// Bridge between Zig WASM physics and TypeScript

export interface KerrPhysics {
  memory: WebAssembly.Memory;
  setParams(mass: number, spin: number): void;
  traceRay(cam_r: number, cam_theta: number, cam_phi: number, alpha: number, beta: number): number;
  traceRayBatch(
    cam_r: number, cam_theta: number, cam_phi: number,
    alpha_min: number, alpha_max: number,
    beta_min: number, beta_max: number,
    grid_w: number, grid_h: number,
  ): void;
  getResultsPtr(): number;
  getResultCount(): number;
  resetResults(): void;
  computeISCO(): number;
  computePhotonSphere(): number;
}

export async function loadKerrWasm(): Promise<KerrPhysics> {
  const response = await fetch('/wasm/kerr.wasm');
  const bytes = await response.arrayBuffer();

  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {},
  });

  return instance.exports as unknown as KerrPhysics;
}

export function readResults(physics: KerrPhysics): Float64Array {
  const ptr = physics.getResultsPtr();
  const count = physics.getResultCount();
  return new Float64Array(physics.memory.buffer, ptr, count * 4);
}
