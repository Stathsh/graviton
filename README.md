# Graviton

**Real-time black hole astrophysics simulator running entirely in the browser.**

Graviton renders a physically accurate spinning (Kerr) black hole with gravitational lensing, an accretion disk of 200,000 GPU-computed particles, and relativistic visual effects — all with zero server-side computation.

### [Live Demo → graviton.alexstath.com](https://graviton.alexstath.com)

> Requires a WebGPU-capable browser: Chrome 113+, Edge 113+, or Firefox 141+.

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Physics Engine** | Zig 0.14 → WebAssembly | Kerr metric geodesic integrator using 4th-order Runge-Kutta. Computes ISCO, photon sphere, and ray-traces null geodesics through curved spacetime. 15KB binary. |
| **Particle Simulation** | WebGPU Compute Shaders (WGSL) | 200K accretion disk particles simulated on the GPU with Kerr-approximated gravity, frame-dragging, and velocity Verlet integration. |
| **Rendering** | WebGPU Render Pipelines (WGSL) | Additive-blended billboard particles with blackbody radiation coloring, gravitational redshift, and Doppler shifting. Full-screen gravitational lensing via GPU ray-marching. |
| **Application** | TypeScript | Zero-dependency orchestration — raw WebGPU API, custom matrix math, WASM bridge. No frameworks. |
| **Build** | esbuild + Zig build system | Sub-second builds. TypeScript bundled to ESM, Zig cross-compiled to `wasm32-freestanding`. |
| **Deploy** | Docker + Cloudflare Tunnel | Multi-stage Docker build, served via Cloudflare edge network. |

## Physics

Graviton simulates a **Kerr black hole** — a rotating black hole described by two parameters: mass (M) and spin (a).

**What the Zig/WASM module computes:**
- **Geodesic equations** in Boyer-Lindquist coordinates with the full Kerr metric
- **Innermost Stable Circular Orbit (ISCO)** — the smallest stable orbit, which depends on spin. For a non-spinning black hole ISCO = 6M, for maximal spin it drops to ~1M
- **Photon sphere** — the radius where light orbits the black hole
- **Null geodesic ray tracing** — photon paths bent by spacetime curvature, using the Carter constant as the third integral of motion

**What the WebGPU compute shaders simulate:**
- Particle orbits with Kerr-corrected gravitational acceleration
- Lense-Thirring frame-dragging (spacetime itself rotates near the black hole)
- Post-Newtonian orbital corrections
- Gravitational redshift: `z = 1/√(1 - rₛ/r)`
- Relativistic Doppler shift from orbital velocity

**What the WebGPU fragment shaders render:**
- Blackbody radiation spectrum (1,000K → 30,000K) mapped to RGB
- Gravitational lensing of background starfield via ray-marching through curved spacetime
- Einstein ring and photon ring glow
- Hawking radiation glow at the event horizon boundary

## Interactive Controls

| Parameter | Effect |
|-----------|--------|
| **Mass** | Changes the Schwarzschild radius and gravitational strength |
| **Spin (a/M)** | Controls frame-dragging intensity. Higher spin = smaller ISCO, asymmetric disk |
| **Accretion Rate** | How far the disk extends and how quickly particles are spawned |
| **Disk Temperature** | Base temperature of accreting matter |
| **Camera Distance** | Orbital distance from the black hole |
| **Viewing Angle** | Inclination — edge-on vs face-on view of the accretion disk |
| **Color Scheme** | Blackbody (physically accurate), Plasma, Cool Blue, or Interstellar |

## Project Structure

```
graviton/
├── zig/
│   ├── build.zig              # Zig build config (wasm32-freestanding target)
│   └── src/
│       └── kerr.zig           # Kerr metric geodesic integrator (307 lines)
├── src/
│   ├── main.ts                # Application entry — WebGPU orchestration
│   ├── gpu.ts                 # WebGPU device initialization
│   ├── math.ts                # Matrix/vector math (no dependencies)
│   ├── wasm-bridge.ts         # Zig WASM ↔ TypeScript interface
│   └── shaders/
│       ├── particles-compute.wgsl   # GPU compute: particle physics
│       ├── particles-render.wgsl    # Vertex/fragment: accretion disk
│       └── lensing.wgsl             # Full-screen: gravitational lensing
├── public/
│   └── index.html
├── build.js                   # esbuild bundler + WGSL loader
├── server.js                  # Static file server
├── Dockerfile                 # Multi-stage build
└── docker-compose.yml
```

## Build & Run

**Prerequisites:** [Zig 0.14+](https://ziglang.org/download/), Node.js 22+

```bash
# Install dependencies
npm install

# Compile Zig → WASM and bundle TypeScript
npm run build

# Start dev server
npm run dev
# → http://localhost:8004
```

**Docker:**

```bash
docker compose up -d --build
```

## How It Works

1. **Zig compiles to a 15KB WASM binary** that exposes functions for setting black hole parameters, computing orbital radii, and tracing photon geodesics through Kerr spacetime.

2. **On each frame**, the TypeScript app updates parameters and dispatches a **WebGPU compute shader** that runs 200,000 particle physics calculations in parallel on the GPU — gravity, frame-dragging, orbital mechanics.

3. **Two render passes** execute: first, a full-screen triangle renders the gravitational lensing background (starfield warped by ray-marching through curved spacetime). Then, 200K billboard particles render the accretion disk with additive blending and blackbody coloring.

4. **The Zig WASM module** provides the physically accurate orbital parameters (ISCO, photon sphere radius, event horizon) that the GPU shaders use to correctly position and color particles.

## License

MIT
