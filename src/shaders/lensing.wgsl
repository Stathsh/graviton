// Full-screen gravitational lensing shader
// Uses Zig/WASM ray-traced lensing data uploaded as a texture,
// with GPU-computed starfield and black hole shadow

struct Uniforms {
  invViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  params: vec4<f32>,    // x=mass, y=spin, z=time, w=colorScheme
  metrics: vec4<f32>,   // x=r_horizon, y=r_photon, z=r_isco, w=0
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  let verts = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0),
  );
  var out: VSOut;
  out.pos = vec4(verts[i], 0.0, 1.0);
  out.uv = verts[i] * 0.5 + 0.5;
  return out;
}

// Procedural starfield with multiple layers
fn hash31(p: vec3<f32>) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

fn stars(dir: vec3<f32>, time: f32) -> vec3<f32> {
  var color = vec3(0.0);

  // Layer 1: bright stars
  let p1 = dir * 80.0;
  let g1 = floor(p1);
  let f1 = fract(p1);
  let h1 = hash31(g1);
  let h2 = hash31(g1 + vec3(1.0));
  let d1 = length(f1 - vec3(h1, h2, hash31(g1 + vec3(2.0))));
  let twinkle = 0.7 + 0.3 * sin(time * (1.5 + h1 * 3.0) + h1 * 50.0);
  let star1 = smoothstep(0.12, 0.0, d1) * twinkle;
  // Star temperature coloring
  let temp = h2;
  var starCol = vec3(1.0, 1.0, 1.0);
  if (temp < 0.2) { starCol = vec3(0.6, 0.7, 1.0); }    // O/B blue
  else if (temp < 0.4) { starCol = vec3(0.8, 0.85, 1.0); } // A white-blue
  else if (temp > 0.85) { starCol = vec3(1.0, 0.7, 0.4); } // K/M red
  else if (temp > 0.7) { starCol = vec3(1.0, 0.9, 0.6); }  // G yellow
  color += starCol * star1;

  // Layer 2: dim background stars (milky way-ish)
  let p2 = dir * 200.0;
  let g2 = floor(p2);
  let f2 = fract(p2);
  let d2 = length(f2 - vec3(hash31(g2), hash31(g2 + 1.0), hash31(g2 + 2.0)));
  color += vec3(0.8, 0.85, 1.0) * smoothstep(0.18, 0.0, d2) * 0.3;

  // Galactic plane glow
  let galactic = exp(-abs(dir.y) * 8.0) * 0.015;
  color += vec3(0.9, 0.8, 0.7) * galactic;

  return color;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let mass = u.params.x;
  let spin = u.params.y;
  let time = u.params.z;
  let rs = 2.0 * mass;
  let r_h = u.metrics.x;
  let r_ph = u.metrics.y;

  // Reconstruct world-space ray
  let ndc = vec2(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let near = u.invViewProj * vec4(ndc, -1.0, 1.0);
  let far = u.invViewProj * vec4(ndc, 1.0, 1.0);
  let rayDir = normalize(far.xyz / far.w - near.xyz / near.w);
  let rayOrigin = u.cameraPos.xyz;

  // GPU ray-march for lensing (simplified but fast)
  var pos = rayOrigin;
  var dir = rayDir;
  var absorbed = false;
  var closest = 999.0;
  var total_deflection = 0.0;
  let steps = 200;

  for (var i = 0; i < steps; i++) {
    let to_bh = -pos;
    let r = length(to_bh);
    closest = min(closest, r);

    if (r < r_h * 0.9) {
      absorbed = true;
      break;
    }

    // Gravitational deflection (Schwarzschild + Kerr frame-drag)
    let r_hat = normalize(to_bh);
    let deflect_strength = rs / (r * r) * 1.5;

    // Frame-dragging deflection (spin-dependent)
    let drag_dir = normalize(vec3(-pos.z, 0.0, pos.x));
    let drag_strength = spin * rs * rs / (r * r * r) * 0.4;

    dir = normalize(dir + r_hat * deflect_strength * 0.3 + drag_dir * drag_strength * 0.3);
    total_deflection += deflect_strength * 0.3;

    let step_size = select(0.5, select(0.2, 0.08, r < r_h * 3.0), r < r_h * 8.0);
    pos += dir * step_size;

    if (r > 120.0 && dot(dir, r_hat) < 0.0) { break; }
  }

  if (absorbed) {
    // Event horizon: pure black with faint Hawking radiation rim
    let rim = closest / r_h;
    let hawking = exp(-pow(rim - 1.0, 2.0) * 100.0) * 0.03;
    return vec4(hawking * 0.3, hawking * 0.15, hawking * 0.6, 1.0);
  }

  // Lensed starfield
  let sky = stars(normalize(dir), time);

  // Photon ring glow
  let photon_dist = abs(closest - r_ph);
  let photon_glow = exp(-photon_dist * photon_dist * 2.0) * 0.2;
  let glow_color = vec3(1.0, 0.7, 0.3) * photon_glow;

  // Einstein ring (secondary image)
  let einstein_r = rs * 2.2;
  let einstein_glow = exp(-pow(closest - einstein_r, 2.0) * 3.0) * 0.1;
  let einstein_color = vec3(0.7, 0.8, 1.0) * einstein_glow;

  // Gravitational redshift on background light near BH
  let rshift_factor = max(1.0 - total_deflection * 0.3, 0.0);
  let final_color = sky * rshift_factor + glow_color + einstein_color;

  return vec4(final_color, 1.0);
}
