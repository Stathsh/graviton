// WebGPU Vertex/Fragment shader — Accretion disk particle rendering
// Physically-motivated coloring with blackbody radiation and relativistic effects

struct Uniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  params: vec4<f32>,  // x=mass, y=time, z=colorScheme, w=isco
}

struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
  meta: vec4<f32>,  // x=orbital_r, y=redshift, z=doppler, w=alpha
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
  @location(3) glow: f32,
}

const QUAD = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
);

// Attempt at planck blackbody → RGB
fn blackbody(temp_k: f32) -> vec3<f32> {
  let t = clamp(temp_k, 1000.0, 40000.0) / 1000.0;

  // Red channel
  var r: f32;
  if (t <= 6.6) {
    r = 1.0;
  } else {
    r = 1.292936 * pow(t - 6.0, -0.1332047);
  }

  // Green channel
  var g: f32;
  if (t <= 6.6) {
    g = 0.39008 * log(t) - 0.63184;
  } else {
    g = 1.129891 * pow(t - 6.0, -0.0755148);
  }

  // Blue channel
  var b: f32;
  if (t >= 6.6) {
    b = 1.0;
  } else if (t <= 1.9) {
    b = 0.0;
  } else {
    b = 0.54320 * log(t - 1.0) - 1.19625;
  }

  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

fn colorScheme(base: vec3<f32>, scheme: f32, glow_factor: f32) -> vec3<f32> {
  let s = i32(scheme);
  if (s == 1) { // Plasma — purple/magenta shift
    return base * vec3(1.2, 0.6, 1.4);
  }
  if (s == 2) { // Cool Blue
    return vec3(base.b * 0.6, base.g * 0.8 + base.b * 0.3, base.b * 1.5 + base.r * 0.2);
  }
  if (s == 3) { // Interstellar — warm golden
    return base * vec3(1.3, 0.9, 0.5) + vec3(glow_factor * 0.2, glow_factor * 0.1, 0.0);
  }
  // Default: natural blackbody
  return base;
}

@vertex
fn vs_main(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> VSOut {
  let p = particles[iid];
  let quad = QUAD[vid];
  let world = p.pos.xyz;
  let isco = uniforms.params.w;
  let mass = uniforms.params.x;
  let rs = 2.0 * mass;

  // Particle size: bigger near ISCO, smaller far away
  let r = p.meta.x;
  let inner_glow = smoothstep(isco * 3.0, isco, r);
  let size = 0.04 + inner_glow * 0.08;

  // Billboard towards camera
  let toCam = normalize(uniforms.cameraPos.xyz - world);
  let right = normalize(cross(toCam, vec3(0.0, 1.0, 0.0)));
  let up = cross(right, toCam);
  let billboard = world + (right * quad.x + up * quad.y) * size;

  // Color from temperature + relativistic corrections
  let effective_temp = p.vel.w * p.meta.z; // temperature * doppler
  let base_color = blackbody(effective_temp);
  let scheme = uniforms.params.z;
  let final_color = colorScheme(base_color, scheme, inner_glow);

  // Intensity boost near ISCO (matter is hottest/brightest there)
  let intensity = 1.0 + inner_glow * 2.0;

  var out: VSOut;
  out.pos = uniforms.viewProj * vec4(billboard, 1.0);
  out.uv = quad;
  out.color = final_color * intensity;
  out.alpha = p.meta.w;
  out.glow = inner_glow;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);

  // Soft particle with glow halo
  let core = exp(-d * d * 4.0);
  let halo = exp(-d * d * 1.5) * in.glow * 0.3;
  let alpha = (core + halo) * in.alpha;

  if (alpha < 0.002) { discard; }

  return vec4(in.color * (core + halo * 0.5), alpha);
}
