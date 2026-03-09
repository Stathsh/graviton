// Particle rendering — tiny bright particles for dense fluid look

struct Uniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  params: vec4<f32>,   // x=mass, y=time, z=colorScheme, w=isco
  extra: vec4<f32>,    // x=showTimeDilation, y=0, z=0, w=0
}

struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
  meta: vec4<f32>,
  extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
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

fn blackbody(temp_k: f32) -> vec3<f32> {
  let t = clamp(temp_k, 1000.0, 60000.0) / 1000.0;
  var r: f32; var g: f32; var b: f32;
  if (t <= 6.6) { r = 1.0; } else { r = 1.29 * pow(t - 6.0, -0.133); }
  if (t <= 6.6) { g = 0.39 * log(t) - 0.63; } else { g = 1.13 * pow(t - 6.0, -0.076); }
  if (t >= 6.6) { b = 1.0; } else if (t <= 1.9) { b = 0.0; } else { b = 0.543 * log(t - 1.0) - 1.196; }
  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

fn apply_scheme(c: vec3<f32>, s: f32) -> vec3<f32> {
  let si = i32(s);
  if (si == 1) { return c * vec3(1.2, 0.6, 1.4); }
  if (si == 2) { return vec3(c.b * 0.6, c.g * 0.8 + c.b * 0.3, c.b * 1.5); }
  if (si == 3) { return c * vec3(1.4, 0.95, 0.5); }
  return c;
}

fn td_color(td: f32) -> vec3<f32> {
  let t = clamp(td, 0.0, 1.0);
  if (t < 0.3) { return mix(vec3(0.9, 0.0, 0.0), vec3(1.0, 0.4, 0.0), t / 0.3); }
  if (t < 0.6) { return mix(vec3(1.0, 0.4, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.3) / 0.3); }
  return mix(vec3(1.0, 1.0, 1.0), vec3(0.3, 0.7, 1.0), (t - 0.6) / 0.4);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
  let p = particles[iid];
  let q = QUAD[vid];
  let w = p.pos.xyz;
  let isco = u.params.w;
  let ptype = p.extra.x;

  var size: f32;
  var color: vec3<f32>;
  var glow_v: f32;

  if (ptype > 1.5) {
    // --- Star (tidal disruption) ---
    let r = p.meta.x;
    let stretch = p.meta.y;
    let grav_rs = p.meta.z;

    // Size: stretch along velocity direction (spaghettification)
    size = 0.03 + min(stretch * 0.01, 0.08);
    glow_v = smoothstep(isco * 8.0, isco, r);

    // Color: star starts yellow-white, heats to hot orange/white near BH
    let bb = blackbody(p.vel.w);
    // Boost warm colors for stellar look
    color = bb * vec3(1.1, 0.95, 0.85);
    // Tidal heating glow — gets brighter and whiter as it heats
    let heat_glow = smoothstep(6000.0, 40000.0, p.vel.w);
    color = mix(color, vec3(2.0, 1.5, 1.2), heat_glow * 0.6);
    // Inner region extreme brightness
    color *= (1.0 + glow_v * 3.0);
    color *= grav_rs; // gravitational dimming
  } else if (ptype < 0.5) {
    let r = p.meta.x;
    let ig = smoothstep(isco * 4.0, isco * 0.8, r);
    // Tiny particles — denser look
    size = 0.02 + ig * 0.06;
    glow_v = ig;

    let eff_temp = p.vel.w * p.meta.z; // temp * doppler
    let bb = blackbody(eff_temp);
    color = apply_scheme(bb, u.params.z);

    if (u.extra.x > 0.5) {
      color = mix(color, td_color(p.extra.y), 0.55);
    }

    // Bright inner disk
    color *= (1.0 + ig * 4.0);
    // Doppler brightening on approaching side
    color *= (0.6 + p.meta.z * 0.8);
  } else {
    let speed = p.meta.z;
    let beam = p.meta.y;
    size = 0.015 + speed * 0.04;
    glow_v = speed;
    let bb = blackbody(p.vel.w);
    // Jets: electric blue-violet glow
    color = vec3(bb.r * 0.3 + 0.1, bb.g * 0.4 + 0.15, bb.b * 1.2 + 0.5) * beam * 1.8;
  }

  let toCam = normalize(u.cameraPos.xyz - w);
  let right = normalize(cross(toCam, vec3(0.0, 1.0, 0.0)));
  let up = cross(right, toCam);
  let bp = w + (right * q.x + up * q.y) * size;

  var out: VSOut;
  out.pos = u.viewProj * vec4(bp, 1.0);
  out.uv = q;
  out.color = color;
  out.alpha = p.meta.w;
  out.glow = glow_v;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);

  // Sharp bright core + soft wide glow
  let core = exp(-d * d * 8.0);
  let mid = exp(-d * d * 3.0) * 0.5;
  let halo = exp(-d * d * 0.8) * in.glow * 0.3;
  let total = core + mid + halo;
  let alpha = total * in.alpha;

  if (alpha < 0.001) { discard; }

  // HDR: allow values > 1 for bloom
  let final_color = in.color * total;
  return vec4(final_color, alpha);
}
