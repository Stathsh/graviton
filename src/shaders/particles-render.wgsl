// Particle rendering — accretion disk + relativistic jets
// Blackbody radiation, time dilation coloring, volumetric appearance

struct Uniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  params: vec4<f32>,   // x=mass, y=time, z=colorScheme, w=isco
  extra: vec4<f32>,    // x=showTimeDilation, y=bloomIntensity, z=0, w=0
}

struct Particle {
  pos: vec4<f32>,
  vel: vec4<f32>,
  meta: vec4<f32>,
  extra: vec4<f32>,   // x=type, y=time_dilation, z=initial_r, w=turbulence
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
  @location(3) glow: f32,
  @location(4) particle_type: f32,
}

const QUAD = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
);

fn blackbody(temp_k: f32) -> vec3<f32> {
  let t = clamp(temp_k, 1000.0, 60000.0) / 1000.0;
  var r: f32; var g: f32; var b: f32;
  if (t <= 6.6) { r = 1.0; } else { r = 1.292936 * pow(t - 6.0, -0.1332047); }
  if (t <= 6.6) { g = 0.39008 * log(t) - 0.63184; } else { g = 1.129891 * pow(t - 6.0, -0.0755148); }
  if (t >= 6.6) { b = 1.0; } else if (t <= 1.9) { b = 0.0; } else { b = 0.54320 * log(t - 1.0) - 1.19625; }
  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

fn colorScheme(base: vec3<f32>, scheme: f32, glow_factor: f32) -> vec3<f32> {
  let s = i32(scheme);
  if (s == 1) { return base * vec3(1.2, 0.6, 1.4); }
  if (s == 2) { return vec3(base.b * 0.6, base.g * 0.8 + base.b * 0.3, base.b * 1.5 + base.r * 0.2); }
  if (s == 3) { return base * vec3(1.3, 0.9, 0.5) + vec3(glow_factor * 0.2, glow_factor * 0.1, 0.0); }
  return base;
}

// Time dilation visualization: blue = fast time, red = slow time
fn timeDilationColor(td: f32) -> vec3<f32> {
  // td=1 → normal, td→0 → extreme dilation
  let t = clamp(td, 0.0, 1.0);
  // Red (frozen) → Orange → Yellow → White → Cyan → Blue (fast)
  if (t < 0.3) { return mix(vec3(0.8, 0.0, 0.0), vec3(1.0, 0.4, 0.0), t / 0.3); }
  if (t < 0.5) { return mix(vec3(1.0, 0.4, 0.0), vec3(1.0, 0.8, 0.2), (t - 0.3) / 0.2); }
  if (t < 0.7) { return mix(vec3(1.0, 0.8, 0.2), vec3(1.0, 1.0, 1.0), (t - 0.5) / 0.2); }
  if (t < 0.85) { return mix(vec3(1.0, 1.0, 1.0), vec3(0.4, 0.9, 1.0), (t - 0.7) / 0.15); }
  return mix(vec3(0.4, 0.9, 1.0), vec3(0.2, 0.5, 1.0), (t - 0.85) / 0.15);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> VSOut {
  let p = particles[iid];
  let quad = QUAD[vid];
  let world = p.pos.xyz;
  let isco = u.params.w;
  let ptype = p.extra.x;

  var size: f32;
  var color: vec3<f32>;
  var glow_val: f32;

  if (ptype < 0.5) {
    // Disk particle
    let r = p.meta.x;
    let inner_glow = smoothstep(isco * 4.0, isco, r);
    size = 0.04 + inner_glow * 0.1;
    glow_val = inner_glow;

    let effective_temp = p.vel.w * p.meta.z;
    let base_color = blackbody(effective_temp);
    let scheme = u.params.z;
    color = colorScheme(base_color, scheme, inner_glow);

    // Time dilation overlay
    if (u.extra.x > 0.5) {
      let td_color = timeDilationColor(p.extra.y);
      color = mix(color, td_color, 0.6);
    }

    let intensity = 1.0 + inner_glow * 3.0;
    color *= intensity;
  } else {
    // Jet particle — electric blue/white/violet
    let speed = p.meta.z;
    let beam = p.meta.y;
    size = 0.03 + speed * 0.06;
    glow_val = speed;

    // Hot jet: blue-white core
    let jet_temp = p.vel.w;
    let base = blackbody(jet_temp);
    // Tint towards blue-violet for jets
    color = vec3(base.r * 0.4, base.g * 0.6, base.b * 1.5 + 0.3);
    color *= beam * 1.5;

    // Time dilation overlay for jets too
    if (u.extra.x > 0.5) {
      let td_color = timeDilationColor(p.extra.y);
      color = mix(color, td_color, 0.4);
    }
  }

  // Billboard
  let toCam = normalize(u.cameraPos.xyz - world);
  let right = normalize(cross(toCam, vec3(0.0, 1.0, 0.0)));
  let up = cross(right, toCam);
  let billboard = world + (right * quad.x + up * quad.y) * size;

  var out: VSOut;
  out.pos = u.viewProj * vec4(billboard, 1.0);
  out.uv = quad;
  out.color = color;
  out.alpha = p.meta.w;
  out.glow = glow_val;
  out.particle_type = ptype;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);

  var core: f32;
  var halo: f32;

  if (in.particle_type < 0.5) {
    // Disk: soft round particle
    core = exp(-d * d * 4.0);
    halo = exp(-d * d * 1.2) * in.glow * 0.4;
  } else {
    // Jet: elongated, streaky
    core = exp(-d * d * 6.0);
    halo = exp(-d * d * 1.0) * in.glow * 0.6;
  }

  let alpha = (core + halo) * in.alpha;
  if (alpha < 0.001) { discard; }

  // Extra bloom contribution for bright areas
  let bloom_contrib = in.glow * core * 0.3;
  let final_color = in.color * (core + halo * 0.5) + in.color * bloom_contrib;

  return vec4(final_color, alpha);
}
