// WebGPU Compute Shader — Accretion disk particle physics
// Kerr-metric approximation with frame-dragging and relativistic corrections

struct Params {
  mass: f32,
  spin: f32,
  dt: f32,
  accretion_rate: f32,
  time: f32,
  particle_count: u32,
  isco: f32,     // Innermost stable circular orbit (from Zig WASM)
  photon_r: f32, // Photon sphere radius (from Zig WASM)
}

struct Particle {
  pos: vec4<f32>,    // xyz position, w = age
  vel: vec4<f32>,    // xyz velocity, w = temperature
  meta: vec4<f32>,   // x = orbital_r, y = redshift, z = doppler, w = alpha
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand(id: u32, salt: u32) -> f32 {
  return f32(pcg_hash(id * 1099087573u + salt)) / 4294967295.0;
}

// Boyer-Lindquist coordinate helpers
fn sigma_val(r: f32, cos_th: f32) -> f32 {
  let a = params.spin;
  return r * r + a * a * cos_th * cos_th;
}

fn delta_val(r: f32) -> f32 {
  let a = params.spin;
  return r * r - 2.0 * params.mass * r + a * a;
}

// Kerr metric gravitational acceleration with frame-dragging
fn kerr_accel(pos: vec3<f32>) -> vec3<f32> {
  let r = length(pos);
  let M = params.mass;
  let a_spin = params.spin;
  let rs = 2.0 * M;
  let r_h = M + sqrt(max(M * M - a_spin * a_spin, 0.0));

  if (r < r_h * 0.8) { return vec3<f32>(0.0); }

  // Radial direction
  let r_hat = pos / r;

  // Newtonian gravity with post-Newtonian correction
  let pn_factor = 1.0 + 1.5 * rs / r + 2.0 * rs * rs / (r * r);
  var acc = -M / (r * r) * r_hat * pn_factor;

  // Frame-dragging (Lense-Thirring precession)
  // Creates a torque that drags particles in the direction of BH spin
  let omega_fd = 2.0 * M * a_spin / (r * r * r);
  let drag_dir = vec3<f32>(-pos.z, 0.0, pos.x); // tangential
  acc += omega_fd * normalize(drag_dir) * length(pos.xz);

  // Effective potential barrier near ISCO
  let l = length(cross(pos, vec3<f32>(0.0))) ; // angular momentum proxy
  if (r < params.isco * 1.2 && r > r_h) {
    let repulsion = 0.5 * rs / (r * r * r);
    acc += r_hat * repulsion;
  }

  return acc;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.particle_count) { return; }

  var p = particles[idx];
  let dt = params.dt;
  let M = params.mass;
  let a_spin = params.spin;
  let rs = 2.0 * M;
  let r_h = M + sqrt(max(M * M - a_spin * a_spin, 0.0));
  let seed = bitcast<u32>(params.time * 1000.0) + idx;

  let r = length(p.pos.xyz);
  let needs_respawn = r < r_h * 0.9 || r > 80.0 || p.pos.w > 40.0 || r != r; // NaN check

  if (needs_respawn) {
    let angle = rand(idx, seed) * 6.283185;
    let radius = params.isco + rand(idx, seed + 1u) * M * 12.0 * params.accretion_rate;
    let height = (rand(idx, seed + 2u) - 0.5) * 0.15 * sqrt(radius / params.isco);

    p.pos = vec4<f32>(
      cos(angle) * radius,
      height,
      sin(angle) * radius,
      0.0,
    );

    // Keplerian orbital velocity with Kerr correction
    let v_kepler = sqrt(M / radius);
    let kerr_correction = 1.0 + a_spin * sqrt(M) / (radius * sqrt(radius));
    let tangent = vec3<f32>(-sin(angle), 0.0, cos(angle));
    let v_orbit = v_kepler * kerr_correction;

    // Add slight radial drift (accretion)
    let radial = normalize(p.pos.xyz) * (-0.002 * params.accretion_rate);

    p.vel = vec4<f32>(
      tangent * v_orbit + radial,
      4000.0 + rand(idx, seed + 3u) * 16000.0,
    );

    p.meta = vec4<f32>(radius, 1.0, 1.0, 0.0);
  }

  // Velocity Verlet integration
  let acc1 = kerr_accel(p.pos.xyz);
  let new_pos = p.pos.xyz + p.vel.xyz * dt + 0.5 * acc1 * dt * dt;
  let acc2 = kerr_accel(new_pos);
  let new_vel = p.vel.xyz + 0.5 * (acc1 + acc2) * dt;

  p.pos = vec4<f32>(new_pos, p.pos.w + dt);
  p.vel = vec4<f32>(new_vel, p.vel.w);

  // Compute visual metadata
  let new_r = length(new_pos);
  let orbital_v = length(new_vel);

  // Gravitational redshift: z = 1/sqrt(1 - rs/r) - 1
  let redshift = 1.0 / sqrt(max(1.0 - rs / new_r, 0.01));

  // Doppler factor (simplified): approaching = blueshift, receding = redshift
  let radial_v = dot(normalize(new_pos), new_vel);
  let doppler = sqrt((1.0 - radial_v) / (1.0 + radial_v + 0.001));

  // Temperature increases closer to ISCO
  let temp_boost = pow(params.isco / max(new_r, params.isco), 0.75);
  p.vel.w = mix(p.vel.w, p.vel.w * temp_boost, 0.01);

  // Alpha based on position, temperature, and age
  let disk_height = abs(new_pos.y);
  let disk_alpha = exp(-disk_height * disk_height * 50.0);
  let age_alpha = smoothstep(40.0, 30.0, p.pos.w);
  let inner_boost = smoothstep(params.isco * 4.0, params.isco, new_r);

  p.meta = vec4<f32>(
    new_r,
    redshift,
    doppler,
    disk_alpha * age_alpha * (0.4 + inner_boost * 0.6),
  );

  particles[idx] = p;
}
