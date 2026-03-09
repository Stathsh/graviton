// WebGPU Compute — Accretion disk + relativistic jet particle physics
// Kerr metric with frame-dragging, time dilation, volumetric disk, polar jets

struct Params {
  mass: f32,
  spin: f32,
  dt: f32,
  accretion_rate: f32,
  time: f32,
  particle_count: u32,
  isco: f32,
  photon_r: f32,
}

// Particle types: 0 = disk, 1 = jet
struct Particle {
  pos: vec4<f32>,    // xyz position, w = age
  vel: vec4<f32>,    // xyz velocity, w = temperature
  meta: vec4<f32>,   // x = orbital_r, y = redshift, z = doppler, w = alpha
  extra: vec4<f32>,  // x = type (0=disk,1=jet), y = time_dilation, z = initial_r, w = turbulence
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

fn kerr_accel(pos: vec3<f32>) -> vec3<f32> {
  let r = length(pos);
  let M = params.mass;
  let a_spin = params.spin;
  let rs = 2.0 * M;
  let r_h = M + sqrt(max(M * M - a_spin * a_spin, 0.0));

  if (r < r_h * 0.8) { return vec3<f32>(0.0); }

  let r_hat = pos / r;
  let pn_factor = 1.0 + 1.5 * rs / r + 2.0 * rs * rs / (r * r);
  var acc = -M / (r * r) * r_hat * pn_factor;

  let omega_fd = 2.0 * M * a_spin / (r * r * r);
  let drag_dir = vec3<f32>(-pos.z, 0.0, pos.x);
  acc += omega_fd * normalize(drag_dir) * length(pos.xz);

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
  let needs_respawn = r < r_h * 0.9 || r > 100.0 || p.pos.w > 50.0 || r != r;

  // 10% of particles are jet particles
  let is_jet = idx >= params.particle_count * 9u / 10u;

  if (needs_respawn) {
    if (is_jet) {
      // --- Jet particle spawn ---
      let jet_r = r_h * 1.2 + rand(idx, seed) * r_h * 0.5;
      let angle = rand(idx, seed + 1u) * 6.283185;
      let spread = rand(idx, seed + 2u) * 0.4;
      // Choose top or bottom jet
      let pole_sign = select(-1.0, 1.0, rand(idx, seed + 3u) > 0.5);

      p.pos = vec4<f32>(
        cos(angle) * spread * jet_r * 0.3,
        pole_sign * jet_r * 0.5,
        sin(angle) * spread * jet_r * 0.3,
        0.0,
      );

      // Jet velocity — primarily vertical, very fast
      let jet_speed = 0.4 + rand(idx, seed + 4u) * 0.5; // up to ~0.9c
      let wobble_x = (rand(idx, seed + 5u) - 0.5) * 0.08;
      let wobble_z = (rand(idx, seed + 6u) - 0.5) * 0.08;

      p.vel = vec4<f32>(
        wobble_x * jet_speed,
        pole_sign * jet_speed,
        wobble_z * jet_speed,
        50000.0 + rand(idx, seed + 7u) * 100000.0, // jets are extremely hot
      );

      p.extra = vec4<f32>(1.0, 1.0, jet_r, rand(idx, seed + 8u));
    } else {
      // --- Disk particle spawn (volumetric) ---
      let angle = rand(idx, seed) * 6.283185;
      let radius = params.isco + rand(idx, seed + 1u) * M * 14.0 * params.accretion_rate;

      // Volumetric: thicker disk further out, thinner near ISCO
      let disk_scale = 0.08 + 0.25 * sqrt(radius / (params.isco * 4.0));
      let height = (rand(idx, seed + 2u) - 0.5) * disk_scale * radius;

      // Turbulence factor per particle
      let turbulence = 0.5 + rand(idx, seed + 9u) * 0.5;

      p.pos = vec4<f32>(
        cos(angle) * radius,
        height,
        sin(angle) * radius,
        0.0,
      );

      let v_kepler = sqrt(M / radius);
      let kerr_correction = 1.0 + a_spin * sqrt(M) / (radius * sqrt(radius));
      let tangent = vec3<f32>(-sin(angle), 0.0, cos(angle));
      let v_orbit = v_kepler * kerr_correction;
      let radial = normalize(p.pos.xyz) * (-0.003 * params.accretion_rate);

      // Add turbulent velocity component
      let turb_vel = vec3<f32>(
        (rand(idx, seed + 10u) - 0.5) * 0.02 * turbulence,
        (rand(idx, seed + 11u) - 0.5) * 0.01 * turbulence,
        (rand(idx, seed + 12u) - 0.5) * 0.02 * turbulence,
      );

      p.vel = vec4<f32>(
        tangent * v_orbit + radial + turb_vel,
        3000.0 + rand(idx, seed + 3u) * 17000.0,
      );

      p.extra = vec4<f32>(0.0, 1.0, radius, turbulence);
    }
    p.meta = vec4<f32>(0.0, 1.0, 1.0, 0.0);
  }

  let particle_type = p.extra.x;

  if (particle_type < 0.5) {
    // --- Disk particle physics ---
    let acc1 = kerr_accel(p.pos.xyz);
    let new_pos = p.pos.xyz + p.vel.xyz * dt + 0.5 * acc1 * dt * dt;
    let acc2 = kerr_accel(new_pos);
    let new_vel = p.vel.xyz + 0.5 * (acc1 + acc2) * dt;

    p.pos = vec4<f32>(new_pos, p.pos.w + dt);
    p.vel = vec4<f32>(new_vel, p.vel.w);

    let new_r = length(new_pos);

    // Gravitational redshift
    let redshift = 1.0 / sqrt(max(1.0 - rs / new_r, 0.01));

    // Doppler
    let radial_v = dot(normalize(new_pos), new_vel);
    let doppler = sqrt(max((1.0 - radial_v) / (1.0 + radial_v + 0.001), 0.01));

    // Temperature evolution — hotter near ISCO
    let temp_boost = pow(params.isco / max(new_r, params.isco), 0.75);
    p.vel.w = mix(p.vel.w, p.vel.w * temp_boost, 0.01);

    // Time dilation: dt_proper/dt_coord = sqrt(1 - rs/r) for Schwarzschild
    let time_dilation = sqrt(max(1.0 - rs / new_r, 0.001));

    // Volumetric alpha with turbulence
    let disk_height = abs(new_pos.y);
    let disk_thickness = 0.08 + 0.25 * sqrt(new_r / (params.isco * 4.0));
    let disk_alpha = exp(-disk_height * disk_height / (disk_thickness * disk_thickness * new_r * new_r + 0.01));
    let age_alpha = smoothstep(50.0, 35.0, p.pos.w);
    let inner_boost = smoothstep(params.isco * 5.0, params.isco, new_r);

    p.meta = vec4<f32>(
      new_r,
      redshift,
      doppler,
      disk_alpha * age_alpha * (0.3 + inner_boost * 0.7),
    );
    p.extra.y = time_dilation;
  } else {
    // --- Jet particle physics ---
    let pole_dir = sign(p.pos.y);

    // Jet acceleration along polar axis with magnetic collimation
    let height = abs(p.pos.y);
    let lateral_dist = length(p.pos.xz);

    // Magnetic collimation — push particles towards axis
    let collimation = -normalize(vec3<f32>(p.pos.x, 0.0, p.pos.z)) * 0.02 / (lateral_dist + 0.5);

    // Continued acceleration along jet axis
    let jet_accel = vec3<f32>(0.0, pole_dir * 0.15 / (height + 1.0), 0.0);

    // Precession from spin
    let prec_angle = params.time * a_spin * 0.3;
    let precession = vec3<f32>(
      sin(prec_angle) * 0.005,
      0.0,
      cos(prec_angle) * 0.005,
    );

    let new_vel = p.vel.xyz + (jet_accel + collimation + precession) * dt;
    let new_pos = p.pos.xyz + new_vel * dt;

    p.pos = vec4<f32>(new_pos, p.pos.w + dt);
    p.vel = vec4<f32>(new_vel, p.vel.w * 0.999); // slow cooling

    let new_r = length(new_pos);
    let speed = length(new_vel);

    // Relativistic beaming — jet brightness depends on viewing angle
    let beam_factor = 1.0 / max(1.0 - speed * 0.5, 0.1);

    // Time dilation from velocity (special relativistic)
    let lorentz = 1.0 / sqrt(max(1.0 - speed * speed, 0.01));
    let time_dilation = 1.0 / lorentz;

    // Jet alpha: bright core, fading with distance
    let core_alpha = exp(-lateral_dist * lateral_dist * 8.0);
    let height_fade = exp(-height * 0.04);
    let age_fade = smoothstep(50.0, 40.0, p.pos.w);

    p.meta = vec4<f32>(
      new_r,
      beam_factor,
      speed,
      core_alpha * height_fade * age_fade * 0.6,
    );
    p.extra.y = time_dilation;
  }

  particles[idx] = p;
}
