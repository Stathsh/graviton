// Particle physics — dense accretion disk with spiral arms + dramatic jets

struct Params {
  mass: f32,
  spin: f32,
  dt: f32,
  accretion_rate: f32,
  time: f32,
  particle_count: u32,
  isco: f32,
  photon_r: f32,
  star_time: f32,
  star_start: f32,
  star_count: f32,
  _pad: f32,
}

struct Particle {
  pos: vec4<f32>,    // xyz, w=age
  vel: vec4<f32>,    // xyz, w=temperature
  meta: vec4<f32>,   // x=r, y=redshift, z=doppler, w=alpha
  extra: vec4<f32>,  // x=type(0=disk,1=jet), y=time_dilation, z=spiral_phase, w=turbulence
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn pcg(n: u32) -> u32 {
  var s = n * 747796405u + 2891336453u;
  var w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand(id: u32, salt: u32) -> f32 {
  return f32(pcg(id * 1099087573u + salt)) / 4294967295.0;
}

fn kerr_accel(pos: vec3<f32>) -> vec3<f32> {
  let r = length(pos);
  let M = params.mass;
  let a = params.spin;
  let rs = 2.0 * M;
  let r_h = M + sqrt(max(M * M - a * a, 0.0));
  if (r < r_h * 0.7) { return vec3(0.0); }

  let r_hat = pos / r;
  let pn = 1.0 + 1.5 * rs / r + 2.0 * rs * rs / (r * r);
  var acc = -M / (r * r) * r_hat * pn;

  // Frame-dragging
  let omega = 2.0 * M * a / (r * r * r);
  let drag = vec3(-pos.z, 0.0, pos.x);
  acc += omega * normalize(drag) * length(pos.xz);

  return acc;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.particle_count) { return; }

  var p = particles[idx];
  let dt = params.dt;
  let M = params.mass;
  let a = params.spin;
  let rs = 2.0 * M;
  let r_h = M + sqrt(max(M * M - a * a, 0.0));
  let seed = bitcast<u32>(params.time * 1000.0) + idx;

  let r = length(p.pos.xyz);
  let is_star = p.extra.x > 1.5;
  let dead = (r < r_h * 0.8 || r > 120.0 || p.pos.w > 60.0 || r != r) && !is_star;

  // 12% jet particles
  let is_jet = idx >= params.particle_count - params.particle_count / 8u;

  if (dead) {
    if (is_jet) {
      let pole = select(-1.0, 1.0, rand(idx, seed + 99u) > 0.5);
      let ang = rand(idx, seed) * 6.283;
      let spread = rand(idx, seed + 1u) * 0.25;
      let base_r = r_h * (1.0 + rand(idx, seed + 2u) * 0.4);

      p.pos = vec4(
        cos(ang) * spread * base_r * 0.25,
        pole * base_r * 0.3,
        sin(ang) * spread * base_r * 0.25,
        0.0,
      );

      let speed = 0.35 + rand(idx, seed + 3u) * 0.55;
      p.vel = vec4(
        (rand(idx, seed + 4u) - 0.5) * 0.06,
        pole * speed,
        (rand(idx, seed + 5u) - 0.5) * 0.06,
        40000.0 + rand(idx, seed + 6u) * 120000.0,
      );
      p.extra = vec4(1.0, 1.0, rand(idx, seed + 7u) * 6.283, 0.0);
    } else {
      // Dense disk with spiral structure
      let ang = rand(idx, seed) * 6.283;

      // Bias radius towards ISCO (denser inner disk)
      let u_r = rand(idx, seed + 1u);
      let radius = params.isco * (1.0 + pow(u_r, 0.5) * 8.0 * params.accretion_rate);

      // Spiral arm offset
      let spiral_phase = ang + radius * 0.15 - params.time * 0.2;

      // Volumetric height — thinner near ISCO, thicker further out
      let scale_h = 0.01 + 0.06 * pow(radius / (params.isco * 5.0), 1.5);
      let h = (rand(idx, seed + 2u) - 0.5) * scale_h * radius;
      let turb = 0.3 + rand(idx, seed + 8u) * 0.7;

      p.pos = vec4(cos(ang) * radius, h, sin(ang) * radius, 0.0);

      let v_k = sqrt(M / radius);
      let kerr_boost = 1.0 + a * sqrt(M) / (radius * sqrt(radius));
      let tangent = vec3(-sin(ang), 0.0, cos(ang));
      let drift = normalize(p.pos.xyz) * (-0.004 * params.accretion_rate);
      let turb_v = vec3(
        (rand(idx, seed + 9u) - 0.5) * 0.015 * turb,
        (rand(idx, seed + 10u) - 0.5) * 0.005 * turb,
        (rand(idx, seed + 11u) - 0.5) * 0.015 * turb,
      );

      p.vel = vec4(tangent * v_k * kerr_boost + drift + turb_v, 3000.0 + rand(idx, seed + 3u) * 17000.0);
      p.extra = vec4(0.0, 1.0, spiral_phase, turb);
    }
    p.meta = vec4(0.0, 1.0, 1.0, 0.0);
  }

  if (is_star) {
    // --- Star (tidal disruption) ---
    let star_dead = r < r_h * 0.8 || r > 200.0 || p.pos.w > 120.0 || r != r;
    if (star_dead) {
      // Hide dead star particles far away with zero alpha
      p.meta = vec4(r, 1.0, 1.0, 0.0);
      p.pos = vec4(9999.0, 9999.0, 9999.0, p.pos.w);
      particles[idx] = p;
      return;
    }

    // Pure gravitational physics — Verlet integration
    let acc1 = kerr_accel(p.pos.xyz);
    let np = p.pos.xyz + p.vel.xyz * dt + 0.5 * acc1 * dt * dt;
    let acc2 = kerr_accel(np);
    let nv = p.vel.xyz + 0.5 * (acc1 + acc2) * dt;

    p.pos = vec4(np, p.pos.w + dt);
    p.vel = vec4(nv, p.vel.w);

    let nr = length(np);

    // Tidal heating: temperature increases as particle approaches BH
    let tidal_strength = pow(M / max(nr, r_h), 3.0);
    let heat_rate = tidal_strength * 800.0;
    p.vel.w = min(p.vel.w + heat_rate * dt, 80000.0);

    // Spaghettification: stretch factor based on tidal gradient
    let stretch = 1.0 + tidal_strength * 2.0;

    // Gravitational redshift
    let grav_redshift = sqrt(max(1.0 - rs / nr, 0.01));

    // Alpha: fade very old or very far particles
    let age_fade = smoothstep(120.0, 80.0, p.pos.w);
    let dist_fade = smoothstep(180.0, 100.0, nr);
    let inner_bright = smoothstep(params.isco * 8.0, params.isco, nr);
    let alpha = age_fade * dist_fade * (0.4 + inner_bright * 0.6);

    // meta: x=r, y=stretch, z=grav_redshift, w=alpha
    p.meta = vec4(nr, stretch, grav_redshift, alpha);
    p.extra.y = sqrt(max(1.0 - rs / nr, 0.001)); // time dilation
  } else if (p.extra.x < 0.5) {
    // --- Disk ---
    let acc1 = kerr_accel(p.pos.xyz);
    let np = p.pos.xyz + p.vel.xyz * dt + 0.5 * acc1 * dt * dt;
    let acc2 = kerr_accel(np);
    let nv = p.vel.xyz + 0.5 * (acc1 + acc2) * dt;

    p.pos = vec4(np, p.pos.w + dt);
    p.vel = vec4(nv, p.vel.w);

    let nr = length(np);
    let redshift = 1.0 / sqrt(max(1.0 - rs / nr, 0.01));
    let rv = dot(normalize(np), nv);
    let doppler = sqrt(max((1.0 - rv) / (1.0 + rv + 0.001), 0.01));
    let td = sqrt(max(1.0 - rs / nr, 0.001));

    // Temperature: hotter near ISCO
    let t_boost = pow(params.isco / max(nr, params.isco), 0.75);
    p.vel.w = mix(p.vel.w, p.vel.w * t_boost, 0.015);

    // Spiral arm modulation
    let phi = atan2(np.z, np.x);
    let spiral = sin(phi * 3.0 - nr * 0.2 + p.extra.z) * 0.3 + 0.7;
    let spiral2 = sin(phi * 7.0 + nr * 0.4 - params.time * 0.3) * 0.15 + 0.85;

    // Disk density envelope
    let disk_h = abs(np.y);
    let thickness = 0.01 + 0.06 * pow(nr / (params.isco * 5.0), 1.5);
    let envelope = exp(-disk_h * disk_h / (thickness * thickness * nr * nr + 0.001));
    let age_fade = smoothstep(60.0, 40.0, p.pos.w);
    let inner_glow = smoothstep(params.isco * 5.0, params.isco, nr);

    let alpha = envelope * age_fade * spiral * spiral2 * (0.2 + inner_glow * 0.8);

    p.meta = vec4(nr, redshift, doppler, alpha);
    p.extra.y = td;
  } else {
    // --- Jet ---
    let pole = sign(p.pos.y);
    let height = abs(p.pos.y);
    let lat = length(p.pos.xz);

    // Magnetic collimation + acceleration
    let collimate = -normalize(vec3(p.pos.x, 0.0, p.pos.z)) * 0.03 / (lat + 0.3);
    let boost = vec3(0.0, pole * 0.2 / (height + 0.5), 0.0);

    // Helical twist from BH spin
    let twist_rate = a * 0.5 / (height + 1.0);
    let twist = vec3(
      -sin(params.time * twist_rate + height * 0.3) * 0.01,
      0.0,
      cos(params.time * twist_rate + height * 0.3) * 0.01,
    );

    let nv = p.vel.xyz + (boost + collimate + twist) * dt;
    let np = p.pos.xyz + nv * dt;
    p.pos = vec4(np, p.pos.w + dt);
    p.vel = vec4(nv, p.vel.w * 0.9985);

    let speed = length(nv);
    let beam = 1.0 / max(1.0 - speed * 0.6, 0.05);
    let lorentz = 1.0 / sqrt(max(1.0 - speed * speed, 0.01));

    let core = exp(-lat * lat * 12.0);
    let h_fade = exp(-height * 0.03);
    let age = smoothstep(60.0, 45.0, p.pos.w);

    p.meta = vec4(length(np), beam, speed, core * h_fade * age * 0.7);
    p.extra.y = 1.0 / lorentz;
  }

  particles[idx] = p;
}
