// Kerr Metric Geodesic Integrator — compiled to WebAssembly
// Implements null geodesic ray tracing in Kerr spacetime for
// accurate gravitational lensing around spinning black holes.
//
// Uses Boyer-Lindquist coordinates and 4th-order Runge-Kutta integration.

const std = @import("std");
const math = std.math;

// Kerr metric parameters (set from JS)
var M: f64 = 1.0; // Black hole mass
var a: f64 = 0.0; // Spin parameter (a = J/M, |a| <= M)

// Boyer-Lindquist metric functions
fn sigma(r: f64, theta: f64) f64 {
    return r * r + a * a * math.cos(theta) * math.cos(theta);
}

fn delta(r: f64) f64 {
    return r * r - 2.0 * M * r + a * a;
}

// State vector: [r, theta, phi, pr, ptheta]
// We use the constants of motion (E, L, Q) to reduce the system

const State = struct {
    r: f64,
    theta: f64,
    phi: f64,
    p_r: f64,
    p_theta: f64,
};

// Compute derivatives for geodesic equations in Kerr spacetime
fn geodesicDerivatives(s: State, energy: f64, lz: f64, carter_q: f64) State {
    const r = s.r;
    const th = s.theta;
    const pr = s.p_r;
    const pth = s.p_theta;

    const sin_th = math.sin(th);
    const cos_th = math.cos(th);
    const sig = sigma(r, th);
    const del = delta(r);

    if (sig < 1e-10 or @abs(sin_th) < 1e-10) {
        return State{ .r = 0, .theta = 0, .phi = 0, .p_r = 0, .p_theta = 0 };
    }

    // dr/dlambda
    const dr = del * pr / sig;

    // dtheta/dlambda
    const dtheta = pth / sig;

    // dphi/dlambda
    const dphi = (a * energy * (r * r + a * a) - a * a * lz + lz / (sin_th * sin_th) * (sig - 2.0 * M * r)) / (sig * del + 1e-20);

    // dp_r/dlambda (radial momentum equation)
    const r2 = r * r;
    const a2 = a * a;
    const ddelta = 2.0 * r - 2.0 * M;
    const dsigma_dr = 2.0 * r;

    const R = (energy * (r2 + a2) - a * lz) * (energy * (r2 + a2) - a * lz) - del * (carter_q + (lz - a * energy) * (lz - a * energy));

    const dR_dr = 4.0 * energy * r * (energy * (r2 + a2) - a * lz) - ddelta * (carter_q + (lz - a * energy) * (lz - a * energy));

    const dp_r = (dR_dr * sig - R * dsigma_dr) / (2.0 * sig * sig + 1e-20);

    // dp_theta/dlambda
    const dsigma_dth = -2.0 * a2 * cos_th * sin_th;
    const theta_pot = carter_q - cos_th * cos_th * (a2 * (1.0 - energy * energy) + lz * lz / (sin_th * sin_th + 1e-20));
    const dtheta_pot = 2.0 * cos_th * sin_th * a2 * (1.0 - energy * energy) + 2.0 * lz * lz * cos_th / (sin_th * sin_th * sin_th + 1e-20);

    const dp_theta = (dtheta_pot * sig - theta_pot * dsigma_dth) / (2.0 * sig * sig + 1e-20);

    return State{
        .r = dr,
        .theta = dtheta,
        .phi = dphi,
        .p_r = dp_r,
        .p_theta = dp_theta,
    };
}

// RK4 integration step
fn rk4Step(s: State, dt: f64, energy: f64, lz: f64, carter_q: f64) State {
    const k1 = geodesicDerivatives(s, energy, lz, carter_q);

    const s2 = State{
        .r = s.r + 0.5 * dt * k1.r,
        .theta = s.theta + 0.5 * dt * k1.theta,
        .phi = s.phi + 0.5 * dt * k1.phi,
        .p_r = s.p_r + 0.5 * dt * k1.p_r,
        .p_theta = s.p_theta + 0.5 * dt * k1.p_theta,
    };
    const k2 = geodesicDerivatives(s2, energy, lz, carter_q);

    const s3 = State{
        .r = s.r + 0.5 * dt * k2.r,
        .theta = s.theta + 0.5 * dt * k2.theta,
        .phi = s.phi + 0.5 * dt * k2.phi,
        .p_r = s.p_r + 0.5 * dt * k2.p_r,
        .p_theta = s.p_theta + 0.5 * dt * k2.p_theta,
    };
    const k3 = geodesicDerivatives(s3, energy, lz, carter_q);

    const s4 = State{
        .r = s.r + dt * k3.r,
        .theta = s.theta + dt * k3.theta,
        .phi = s.phi + dt * k3.phi,
        .p_r = s.p_r + dt * k3.p_r,
        .p_theta = s.p_theta + dt * k3.p_theta,
    };
    const k4 = geodesicDerivatives(s4, energy, lz, carter_q);

    return State{
        .r = s.r + dt / 6.0 * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r),
        .theta = s.theta + dt / 6.0 * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta),
        .phi = s.phi + dt / 6.0 * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi),
        .p_r = s.p_r + dt / 6.0 * (k1.p_r + 2.0 * k2.p_r + 2.0 * k3.p_r + k4.p_r),
        .p_theta = s.p_theta + dt / 6.0 * (k1.p_theta + 2.0 * k2.p_theta + 2.0 * k3.p_theta + k4.p_theta),
    };
}

// ---------- Exported WASM API ----------

// Shared memory buffer for ray results (written by Zig, read by JS)
// Layout per ray: [final_theta, final_phi, absorbed, redshift] (4 f64s)
var ray_results: [4096 * 4]f64 = undefined;
var ray_count: u32 = 0;

export fn getResultsPtr() [*]f64 {
    return &ray_results;
}

export fn setParams(mass: f64, spin: f64) void {
    M = mass;
    a = @min(spin, mass * 0.998); // Enforce a < M
}

// Trace a single photon ray from camera position
// Returns: 0 = escaped (hit sky), 1 = absorbed by black hole, 2 = hit accretion disk
export fn traceRay(
    // Camera position in Boyer-Lindquist-like coords
    cam_r: f64,
    cam_theta: f64,
    cam_phi: f64,
    // Ray direction (local sky angles)
    ray_alpha: f64, // horizontal angle
    ray_beta: f64, // vertical angle
) u32 {
    const r0 = cam_r;
    const th0 = cam_theta;

    // Convert screen angles to conserved quantities
    // For a distant observer, these relate to impact parameters
    const sin_th0 = math.sin(th0);
    const b = ray_alpha * r0 * sin_th0; // angular momentum parameter
    const q_param = ray_beta * ray_beta * r0 * r0 + (b * b - a * a) * math.cos(th0) * math.cos(th0);

    // Conserved quantities (E=1 for null geodesics)
    const energy: f64 = 1.0;
    const lz = b;
    const carter_q = @max(q_param, 0.0);

    // Initial momenta from conserved quantities
    const sig0 = sigma(r0, th0);
    const del0 = delta(r0);

    const R0 = (energy * (r0 * r0 + a * a) - a * lz) * (energy * (r0 * r0 + a * a) - a * lz) - del0 * (carter_q + (lz - a * energy) * (lz - a * energy));

    const pr0 = -math.sqrt(@max(R0, 0.0)) / sig0; // negative = ingoing
    const theta_pot = carter_q - math.cos(th0) * math.cos(th0) * (a * a * (1.0 - energy * energy) + lz * lz / (sin_th0 * sin_th0 + 1e-20));
    const pth0 = if (ray_beta >= 0)
        math.sqrt(@max(theta_pot, 0.0)) / sig0
    else
        -math.sqrt(@max(theta_pot, 0.0)) / sig0;

    var state = State{
        .r = r0,
        .theta = th0,
        .phi = cam_phi,
        .p_r = pr0,
        .p_theta = pth0,
    };

    // Event horizon radius
    const r_horizon = M + math.sqrt(M * M - a * a);

    // Adaptive step integration
    const max_steps: u32 = 800;
    var step: u32 = 0;
    var result: u32 = 0; // escaped

    while (step < max_steps) : (step += 1) {
        // Adaptive step size based on proximity to horizon
        const r_ratio = state.r / r_horizon;
        const dt: f64 = if (r_ratio < 2.0) 0.05 else if (r_ratio < 5.0) 0.2 else 0.5;

        state = rk4Step(state, dt, energy, lz, carter_q);

        // Clamp theta to valid range
        if (state.theta < 0.01) state.theta = 0.01;
        if (state.theta > math.pi - 0.01) state.theta = math.pi - 0.01;

        // Check absorption (inside event horizon)
        if (state.r < r_horizon * 1.01) {
            result = 1;
            break;
        }

        // Check accretion disk crossing (thin disk in equatorial plane)
        if (@abs(state.theta - math.pi / 2.0) < 0.02 and state.r > r_horizon * 1.5 and state.r < 20.0 * M) {
            result = 2;
            break;
        }

        // Escaped to infinity
        if (state.r > 100.0 * M) {
            break;
        }
    }

    // Compute gravitational redshift
    const r_f = state.r;
    const sig_f = sigma(r_f, state.theta);
    const del_f = delta(r_f);
    const redshift = if (del_f > 0)
        math.sqrt(del_f * sig_f) / (sig_f + 2.0 * M * r_f)
    else
        0.0;

    // Store results
    const idx = @as(usize, ray_count) * 4;
    if (idx + 3 < ray_results.len) {
        ray_results[idx] = state.theta;
        ray_results[idx + 1] = state.phi;
        ray_results[idx + 2] = @floatFromInt(result);
        ray_results[idx + 3] = redshift;
        ray_count += 1;
    }

    return result;
}

export fn resetResults() void {
    ray_count = 0;
}

export fn getResultCount() u32 {
    return ray_count;
}

// Batch trace rays for a grid (more efficient than calling traceRay repeatedly)
// Writes directly to shared buffer
export fn traceRayBatch(
    cam_r: f64,
    cam_theta: f64,
    cam_phi: f64,
    // Grid parameters
    alpha_min: f64,
    alpha_max: f64,
    beta_min: f64,
    beta_max: f64,
    grid_w: u32,
    grid_h: u32,
) void {
    resetResults();

    const w_f: f64 = @floatFromInt(grid_w);
    const h_f: f64 = @floatFromInt(grid_h);

    var y: u32 = 0;
    while (y < grid_h) : (y += 1) {
        var x: u32 = 0;
        while (x < grid_w) : (x += 1) {
            const u: f64 = @as(f64, @floatFromInt(x)) / (w_f - 1.0);
            const v: f64 = @as(f64, @floatFromInt(y)) / (h_f - 1.0);

            const alpha = alpha_min + u * (alpha_max - alpha_min);
            const beta = beta_min + v * (beta_max - beta_min);

            _ = traceRay(cam_r, cam_theta, cam_phi, alpha, beta);
        }
    }
}

// Compute ISCO (Innermost Stable Circular Orbit) for Kerr
export fn computeISCO() f64 {
    const a_star = a / M;
    const z1 = 1.0 + math.cbrt(1.0 - a_star * a_star) * (math.cbrt(1.0 + a_star) + math.cbrt(1.0 - a_star));
    const z2 = math.sqrt(3.0 * a_star * a_star + z1 * z1);

    if (a_star >= 0) {
        return M * (3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
    } else {
        return M * (3.0 + z2 + math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
    }
}

// Compute photon sphere radius for Kerr (prograde)
export fn computePhotonSphere() f64 {
    const a_star = a / M;
    return 2.0 * M * (1.0 + math.cos(2.0 / 3.0 * math.acos(-a_star)));
}
