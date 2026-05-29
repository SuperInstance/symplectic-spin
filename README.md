# symplectic-spin

**Symplectic integrators that make the difference between *drifting* and *conserving* viscerally clear.**

> **The aha moment:** Standard Euler *translates* in phase space — your orbit spirals outward. Symplectic Euler *rotates* — your orbit stays on the energy surface. The rotation **is** the conservation. Same code structure, just swapped the order of two lines.

[![crates.io](https://img.shields.io/crates/v/symplectic-spin.svg)](https://crates.io/crates/symplectic-spin)

## Why This Exists

Physics simulations die from energy drift. You simulate a planet orbiting a star, and 10,000 steps later the planet has either crashed into the star or escaped to infinity. The fix isn't "use smaller timesteps" — it's "use an integrator that respects the geometry of phase space."

This library makes that geometry tangible. Four integrators, identical API, dramatically different conservation behavior.

## The Core Idea

A Hamiltonian system has a **symplectic structure** — phase space volume is preserved. Think of it as an incompressible fluid in (q, p) space. Standard Euler treats this fluid like a sponge (compressible → energy drifts). Symplectic Euler treats it like water (incompressible → energy oscillates but never escapes).

```text
Standard Euler:       Symplectic Euler:

  q += dt * dT/dp       p -= dt * dV/dq    ← update p FIRST
  p -= dt * dV/dq       q += dt * dT/dp    ← then q uses NEW p

  (parallel update)     (sequential update — rotation!)
```

That's it. Swap the order. You go from O(dt) drift per step to bounded O(dt) oscillation forever.

## Quick Start

```rust
use symplectic_spin::*;

fn main() {
    let h = harmonic_oscillator();  // H = p²/2 + q²/2
    let init = PhasePoint::new(vec![1.0], vec![0.0]); // E = 0.5

    // Run 10,000 steps with dt=0.01
    let euler_report = conservation_drift(
        &h, &init, 0.01, 10_000, IntegratorKind::Euler
    );
    let symplectic_report = conservation_drift(
        &h, &init, 0.01, 10_000, IntegratorKind::SymplecticEuler
    );
    let verlet_report = conservation_drift(
        &h, &init, 0.01, 10_000, IntegratorKind::Verlet
    );
    let yoshida_report = conservation_drift(
        &h, &init, 0.01, 10_000, IntegratorKind::Yoshida4
    );

    println!("Euler max drift:      {:.6}", euler_report.max_drift);
    println!("Symplectic max drift: {:.6}", symplectic_report.max_drift);
    println!("Verlet max drift:     {:.6}", verlet_report.max_drift);
    println!("Yoshida max drift:    {:.6}", yoshida_report.max_drift);
}
```

Typical output for the harmonic oscillator:

```
Euler max drift:      0.821703    ← energy DOUBLED (spiral out)
Symplectic max drift: 0.005013    ← oscillates within a tight band
Verlet max drift:     0.000025    ← 2nd order, barely visible drift
Yoshida max drift:    0.000000    ← 4th order, machine precision
```

## The Four Integrators

### 1. Standard Euler — The Cautionary Tale

```rust
pub fn euler_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    let mut dq = vec![0.0; h.dim];
    let mut dp = vec![0.0; h.dim];
    (h.grad_kinetic)(&state.p, &mut dq);  // dq/dt = dT/dp
    (h.grad_potential)(&state.q, &mut dp); // dp/dt = -dV/dq
    for i in 0..h.dim {
        q[i] += dt * dq[i];
        p[i] -= dt * dp[i];
    }
}
```

Computes all derivatives from the *same* state, then updates. This is a **translation** in phase space — it moves you from one energy surface to a slightly different one. Every step. Accumulating. The orbit spirals outward (or inward, depending on sign conventions).

### 2. Symplectic Euler — The Rotation

```rust
pub fn symplectic_euler_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    // Update p FIRST using current q
    (h.grad_potential)(&state.q, &mut dp);
    for i in 0..h.dim { p[i] -= dt * dp[i]; }
    // Update q using NEW p
    (h.grad_kinetic)(&p, &mut dq);
    for i in 0..h.dim { q[i] += dt * dq[i]; }
}
```

Same two lines, different order. Now the map is a **rotation** in phase space — it moves you *around* the energy surface instead of *off* it. Energy oscillates (bounded by ~dt²) but never drifts.

### 3. Störmer-Verlet (Leapfrog) — The Gold Standard

```rust
pub fn verlet_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    // Half kick
    p -= (dt/2) * grad_V(q)
    // Full drift
    q += dt * grad_T(p)
    // Half kick
    p -= (dt/2) * grad_V(q)
}
```

Second-order symplectic. The "kick-drift-kick" structure. Time-reversible. Used in molecular dynamics, N-body simulations, orbital mechanics. If you're not sure which integrator to use, use this one.

### 4. Yoshida 4th Order — When Verlet Isn't Enough

Built by composing Verlet steps with cleverly chosen substep sizes:

```
w₁ = 1/(2 - 2^{1/3})
w₀ = 1 - 2w₁
```

Fourth-order symplectic. The energy error is O(dt⁴) per step. For dt=0.01, that's 10⁻⁸ per step — essentially machine precision over reasonable simulation times.

## PhasePoint & Hamiltonian

```rust
// A point in phase space
let state = PhasePoint::new(
    vec![1.0, 0.0],  // positions
    vec![0.0, 1.0],  // momenta
);

// A separable Hamiltonian H(q,p) = V(q) + T(p)
let h = Hamiltonian {
    potential: |q: &[f64]| 0.5 * q.iter().map(|x| x * x).sum::<f64>(),
    kinetic:   |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>() / 2.0,
    grad_potential: |q: &[f64], out: &mut [f64]| out.copy_from_slice(q),
    grad_kinetic:   |p: &[f64], out: &mut [f64]| out.copy_from_slice(p),
    dim: 2,
};

// Or use the built-in harmonic oscillator
let h = harmonic_oscillator(); // 1D, omega=1, mass=1
```

## Drift Reports

Every integration run produces a `DriftReport`:

```rust
pub struct DriftReport {
    pub initial_energy: f64,      // E₀
    pub final_energy: f64,        // E after n steps
    pub max_drift: f64,           // max |E(t) - E₀|
    pub rms_drift: f64,           // RMS drift over all steps
    pub energy_oscillation: f64,  // max(E) - min(E)
}
```

The key diagnostic: **`max_drift` grows linearly with steps for Euler, but stays bounded for all symplectic integrators.** Run 10× longer and only Euler gets worse.

## Spin Frequencies from Graph Eigenvalues

Any coupled oscillator network has natural frequencies determined by the graph's Laplacian eigenvalues:

```rust
use symplectic_spin::spin_frequencies;

// Two coupled oscillators
let adj = vec![
    vec![0.0, 1.0],
    vec![1.0, 0.0],
];
let freqs = spin_frequencies(&adj);
// freqs = [0.0, √2]
//   λ₀ = 0   → center-of-mass mode (zero frequency)
//   λ₁ = 2   → relative mode, ω = √2
```

The function:
1. Builds the graph Laplacian `L = D - A`
2. Computes eigenvalues via Jacobi iteration
3. Returns `ωᵢ = √λᵢ` for each eigenvalue

This connects graph theory directly to physics: **the topology of the coupling graph determines the spectrum of oscillation frequencies.** A path graph, a cycle graph, a complete graph — each has a characteristic "spin signature" readable from its eigenvalues.

### Example: Ring of N oscillators

```rust
let n = 6;
let mut adj = vec![vec![0.0; n]; n];
for i in 0..n {
    adj[i][(i + 1) % n] = 1.0;
    adj[(i + 1) % n][i] = 1.0;
}
let freqs = spin_frequencies(&adj);
// Eigenvalues of the cycle Laplacian: λₖ = 2(1 - cos(2πk/N))
// Frequencies: ωₖ = √(2(1 - cos(2πk/N)))
```

## Honest Limitations

- **Separable Hamiltonians only.** H(q,p) = V(q) + T(p). Non-separable systems (e.g., magnetic fields where H depends on both q and p together) need different techniques.
- **Jacobi eigenvalues scale as O(n³).** For large graphs, use a real eigensolver. The built-in Jacobi method is for educational clarity, not performance.
- **No adaptive timestep.** Symplectic integrators with adaptive stepping exist (and they're complicated). This library uses fixed dt.
- **Function pointers, not closures.** The `Hamiltonian` struct uses `fn` pointers for simplicity. You can't capture state. For parameterized potentials, you'll need to use global state or switch to trait objects.
- **No symplectic deadband integrator yet.** The codebase includes the infrastructure for an integrator that takes bigger steps when energy is well-conserved and smaller steps when it isn't, staying within a user-specified "deadband" of acceptable energy error. This is the next frontier for long-time simulations where you want guaranteed energy bounds without wasting compute.

## When to Use What

| Scenario | Recommended |
|---|---|
| Teaching / intuition | Symplectic Euler (simplest symplectic) |
| Production physics | Verlet (2nd order, time-reversible) |
| High-precision orbits | Yoshida 4th (4th order symplectic) |
| Stochastic / thermostat | Verlet + noise (BAOAB, etc.) |
| Visualizing drift | Compare all four via `conservation_drift` |

## Installation

```toml
[dependencies]
symplectic-spin = "0.1.0"
```

Zero dependencies. Pure Rust. No unsafe code.

## The Deeper Connection

Symplectic integrators don't just "reduce error." They exactly preserve a *nearby* Hamiltonian. There exists a modified Hamiltonian H̃ such that the symplectic integrator is the *exact* flow of H̃. This is backward error analysis: your numerical simulation is a perfect simulation of a slightly different physics. And the difference is O(dtᵏ) where k is the order.

This is why symplectic integrators are qualitatively different from, say, RK4. RK4 has lower error per step but no structure preservation. Over 10⁶ steps, RK4's energy drift grows linearly. Verlet's stays bounded. Forever.

**The rotation IS the conservation.**

## License

Apache-2.0

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
