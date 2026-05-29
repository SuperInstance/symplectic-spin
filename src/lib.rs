//! Symplectic integrators: Euler drifts, Verlet conserves.
//! Spin abstracts time as rotation in phase space.

/// A point in phase space (position q, momentum p)
#[derive(Clone)]
pub struct PhasePoint {
    pub q: Vec<f64>,
    pub p: Vec<f64>,
}

impl PhasePoint {
    pub fn new(q: Vec<f64>, p: Vec<f64>) -> Self { Self { q, p } }
    pub fn zero(dim: usize) -> Self { Self { q: vec![0.0; dim], p: vec![0.0; dim] } }
}

/// A separable Hamiltonian H(q,p) = V(q) + T(p)
/// Uses function pointers for simplicity (no closures)
pub struct Hamiltonian {
    pub potential: fn(&[f64]) -> f64,
    pub kinetic: fn(&[f64]) -> f64,
    pub grad_potential: fn(&[f64], &mut [f64]),
    pub grad_kinetic: fn(&[f64], &mut [f64]),
    pub dim: usize,
}

/// Harmonic oscillator V(q) = 0.5 * omega^2 * sum(q^2)
pub fn harmonic_potential(q: &[f64]) -> f64 {
    0.5 * q.iter().map(|x| x * x).sum::<f64>()
}

/// Harmonic oscillator gradient dV/dq = q (omega=1)
pub fn harmonic_grad_potential(q: &[f64], out: &mut [f64]) {
    out.copy_from_slice(q);
}

/// Standard kinetic T(p) = sum(p^2)/2
pub fn standard_kinetic(p: &[f64]) -> f64 {
    p.iter().map(|x| x * x).sum::<f64>() / 2.0
}

/// Standard kinetic gradient dT/dp = p
pub fn standard_grad_kinetic(p: &[f64], out: &mut [f64]) {
    out.copy_from_slice(p);
}

/// Create a 1D harmonic oscillator (omega=1, mass=1)
pub fn harmonic_oscillator() -> Hamiltonian {
    Hamiltonian {
        potential: harmonic_potential,
        kinetic: standard_kinetic,
        grad_potential: harmonic_grad_potential,
        grad_kinetic: standard_grad_kinetic,
        dim: 1,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IntegratorKind { Euler, SymplecticEuler, Verlet, Yoshida4 }

#[derive(Debug)]
pub struct DriftReport {
    pub initial_energy: f64,
    pub final_energy: f64,
    pub max_drift: f64,
    pub rms_drift: f64,
    pub energy_oscillation: f64,
}

/// Standard Euler step — translates, DRIFTS
pub fn euler_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    let mut dq = vec![0.0; h.dim];
    let mut dp = vec![0.0; h.dim];
    (h.grad_kinetic)(&state.p, &mut dq);
    (h.grad_potential)(&state.q, &mut dp);
    let mut q = state.q.clone();
    let mut p = state.p.clone();
    for i in 0..h.dim {
        q[i] += dt * dq[i];
        p[i] -= dt * dp[i];
    }
    PhasePoint::new(q, p)
}

/// Symplectic Euler — rotates, CONSERVES (update p first, then q with NEW p)
pub fn symplectic_euler_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    let mut dp = vec![0.0; h.dim];
    (h.grad_potential)(&state.q, &mut dp);
    let mut p = state.p.clone();
    for i in 0..h.dim { p[i] -= dt * dp[i]; }
    let mut dq = vec![0.0; h.dim];
    (h.grad_kinetic)(&p, &mut dq);
    let mut q = state.q.clone();
    for i in 0..h.dim { q[i] += dt * dq[i]; }
    PhasePoint::new(q, p)
}

/// Störmer-Verlet (leapfrog) — 2nd order symplectic, gold standard
pub fn verlet_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    let half = dt / 2.0;
    let mut dp = vec![0.0; h.dim];
    let mut dq = vec![0.0; h.dim];
    
    (h.grad_potential)(&state.q, &mut dp);
    let mut p = state.p.clone();
    for i in 0..h.dim { p[i] -= half * dp[i]; }
    
    (h.grad_kinetic)(&p, &mut dq);
    let mut q = state.q.clone();
    for i in 0..h.dim { q[i] += dt * dq[i]; }
    
    (h.grad_potential)(&q, &mut dp);
    for i in 0..h.dim { p[i] -= half * dp[i]; }
    PhasePoint::new(q, p)
}

/// Yoshida 4th order symplectic
pub fn yoshida4_step(h: &Hamiltonian, state: &PhasePoint, dt: f64) -> PhasePoint {
    let w1: f64 = 1.0 / (2.0 - 2.0_f64.powf(1.0/3.0));
    let w0: f64 = 1.0 - 2.0 * w1;
    
    let c1 = w1 / 2.0;
    let c2 = (w0 + w1) / 2.0;
    let c3 = c2;
    let c4 = c1;
    let d1 = w1;
    let d2 = w0;
    let d3 = w1;

    let mut s = state.clone();
    let mut buf = vec![0.0; h.dim];
    
    (h.grad_kinetic)(&s.p, &mut buf);
    for i in 0..h.dim { s.q[i] += c1 * dt * buf[i]; }
    (h.grad_potential)(&s.q, &mut buf);
    for i in 0..h.dim { s.p[i] -= d1 * dt * buf[i]; }
    
    (h.grad_kinetic)(&s.p, &mut buf);
    for i in 0..h.dim { s.q[i] += c2 * dt * buf[i]; }
    (h.grad_potential)(&s.q, &mut buf);
    for i in 0..h.dim { s.p[i] -= d2 * dt * buf[i]; }
    
    (h.grad_kinetic)(&s.p, &mut buf);
    for i in 0..h.dim { s.q[i] += c3 * dt * buf[i]; }
    (h.grad_potential)(&s.q, &mut buf);
    for i in 0..h.dim { s.p[i] -= d3 * dt * buf[i]; }
    
    (h.grad_kinetic)(&s.p, &mut buf);
    for i in 0..h.dim { s.q[i] += c4 * dt * buf[i]; }
    s
}

/// Integrate for n steps and return drift report
pub fn conservation_drift(
    h: &Hamiltonian,
    initial: &PhasePoint,
    dt: f64,
    steps: usize,
    kind: IntegratorKind,
) -> DriftReport {
    let initial_energy = (h.kinetic)(&initial.p) + (h.potential)(&initial.q);
    let mut state = initial.clone();
    let mut max_drift = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_e = initial_energy;
    let mut min_e = initial_energy;

    let step_fn = match kind {
        IntegratorKind::Euler => euler_step,
        IntegratorKind::SymplecticEuler => symplectic_euler_step,
        IntegratorKind::Verlet => verlet_step,
        IntegratorKind::Yoshida4 => yoshida4_step,
    };

    for _ in 0..steps {
        state = step_fn(h, &state, dt);
        let e = (h.kinetic)(&state.p) + (h.potential)(&state.q);
        let drift = (e - initial_energy).abs();
        if drift > max_drift { max_drift = drift; }
        sum_sq += drift * drift;
        if e > max_e { max_e = e; }
        if e < min_e { min_e = e; }
    }

    DriftReport {
        initial_energy,
        final_energy: (h.kinetic)(&state.p) + (h.potential)(&state.q),
        max_drift,
        rms_drift: (sum_sq / steps as f64).sqrt(),
        energy_oscillation: max_e - min_e,
    }
}

/// Spin frequencies from graph eigenvalues: ω_i = √λ_i
pub fn spin_frequencies(adj: &[Vec<f64>]) -> Vec<f64> {
    let n = adj.len();
    // Build Laplacian
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let mut deg = 0.0;
        for j in 0..n { deg += adj[i][j]; }
        for j in 0..n { a[i][j] = if i == j { deg } else { -adj[i][j] }; }
    }
    let eigenvalues = jacobi_eigenvalues(&mut a);
    eigenvalues.into_iter().map(|l| if l > 0.0 { l.sqrt() } else { 0.0 }).collect()
}

fn jacobi_eigenvalues(a: &mut Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 * n * n {
        let (mut p, mut q) = (0, 1);
        let mut max_val = 0.0_f64;
        for i in 0..n { for j in (i+1)..n { if a[i][j].abs() > max_val { max_val = a[i][j].abs(); p = i; q = j; } } }
        if max_val < 1e-14 { break; }
        let app = a[p][p]; let aqq = a[q][q]; let apq = a[p][q];
        let theta = if (app - aqq).abs() < 1e-30 { std::f64::consts::FRAC_PI_4 }
                     else { 0.5 * (2.0 * apq / (app - aqq)).atan() };
        let (c, s) = (theta.cos(), theta.sin());
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p]; let aiq = a[i][q];
                a[i][p] = c * aip + s * aiq; a[p][i] = a[i][p];
                a[i][q] = -s * aip + c * aiq; a[q][i] = a[i][q];
            }
        }
        a[p][p] = c*c*app + 2.0*s*c*apq + s*s*aqq;
        a[q][q] = s*s*app - 2.0*s*c*apq + c*c*aqq;
        a[p][q] = 0.0; a[q][p] = 0.0;
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_drifts() {
        let h = harmonic_oscillator();
        let init = PhasePoint::new(vec![1.0], vec![0.0]);
        let report = conservation_drift(&h, &init, 0.01, 10000, IntegratorKind::Euler);
        assert!(report.max_drift > 0.01, "Euler should drift: got {}", report.max_drift);
    }

    #[test]
    fn symplectic_euler_conserves() {
        let h = harmonic_oscillator();
        let init = PhasePoint::new(vec![1.0], vec![0.0]);
        let report = conservation_drift(&h, &init, 0.01, 10000, IntegratorKind::SymplecticEuler);
        assert!(report.energy_oscillation < 0.01, "Symplectic oscillation: {}", report.energy_oscillation);
    }

    #[test]
    fn verlet_conserves() {
        let h = harmonic_oscillator();
        let init = PhasePoint::new(vec![1.0], vec![0.0]);
        let report = conservation_drift(&h, &init, 0.01, 10000, IntegratorKind::Verlet);
        assert!(report.max_drift < 1e-3, "Verlet drift: {}", report.max_drift);
    }

    #[test]
    fn yoshida_conserves() {
        let h = harmonic_oscillator();
        let init = PhasePoint::new(vec![1.0], vec![0.0]);
        let report = conservation_drift(&h, &init, 0.01, 10000, IntegratorKind::Yoshida4);
        assert!(report.max_drift < 1e-9, "Yoshida drift: {}", report.max_drift);
    }

    #[test]
    fn spin_frequencies_correct() {
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let freqs = spin_frequencies(&adj);
        assert_eq!(freqs.len(), 2);
        assert!(freqs[0].abs() < 1e-10, "First freq ~0");
        assert!((freqs[1] - 2.0_f64.sqrt()).abs() < 1e-10, "Second freq √2");
    }
}
