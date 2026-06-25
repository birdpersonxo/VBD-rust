// main.rs — port of main.cpp
// Implements SimulatorParams, StrandSim, and the entry point.
//
// Dependencies (Cargo.toml):
//   nalgebra  = { version = "0.32", features = ["serde-serialize"] }
//   serde     = { version = "1",    features = ["derive"] }
//   serde_json = "1"

mod strand;
mod types;

use std::{fs, io::Write, path::Path};

use nalgebra::Vector3;
use strand::VBDStrand;
use types::*;

// ---------------------------------------------------------------------------
// SimulatorParams
// ---------------------------------------------------------------------------
pub struct SimulatorParams {
    pub num_frames: usize,
    pub substeps: usize,
    pub num_iterations: usize,
    pub dt: FloatingType,
    pub acceleration_rho: FloatingType,
    pub use_acceleration: bool,
    pub gravity: Vec3,
    pub out_path: String,
}

impl Default for SimulatorParams {
    fn default() -> Self {
        Self {
            num_frames: 300,
            substeps: 1,
            num_iterations: 100,
            dt: 0.016_666_66,
            acceleration_rho: 0.5,
            use_acceleration: false,
            gravity: Vector3::new(0.0, -10.0, 0.0),
            out_path: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// StrandSim
// ---------------------------------------------------------------------------
pub struct StrandSim {
    pub strand: VBDStrand,
    pub params: SimulatorParams,
    pub out_path: String,
    pub frame_id: usize,
    pub step: usize,
    pub iter: usize,
}

impl StrandSim {
    pub fn new() -> Self {
        Self {
            strand: VBDStrand::new_empty(),
            params: SimulatorParams::default(),
            out_path: String::new(),
            frame_id: 0,
            step: 0,
            iter: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Scene initialisation helpers
    // -----------------------------------------------------------------------

    /// Three-vertex horizontal strand (mirrors initializeHorizontal).
    pub fn initialize_horizontal(&mut self) {
        let num_verts = 3usize;
        let dis = 0.1_f32;
        let init_height = 2.0_f32;
        let stiffness = 1e6_f32;
        let m0 = 1.0_f32;
        let m1 = 1000.0_f32;

        self.params.out_path = "output/Test4_20Verts_30degree_withSkip_stiffness1e8".to_string();

        let mut pos: Vec<Vec3> = Vec::new();
        let mut masses: Vec<FloatingType> = Vec::new();
        let mut ls: Vec<FloatingType> = Vec::new();

        for iv in 0..num_verts {
            pos.push(Vector3::new(iv as f32 * dis, init_height, 0.0));
            if iv < num_verts - 1 {
                masses.push(m0);
            } else {
                masses.push(m1);
            }
            if iv > 0 {
                ls.push(dis);
            }
        }

        self.strand
            .from_uniform(&pos, &masses, &ls, stiffness, false, 1e2);
    }

    /// Twenty-vertex tilted strand with optional skip springs (mirrors initializeTilted).
    pub fn initialize_tilted(&mut self) {
        let num_verts = 20usize;
        let dis = 0.05_f32;
        let init_height = 2.0_f32;
        let stiffness = 1e8_f32;
        let m0 = 1.0_f32;
        let m1 = 1000.0_f32;
        let add_skip_spring = true;
        let skip_spring_strength = 100.0_f32;
        let tan_angle = 0.577_35_f32; // tan(30°)

        self.params.out_path = "output/Test4_20Verts_30degree_withSkip_stiffness1e8".to_string();
        self.params.use_acceleration = false;
        self.params.acceleration_rho = 0.5;

        let mut pos: Vec<Vec3> = Vec::new();
        let mut masses: Vec<FloatingType> = Vec::new();
        let mut ls: Vec<FloatingType> = Vec::new();

        for iv in 0..num_verts {
            if iv < num_verts - 1 {
                masses.push(m0);
                pos.push(Vector3::new(
                    iv as f32 * dis,
                    init_height + iv as f32 * dis * tan_angle,
                    0.0,
                ));
            } else {
                masses.push(m1);
                pos.push(Vector3::new(
                    (iv + 1) as f32 * dis,
                    init_height + (iv + 1) as f32 * dis * tan_angle,
                    0.0,
                ));
            }
            if iv > 0 {
                let l = (pos[iv] - pos[iv - 1]).norm();
                ls.push(l);
            }
        }

        self.strand.from_uniform(
            &pos,
            &masses,
            &ls,
            stiffness,
            add_skip_spring,
            skip_spring_strength,
        );
    }

    /// Five-vertex mixed-stiffness strand (mirrors initializeStiffRatio).
    pub fn initialize_stiff_ratio(&mut self) {
        let num_verts = 5usize;
        let dis = 0.05_f32;
        let init_height = 2.0_f32;
        let stiffness1 = 1e4_f32;
        let stiffness2 = 1e8_f32;
        let m0 = 0.1_f32;
        let m1 = 0.1_f32;
        let tan_angle = 0.577_35_f32;

        self.params.num_iterations = 100;
        self.params.acceleration_rho = 0.0;
        self.params.out_path = "output/Test8_5Verts_stiffnessRatio48_accelerated".to_string();

        let mut pos: Vec<Vec3> = Vec::new();
        let mut masses: Vec<FloatingType> = Vec::new();
        let mut ls: Vec<FloatingType> = Vec::new();
        let mut stiffnesses: Vec<FloatingType> = Vec::new();

        for iv in 0..num_verts {
            let p = Vector3::new(
                iv as f32 * dis,
                init_height + iv as f32 * dis * tan_angle,
                0.0,
            );
            if iv < num_verts - 1 {
                masses.push(m0);
                stiffnesses.push(if iv % 2 == 1 { stiffness2 } else { stiffness1 });
            } else {
                masses.push(m1);
            }
            pos.push(p);
            if iv > 0 {
                let l = (pos[iv] - pos[iv - 1]).norm();
                ls.push(l);
            }
        }

        self.strand.from_per_edge(&pos, &masses, &ls, stiffnesses);
    }

    /// Top-level initialise — picks a scene and creates the output directory.
    pub fn initialize(&mut self) {
        self.initialize_tilted();
        // Alternatively: self.initialize_stiff_ratio();

        self.out_path = self.params.out_path.clone();
        fs::create_dir_all(&self.out_path).expect("Failed to create output directory");
    }

    // -----------------------------------------------------------------------
    // Chebyshev accelerator
    // -----------------------------------------------------------------------

    /// Compute the Chebyshev relaxation factor ω for a given iteration order.
    pub fn get_accelerator_omega(
        order: usize,
        rho: FloatingType,
        prev_omega: FloatingType,
    ) -> FloatingType {
        match order {
            1 => 1.0,
            2 => 2.0 / (2.0 - rho * rho),
            _ => 4.0 / (4.0 - rho * rho * prev_omega),
        }
    }

    /// Apply the Chebyshev correction to the vertex positions.
    pub fn apply_accelerator(&mut self, omega: FloatingType) {
        if omega > 1.0 {
            // x_new = omega * (x - x_pp) + x_pp
            let prevprev = self.strand.prevprev_pos.clone();
            let current = self.strand.vert_pos.clone();
            self.strand.vert_pos = omega * (current - &prevprev) + prevprev;
        }
    }

    // -----------------------------------------------------------------------
    // Simulation steps
    // -----------------------------------------------------------------------

    /// Apply gravity, compute inertia target, and optionally warm-start.
    pub fn forward_step(&mut self) {
        let dt = self.params.dt;
        let gravity = self.params.gravity;

        // Gravity on all vertices
        for j in 0..self.strand.velocity.ncols() {
            let v = self.strand.velocity.column(j) + dt * gravity;
            self.strand.velocity.column_mut(j).copy_from(&v);
        }
        // Pin vertex 0
        self.strand.velocity.column_mut(0).fill(0.0);

        // Inertia target: x̃ = x + v·dt
        self.strand.inertia = &self.strand.vert_pos + &self.strand.velocity * dt;

        // Save previous positions
        self.strand.vert_prev_pos = self.strand.vert_pos.clone();

        // Warm start with acceleration approximation
        if self.strand.has_velocities_prev {
            let dt_inv = 1.0 / dt;
            let approx_accel = (&self.strand.velocity - &self.strand.velocities_prev) * dt_inv;

            let grav_norm: FloatingType = 10.0;
            let grav_dir = Vector3::new(0.0_f32, -1.0, 0.0);

            let prev_pos = self.strand.vert_prev_pos.clone();
            let vel_prev = self.strand.velocities_prev.clone();
            let num_verts = self.strand.num_verts;

            for iv in 0..num_verts {
                let accel_col = approx_accel.column(iv);
                let mut component = accel_col.dot(&grav_dir);
                if component > grav_norm {
                    component = grav_norm;
                }
                if component < 1e-5 {
                    component = 0.0;
                }

                let new_pos =
                    prev_pos.column(iv) + dt * vel_prev.column(iv) + dt * dt * grav_dir * component;
                self.strand.vert_pos.column_mut(iv).copy_from(&new_pos);
            }
            self.strand.has_approx_acceleration = true;
        } else {
            // Fallback: start from inertia
            self.strand.vert_pos = self.strand.inertia.clone();
        }
    }

    /// Vertex Block Descent solve — one Gauss–Seidel pass over all free vertices.
    pub fn solve(&mut self) {
        let dt = self.params.dt;
        let dt_sqr_recip = 1.0 / (dt * dt);

        let edges = self.strand.edges.clone();
        let num_verts = self.strand.num_verts;

        for iv in 1..num_verts {
            let mass_iv = self.strand.vertex_mass[iv];
            let inertia_col = self.strand.inertia.column(iv).into_owned();
            let pos_col_iv = self.strand.vert_pos.column(iv).into_owned();

            // Inertia term
            let mut f: Vec3 = mass_iv * (inertia_col - pos_col_iv) * dt_sqr_recip;
            let mut h: Mat3 = Mat3::identity() * (mass_iv * dt_sqr_recip);

            // Spring contributions
            let adj = self.strand.vert_adjacent_edges[iv].clone();
            for &edge_id in &adj {
                if edge_id == -1 {
                    continue;
                }
                let eid = edge_id as usize;
                let v1 = edges[eid][0] as usize;
                let v2 = edges[eid][1] as usize;

                let p1 = self.strand.vert_pos.column(v1).into_owned();
                let p2 = self.strand.vert_pos.column(v2).into_owned();

                let diff: Vec3 = p1 - p2;
                let l = diff.norm();
                let l0 = self.strand.org_lengths[eid];
                let k = self.strand.edges_stiffness[eid];

                // Hessian: k * (I - (l0/l)*(I - d⊗d/l²))
                let outer = diff * diff.transpose() / (l * l);
                let h_contrib = k * (Mat3::identity() - (l0 / l) * (Mat3::identity() - outer));
                h += h_contrib;

                // Force: k * (l0 - l) / l * diff  (sign depends on which end)
                let force_mag = k * (l0 - l) / l;
                if v1 == iv {
                    f += force_mag * diff;
                } else {
                    f -= force_mag * diff;
                }
            }

            // Solve 3×3 system: h·dx = f
            let decomp = h.lu();
            if let Some(dx) = decomp.solve(&f) {
                let new_pos = self.strand.vert_pos.column(iv).into_owned() + dx;
                self.strand.vert_pos.column_mut(iv).copy_from(&new_pos);
            }
        }
    }

    /// Update velocities at end of substep.
    pub fn update_velocity(&mut self) {
        let dt = self.params.dt;
        self.strand.velocities_prev = self.strand.velocity.clone();
        self.strand.velocity = (&self.strand.vert_pos - &self.strand.vert_prev_pos) / dt;
        self.strand.velocity.column_mut(0).fill(0.0);
        self.strand.has_velocities_prev = true;
    }

    // -----------------------------------------------------------------------
    // Main simulation loop
    // -----------------------------------------------------------------------
    pub fn simulate(&mut self) {
        self.save_outputs();
        self.params.dt /= self.params.substeps as FloatingType;

        let mut prev_iter_pos = vertices_mat_zeros(self.strand.num_verts);

        for frame in 1..self.params.num_frames {
            self.frame_id = frame;

            for s in 0..self.params.substeps {
                self.step = s;
                self.forward_step();

                let mut omega = 1.0_f32;

                for it in 0..self.params.num_iterations {
                    self.iter = it;
                    prev_iter_pos = self.strand.vert_pos.clone();
                    self.solve();

                    if self.params.use_acceleration {
                        omega = Self::get_accelerator_omega(
                            it + 1,
                            self.params.acceleration_rho,
                            omega,
                        );
                        self.apply_accelerator(omega);
                        self.strand.prevprev_pos = prev_iter_pos.clone();
                    }
                }

                self.update_velocity();
            }

            println!("Frame: {} finished!", frame);
            self.save_outputs();
        }
    }

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------

    /// Write vertex positions as a JSON file for the current frame.
    pub fn save_outputs(&self) {
        let verts: Vec<[FloatingType; 3]> = (0..self.strand.num_verts)
            .map(|iv| {
                let col = self.strand.vert_pos.column(iv);
                [col[0], col[1], col[2]]
            })
            .collect();

        let frame_str = format!("{:08}", self.frame_id);
        let out_file = format!("{}/A{}.json", self.out_path, frame_str);

        let json_value = serde_json::json!({ "pos": verts });
        let json_str = serde_json::to_string_pretty(&json_value).expect("Failed to serialise JSON");

        let mut f = fs::File::create(&out_file)
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", out_file, e));
        writeln!(f, "{}", json_str)
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_file, e));
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
fn main() {
    let mut simulator = StrandSim::new();
    simulator.initialize();
    simulator.simulate();
}
