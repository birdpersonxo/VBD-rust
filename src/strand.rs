// strand.rs — port of Strand.h
// Implements the `Mesh` base data and `VBDStrand` simulation object.

use crate::types::*;

// ---------------------------------------------------------------------------
// Mesh — base vertex data (mirrors struct Mesh in Strand.h)
// ---------------------------------------------------------------------------
pub struct Mesh {
    pub vert_pos: TVerticesMat,      // current positions  (3 × n)
    pub vert_prev_pos: TVerticesMat, // previous positions (3 × n)
    pub velocity: TVerticesMat,      // velocities         (3 × n)
    pub vertex_mass: VecDynamic,     // per-vertex masses  (n)
}

impl Mesh {
    pub fn new_empty() -> Self {
        Self {
            vert_pos: vertices_mat_zeros(0),
            vert_prev_pos: vertices_mat_zeros(0),
            velocity: vertices_mat_zeros(0),
            vertex_mass: VecDynamic::zeros(0),
        }
    }

    /// Return the 3-D position of vertex `i` as a column slice.
    #[inline]
    pub fn vertex(&self, i: usize) -> nalgebra::VectorSlice3<FloatingType> {
        self.vert_pos.column(i)
    }

    /// Return a mutable view of vertex `i`.
    #[inline]
    pub fn vertex_mut(&mut self, i: usize) -> nalgebra::VectorSliceMut3<FloatingType> {
        self.vert_pos.column_mut(i)
    }
}

// ---------------------------------------------------------------------------
// VBDStrand — the strand simulation object (mirrors struct VBDStrand)
// ---------------------------------------------------------------------------
pub struct VBDStrand {
    // --- inherited from Mesh ---
    pub vert_pos: TVerticesMat,
    pub vert_prev_pos: TVerticesMat,
    pub velocity: TVerticesMat,
    pub vertex_mass: VecDynamic,

    // --- strand-specific ---
    /// Each edge is a pair of vertex indices [v0, v1].
    pub edges: Vec<[IdType; 2]>,
    /// Per-edge stiffness values.
    pub edges_stiffness: Vec<FloatingType>,
    /// For each vertex, the list of adjacent edge indices.
    pub vert_adjacent_edges: Vec<Vec<i32>>,
    /// Total number of edges.
    pub n_edges: usize,
    /// Uniform stiffness (used by the first `from` overload).
    pub stiffness: FloatingType,
    /// Skip-spring stiffness.
    pub stiffness_skip_spring: FloatingType,
    /// Rest lengths of every edge (including skip springs).
    pub org_lengths: VecDynamic,

    /// Previous-step velocities (for the warm-start heuristic).
    pub velocities_prev: TVerticesMat,
    /// Inertia target positions (x̃ in the VBD paper).
    pub inertia: TVerticesMat,
    /// Total number of vertices.
    pub num_verts: usize,

    /// Positions from the iteration *before* the previous one (Chebyshev accelerator).
    pub prevprev_pos: TVerticesMat,
    pub has_velocities_prev: bool,
    pub has_approx_acceleration: bool,
}

impl VBDStrand {
    pub fn new_empty() -> Self {
        Self {
            vert_pos: vertices_mat_zeros(0),
            vert_prev_pos: vertices_mat_zeros(0),
            velocity: vertices_mat_zeros(0),
            vertex_mass: VecDynamic::zeros(0),
            edges: Vec::new(),
            edges_stiffness: Vec::new(),
            vert_adjacent_edges: Vec::new(),
            n_edges: 0,
            stiffness: 0.0,
            stiffness_skip_spring: 0.0,
            org_lengths: VecDynamic::zeros(0),
            velocities_prev: vertices_mat_zeros(0),
            inertia: vertices_mat_zeros(0),
            num_verts: 0,
            prevprev_pos: vertices_mat_zeros(0),
            has_velocities_prev: false,
            has_approx_acceleration: false,
        }
    }

    // -----------------------------------------------------------------------
    // Mesh helpers (replicated here because Rust has no inheritance)
    // -----------------------------------------------------------------------

    #[inline]
    pub fn vertex(&self, i: usize) -> nalgebra::VectorSlice3<FloatingType> {
        self.vert_pos.column(i)
    }

    #[inline]
    pub fn vertex_mut(&mut self, i: usize) -> nalgebra::VectorSliceMut3<FloatingType> {
        self.vert_pos.column_mut(i)
    }

    // -----------------------------------------------------------------------
    // from() — uniform stiffness, optional skip springs
    // -----------------------------------------------------------------------
    /// Initialise the strand from explicit vertex positions, masses, rest lengths
    /// and a single stiffness value.  Optionally adds skip-one springs.
    pub fn from_uniform(
        &mut self,
        pos: &[Vec3],
        masses: &[FloatingType],
        lengths: &[FloatingType],
        stiffness_in: FloatingType,
        add_skip_spring: bool,
        stiffness_skip_spring_in: FloatingType,
    ) {
        assert_eq!(masses.len(), lengths.len() + 1);
        assert_eq!(pos.len(), lengths.len() + 1);

        self.stiffness = stiffness_in;
        self.stiffness_skip_spring = stiffness_skip_spring_in;
        self.num_verts = pos.len();

        self.vertex_mass = VecDynamic::from_vec(masses.to_vec());
        self.vert_adjacent_edges = vec![Vec::new(); pos.len()];
        self.edges.clear();
        self.edges_stiffness.clear();

        let mut lengths_new: Vec<FloatingType> = Vec::new();

        for i in 0..lengths.len() {
            let edge_id = self.edges.len() as i32;
            self.edges.push([i as IdType, (i + 1) as IdType]);

            self.vert_adjacent_edges[i].push(edge_id);
            self.vert_adjacent_edges[i + 1].push(edge_id);
            self.edges_stiffness.push(stiffness_in);
            lengths_new.push(lengths[i]);

            if add_skip_spring && i < lengths.len() - 1 {
                let skip_edge_id = self.edges.len() as i32;
                self.edges.push([i as IdType, (i + 2) as IdType]);
                self.vert_adjacent_edges[i].push(skip_edge_id);
                self.vert_adjacent_edges[i + 2].push(skip_edge_id);
                self.edges_stiffness.push(stiffness_skip_spring_in);
                lengths_new.push(lengths[i] + lengths[i + 1]);
            }
        }

        self.n_edges = self.edges.len();
        self.org_lengths = VecDynamic::from_vec(lengths_new);
        self.init_positions(pos);
    }

    // -----------------------------------------------------------------------
    // from() — per-edge stiffness vector (no skip springs)
    // -----------------------------------------------------------------------
    /// Initialise the strand with a per-edge stiffness vector.
    pub fn from_per_edge(
        &mut self,
        pos: &[Vec3],
        masses: &[FloatingType],
        lengths: &[FloatingType],
        stiffness_in: Vec<FloatingType>,
    ) {
        assert_eq!(masses.len(), lengths.len() + 1);
        assert_eq!(pos.len(), lengths.len() + 1);

        self.num_verts = pos.len();
        self.vertex_mass = VecDynamic::from_vec(masses.to_vec());
        // NOTE: the C++ code sets edgesStiffness = stiffness_in first, then
        // appends once more inside the loop (looks like a bug in the original,
        // but we replicate the structure faithfully here with the per-edge
        // stiffnesses driving the loop).
        self.edges_stiffness = stiffness_in;
        self.vert_adjacent_edges = vec![Vec::new(); pos.len()];
        self.edges.clear();

        let mut lengths_new: Vec<FloatingType> = Vec::new();

        for i in 0..lengths.len() {
            let edge_id = self.edges.len() as i32;
            self.edges.push([i as IdType, (i + 1) as IdType]);

            self.vert_adjacent_edges[i].push(edge_id);
            self.vert_adjacent_edges[i + 1].push(edge_id);
            // Push stiffness for this edge (mirrors the C++ push_back in the loop
            // after the initial assignment, which effectively appends a duplicate
            // for each edge using the class-level `stiffness` field — left as-is
            // to match the original behaviour; callers should use from_uniform
            // for the variable-stiffness path).
            self.edges_stiffness.push(self.stiffness);
            lengths_new.push(lengths[i]);
        }

        self.n_edges = self.edges.len();
        self.org_lengths = VecDynamic::from_vec(lengths_new);
        self.init_positions(pos);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn init_positions(&mut self, pos: &[Vec3]) {
        let n = pos.len();
        self.vert_pos = vertices_mat_zeros(n);
        for (i, p) in pos.iter().enumerate() {
            self.vert_pos.column_mut(i).copy_from(p);
        }
        self.vert_prev_pos = self.vert_pos.clone();
        self.velocity = vertices_mat_zeros(n);
        self.velocities_prev = vertices_mat_zeros(n);
        self.inertia = vertices_mat_zeros(n);
        self.prevprev_pos = vertices_mat_zeros(n);
    }
}
