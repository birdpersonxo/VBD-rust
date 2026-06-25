// types.rs — port of Types.h
// Uses nalgebra in place of Eigen.

pub use nalgebra::{
    DMatrix, DVector, Dynamic, Matrix2, Matrix3, Matrix4, SMatrix, Vector2, Vector3, Vector4,
    Vector6,
};

// ---------------------------------------------------------------------------
// Scalar aliases
// ---------------------------------------------------------------------------
pub type FloatingType = f32;
pub type IdType = i32;

// ---------------------------------------------------------------------------
// Fixed-size vectors
// ---------------------------------------------------------------------------
pub type Vec2 = nalgebra::Vector2<FloatingType>;
pub type Vec2I = nalgebra::Vector2<IdType>;
pub type Vec3 = nalgebra::Vector3<FloatingType>;
pub type Vec4 = nalgebra::Vector4<FloatingType>;
pub type Vec6 = nalgebra::SVector<FloatingType, 6>;
pub type Vec9 = nalgebra::SVector<FloatingType, 9>;
pub type Vec12 = nalgebra::SVector<FloatingType, 12>;
pub type Vec3d = nalgebra::Vector3<f64>;
pub type Vec3I = nalgebra::Vector3<IdType>;

// ---------------------------------------------------------------------------
// Fixed-size matrices
// ---------------------------------------------------------------------------
pub type Mat2 = nalgebra::Matrix2<FloatingType>;
pub type Mat3 = nalgebra::Matrix3<FloatingType>;
pub type Mat4 = nalgebra::Matrix4<FloatingType>;
pub type Mat9 = nalgebra::SMatrix<FloatingType, 9, 9>;
pub type Mat12 = nalgebra::SMatrix<FloatingType, 12, 12>;
pub type Mat3x2 = nalgebra::SMatrix<FloatingType, 3, 2>;
pub type Mat3d = nalgebra::Matrix3<f64>;

// ---------------------------------------------------------------------------
// Dynamic types
// ---------------------------------------------------------------------------

/// Column-major matrix with 3 rows and dynamic number of columns.
/// Equivalent to `Eigen::Matrix<FloatingType, 3, Dynamic>` (TVerticesMat).
pub type TVerticesMat = nalgebra::Matrix<
    FloatingType,
    nalgebra::Const<3>,
    nalgebra::Dynamic,
    nalgebra::VecStorage<FloatingType, nalgebra::Const<3>, nalgebra::Dynamic>,
>;

pub type VecDynamic = DVector<FloatingType>;
pub type VecDynamicI = DVector<IdType>;

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Allocate a TVerticesMat of `n` columns, all zeros.
pub fn vertices_mat_zeros(n: usize) -> TVerticesMat {
    TVerticesMat::zeros(n)
}

// ---------------------------------------------------------------------------
// Small helpers (mirror of VBD::Utility namespace)
// ---------------------------------------------------------------------------
pub mod utility {
    use super::VecDynamic;

    pub fn print_vec_info(input: &VecDynamic, name: &str) {
        let abs: VecDynamic = input.abs();
        println!(
            "{}: max {} | min {} | mean {}",
            name,
            abs.max(),
            abs.min(),
            abs.mean()
        );
    }

    pub fn is_zero_approx(s: f64) -> bool {
        s.abs() < 1e-5
    }
}

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------
#[macro_export]
macro_rules! sqr {
    ($x:expr) => {
        ($x) * ($x)
    };
}
