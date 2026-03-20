# Local Eigen (standalone)

This project now expects Eigen headers from this local folder (`external/eigen`) instead of a system-wide `Eigen` install.

Place the upstream Eigen header tree here before building:

- `external/eigen/Eigen/...`
- `external/eigen/unsupported/...` (optional, not currently required)

Suggested source: Eigen 3.4.x release archive.
