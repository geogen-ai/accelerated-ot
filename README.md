# accelerated-ot

Fast optimal transport in Python, with a focus on partial, Gromov-Wasserstein, sliced, and linear-embedding methods.

> **Status:** early design. No code has been released yet.

The design is in [docs/superpowers/specs/2026-11-01-accelerated-ot-design.md](docs/superpowers/specs/2026-11-01-accelerated-ot-design.md).

## Planned modules

| Module | Methods |
|---|---|
| `partial` | Partial OT, 1D partial OT |
| `sliced` | Sliced OT, sliced partial OT |
| `gromov` | Partial GW, fused partial GW, barycenters |
| `linear` | LOT, LOPT, LGW, LPGW |
| `registration` | OPT and SOPT point-cloud registration |

## License

MIT. See [LICENSE](LICENSE).
