# Generate corr2D reference fixtures for crosspeak's cross-implementation test.
#
# Provenance. Run once, by hand; the CSV outputs are committed. CI never runs R.
#
#   corr2D 1.0.3 (Geitner, Fritzsch, Bocklitz, Popp), GPL-3, CRAN.
#   Source: https://github.com/cran/corr2D  (CRAN mirror, commit pinned below)
#   R 4.3.3
#
# corr2D computes the complex correlation matrix by FFT and normalises with
#   Norm = 1 / (pi * (m - 1))     [corr2d.R, default arg]
# whereas Noda's discrete form (and crosspeak) uses 1 / (m - 1). Parseval
# introduces a further factor of m, so
#
#   Re(corr2D$FT) = (m / pi) * Phi_Noda
#
# and the fixtures below are pre-scaled by pi/m so they are directly comparable
# to crosspeak's output. See tests/test_validation.py for the asynchronous case,
# which is NOT related by a scalar.
#
# Requires: foreach, doParallel (pure R, no compilation).

library(foreach); library(doParallel); library(parallel)
source("corr2D/R/corr2d.R")
source("corr2D/R/sim2ddata.R")

# sim2ddata: A -> B -> C first-order kinetics, k1 = 0.2, k2 = 0.8,
# two Gaussians per species. L reduced from 400 to 40 to keep the fixture small.
# t = 0:10 gives m = 11, an ODD number of perturbation points -- required for
# exact agreement (see the Nyquist note in test_validation.py).
d <- sim2ddata(L = 40, t = 0:10)
m <- nrow(d)
r <- corr2d(d, corenumber = 1)

write.csv(d, "sim2d_input.csv", row.names = FALSE)
write.csv(Re(r$FT) * pi / m, "sim2d_sync_noda.csv", row.names = FALSE)
write.csv(Im(r$FT) * pi / m, "sim2d_async_corr2d_scaled.csv", row.names = FALSE)

cat("m =", m, " input", dim(d), " FT", dim(r$FT), "\n")
