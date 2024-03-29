% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{blur_anisotropic}
\alias{blur_anisotropic}
\title{Blur image anisotropically, in an edge-preserving way.}
\usage{
blur_anisotropic(
  im,
  amplitude,
  sharpness = 0.7,
  anisotropy = 0.6,
  alpha = 0.6,
  sigma = 1.1,
  dl = 0.8,
  da = 30,
  gauss_prec = 2,
  interpolation_type = 0L,
  fast_approx = TRUE
)
}
\arguments{
\item{im}{an image}

\item{amplitude}{Amplitude of the smoothing.}

\item{sharpness}{Sharpness.}

\item{anisotropy}{Anisotropy.}

\item{alpha}{Standard deviation of the gradient blur.}

\item{sigma}{Standard deviation of the structure tensor blur.}

\item{dl}{Spatial discretization.}

\item{da}{Angular discretization.}

\item{gauss_prec}{Precision of the diffusion process.}

\item{interpolation_type}{Interpolation scheme.
Can be 0=nearest-neighbor | 1=linear | 2=Runge-Kutta}

\item{fast_approx}{If true, use fast approximation (default TRUE)}
}
\description{
Standard blurring removes noise from images, but tends to smooth away edges in the process. This anisotropic filter preserves edges better.
}
\examples{
\dontshow{cimg.limit.openmp()}
im <- load.image(system.file('extdata/Leonardo_Birds.jpg',package='imager'))
im.noisy <- (im + 80*rnorm(prod(dim(im)))) 
blur_anisotropic(im.noisy,ampl=1e4,sharp=1) \%>\% plot
}
