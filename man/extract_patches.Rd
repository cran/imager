% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{extract_patches}
\alias{extract_patches}
\alias{extract_patches3D}
\title{Extract image patches and return a list}
\usage{
extract_patches(im, cx, cy, wx, wy, boundary_conditions = 0L)

extract_patches3D(im, cx, cy, cz, wx, wy, wz, boundary_conditions = 0L)
}
\arguments{
\item{im}{an image}

\item{cx}{vector of x coordinates for patch centers}

\item{cy}{vector of y coordinates for patch centers}

\item{wx}{vector of patch widths (or single value)}

\item{wy}{vector of patch heights (or single value)}

\item{boundary_conditions}{integer. Can be 0 (Dirichlet, default), 1 (Neumann) 2 (Periodic) 3 (mirror).}

\item{cz}{vector of z coordinates for patch centers}

\item{wz}{vector of coordinates for patch depth}
}
\value{
a list of image patches (cimg objects)
}
\description{
Patches are rectangular (cubic) image regions centered at cx,cy (cz) with width wx and height wy (opt. depth wz)
WARNINGS: 
- values outside of the image region are subject to boundary conditions. The default is to set them to 0 (Dirichlet), other boundary conditions are listed below. 
- widths and heights should be odd integers (they're rounded up otherwise).
}
\section{Functions}{
\itemize{
\item \code{extract_patches3D()}: Extract 3D patches

}}
\examples{
\dontshow{cimg.limit.openmp()}
#2 patches of size 5x5 located at (10,10) and (10,20)
extract_patches(boats,c(10,10),c(10,20),5,5)
}
