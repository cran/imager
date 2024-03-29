% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/imgen.R
\name{imnoise}
\alias{imnoise}
\title{Generate (Gaussian) white-noise image}
\usage{
imnoise(x = 1, y = 1, z = 1, cc = 1, mean = 0, sd = 1, dim = NULL)
}
\arguments{
\item{x}{width}

\item{y}{height}

\item{z}{depth}

\item{cc}{spectrum}

\item{mean}{mean pixel value (default 0)}

\item{sd}{std. deviation of pixel values (default 1)}

\item{dim}{dimension vector (optional, alternative to specifying x,y,z,cc)}
}
\value{
a cimg object
}
\description{
A white-noise image is an image where all pixel values are drawn IID from a certain distribution. Here they are drawn from a Gaussian.
}
\examples{
\dontshow{cimg.limit.openmp()}
imnoise(100,100,cc=3) \%>\% plot(main="White noise in RGB")
imnoise(100,100,cc=3) \%>\% isoblur(5) \%>\% plot(main="Filtered (non-white) noise")
imnoise(dim=dim(boats)) #Noise image of the same size as the boats image
}
\author{
Simon Barthelme
}
