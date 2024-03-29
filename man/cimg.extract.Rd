% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cimg_class.R
\name{cimg.extract}
\alias{cimg.extract}
\alias{frame}
\alias{imcol}
\alias{imrow}
\alias{channel}
\alias{R}
\alias{G}
\alias{B}
\title{Various shortcuts for extracting colour channels, frames, etc}
\usage{
frame(im, index)

imcol(im, x)

imrow(im, y)

channel(im, ind)

R(im)

G(im)

B(im)
}
\arguments{
\item{im}{an image}

\item{index}{frame index}

\item{x}{x coordinate of the row}

\item{y}{y coordinate of the row}

\item{ind}{channel index}
}
\description{
Various shortcuts for extracting colour channels, frames, etc

Extract one frame out of a 4D image/video
}
\section{Functions}{
\itemize{
\item \code{frame()}: Extract frame

\item \code{imcol()}: Extract a particular column from an image

\item \code{imrow()}: Extract a particular row from an image

\item \code{channel()}: Extract an image channel

\item \code{R()}: Extract red channel

\item \code{G()}: Extract green channel

\item \code{B()}: Extract blue channel

}}
\examples{
\dontshow{cimg.limit.openmp()}
#Extract the red channel from the boats image, then the first row, plot
rw <- R(boats) \%>\% imrow(10)
plot(rw,type="l",xlab="x",ylab="Pixel value")
#Note that R(boats) returns an image
R(boats)
#while imrow returns a vector or a list
R(boats) \%>\% imrow(1) \%>\% str
imrow(boats,1) \%>\% str

}
\author{
Simon Barthelme
}
