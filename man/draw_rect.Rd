% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/drawing.R
\name{draw_rect}
\alias{draw_rect}
\title{Draw rectangle on image}
\usage{
draw_rect(im, x0, y0, x1, y1, color = "white", opacity = 1, filled = TRUE)
}
\arguments{
\item{im}{an image}

\item{x0}{x coordinate of the bottom-left corner}

\item{y0}{y coordinate  of the bottom-left corner}

\item{x1}{x coordinate of the top-right corner}

\item{y1}{y coordinate  of the top-right corner}

\item{color}{either a vector, or a string (e.g. "blue")}

\item{opacity}{0: transparent 1: opaque.}

\item{filled}{fill rectangle (default TRUE)}
}
\value{
an image
}
\description{
Add a rectangle to an image. Like other native CImg drawing functions, this is meant to be basic but fast. Use implot for flexible drawing.
}
\examples{
\dontshow{cimg.limit.openmp()}
draw_rect(boats,1,1,50,50,"darkgreen") \%>\% plot
}
\seealso{
implot,draw_circle
}
\author{
Simon Barthelme
}
