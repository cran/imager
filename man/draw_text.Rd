% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/drawing.R
\name{draw_text}
\alias{draw_text}
\title{Draw text on an image}
\usage{
draw_text(im, x, y, text, color, opacity = 1, fsize = 20)
}
\arguments{
\item{im}{an image}

\item{x}{x coord.}

\item{y}{y coord.}

\item{text}{text to draw (a string)}

\item{color}{either a vector or a string (e.g. "red")}

\item{opacity}{0: transparent 1: opaque.}

\item{fsize}{font size (in pix., default 20)}
}
\value{
an image
}
\description{
Like other native CImg drawing functions, this is meant to be basic but fast. Use implot for flexible drawing.
}
\examples{
\dontshow{cimg.limit.openmp()}
draw_text(boats,100,100,"Some text",col="black") \%>\% plot
}
\seealso{
implot,draw_circle,draw_rect
}
\author{
Simon Barthelme
}
